from collections import defaultdict
from typing import Iterable, Iterator

import torch
from compressed_tensors.base import QUANTIZATION_CONFIG_NAME, TRANSFORM_CONFIG_NAME
from compressed_tensors.compressors import compress_module
from compressed_tensors.entrypoints.convert import Converter
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from compressed_tensors.utils import match_quantizable_tensors
from loguru import logger

from llmcompressor.entrypoints.model_free.lifecycle import (
    calibrate_weight,
    initialize_quantized_linear,
    validate_weight_for_quantization,
)
from llmcompressor.entrypoints.model_free.microscale import (
    get_fused_names,
    has_microscale_scheme,
    is_microscale_scheme,
)
from llmcompressor.modifiers.quantization.calibration import (
    apply_calibration_status,
    freeze_module_quantization,
    initialize_observer,
    observe,
    update_qparams,
)
from llmcompressor.observers import FusionHandler

__all__ = ["ModelFreePtqConverter", "DEFAULT_IGNORE"]

# Sensitive layers are ignored by default, in addition to any user-provided
# ignore list:
#
# - MoE routers are small (num_experts x hidden_size) and sensitive:
#   quantizing the routing logits degrades expert selection and downstream
#   accuracy
# - lm_head and embeddings are vocabulary-shaped and sensitive to
#   quantization. Ignoring both also keeps tied lm_head/embed_tokens weights
#   consistent when a checkpoint stores both copies
#
# Model-based quantization excludes routers and embeddings implicitly,
# because they are not nn.Linear modules (e.g. transformers' MoE routers hold
# a bare nn.Parameter), and lm_head is conventionally ignored by users (see
# docs/faq/faq.md). Model-free matching sees tensor names only and its
# default "Linear" target is all-inclusive, so all of these must be ignored
# explicitly here.
#
# Patterns are matched against module names (see match_quantizable_tensors)
# and only match a bare `gate`/`router` submodule, so similarly named gate
# projections (`gate_proj`, `gate_up_proj`, `shared_expert_gate`,
# `forget_gate`) remain quantizable
DEFAULT_IGNORE = (
    # MoE routers
    r"re:.*mlp\.gate$",  # Qwen, DeepSeek, GLM, Kimi, Cohere, OLMoE, ...
    r"re:.*block_sparse_moe\.gate$",  # Mixtral, Kimi K3, dbrx, Phi, ...
    r"re:.*moe\.gate$",  # Step3.5, Grok
    r"re:.*mixer\.gate$",  # Nemotron
    r"re:.*router$",  # Llama 4 (feed_forward.router), gpt-oss, Trinity
    # lm_head and token embeddings, at any depth
    r"re:.*lm_head$",  # lm_head (also nested, e.g. model.lm_head)
    r"re:.*embed_tokens$",  # model/backbone embed_tokens, MTP embed_tokens
)


def _merge_default_ignore(config: QuantizationConfig) -> QuantizationConfig:
    """
    Return a copy of `config` with any missing DEFAULT_IGNORE patterns appended
    to its ignore list, so that MoE routers, lm_head and embedding layers are
    excluded from quantization. The input config is left untouched, and
    patterns already present are not duplicated
    """
    ignore = list(config.ignore) if config.ignore else []
    additions = [pattern for pattern in DEFAULT_IGNORE if pattern not in ignore]
    if not additions:
        return config

    logger.info(
        f"Ignoring sensitive layers by default: {additions}. MoE routers, "
        "lm_head and embedding layers are kept in their original precision"
    )
    return config.model_copy(update={"ignore": ignore + additions})


def _match_tensors_to_schemes(
    tensors: dict[str, torch.Tensor],
    config: QuantizationConfig,
) -> Iterator[tuple[str, str, QuantizationScheme]]:
    """Match each quantizable tensor to its scheme from the config.

    Iterates config groups in order. A tensor matched by an earlier group
    is not re-yielded for later groups (first match wins).
    """
    matched: set[str] = set()
    for scheme in config.config_groups.values():
        for module_name, name in match_quantizable_tensors(
            tensors, config.ignore, scheme.targets
        ):
            if name not in matched:
                matched.add(name)
                yield module_name, name, scheme


class ModelFreePtqConverter(Converter):
    """
    Converter that quantizes and compresses safetensors checkpoints without
    loading a model. Implements the compressed-tensors Converter protocol so
    it can be chained with other converters (e.g. a dequantizer) and used
    with convert_checkpoint. For microscale schemes, it needs the checkpoint's
    `weight_names` to resolve fused weights; `model_free_ptq` passes them.

    MoE routers, lm_head and embedding layers are ignored by default (see
    DEFAULT_IGNORE), in addition to the config's ignore list, because they
    are sensitive to quantization. The default patterns are merged into the
    config so they are respected during matching and recorded in the saved
    quantization config
    """

    def __init__(
        self,
        config: QuantizationConfig,
        weight_names: Iterable[str] | None = None,
    ):
        """
        :param config: quantization config to apply. The DEFAULT_IGNORE
            patterns, which exclude MoE routers, lm_head and embedding
            layers, are merged into the config's ignore list
        :param weight_names: names of all tensors in the checkpoint. Required
            to resolve fused weights for microscale schemes, because a layer's
            fused group depends on its sibling layers. For example, Kimi
            `g_proj` is fused with delta attention q/k/v/b/f_a projections on
            some layers and with MLA q_a/kv_a projections on others
        """
        self.config = _merge_default_ignore(config)

        # primary weight name -> names of the fused partners it loads
        self._fused_partners: dict[str, set[str]] | None = None
        if weight_names is not None:
            self._fused_partners = {}
            for fused_group in get_fused_names(weight_names):
                primary, *partners = fused_group.values()
                self._fused_partners[primary] = set(partners)

    def get_dependencies(self, weight_name: str) -> set[str]:
        """
        For microscale schemes, return fused partner tensor names that must be
        loaded together to compute a shared global scale. Only the primary
        weight of a fused group has dependencies, so the shard owning the
        primary loads the whole group. Standard schemes have no cross-tensor
        dependencies.
        """
        if not has_microscale_scheme(self.config):
            return set()

        if self._fused_partners is None:
            raise ValueError(
                "Resolving fused weights for microscale schemes requires the "
                "checkpoint's weight names. Please construct "
                "ModelFreePtqConverter with `weight_names`"
            )

        return set(self._fused_partners.get(weight_name, ()))

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Quantize and compress all tensors. Device is inferred from the tensors."""
        tensors = split_fused_moe_experts(tensors)
        if has_microscale_scheme(self.config):
            return self._process_microscale(tensors)
        return self._process_standard(tensors)

    def validate(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Validate tensor shapes and simulate the compressed state-dict layout.

        Converter validation runs on meta tensors before conversion. Some
        calibration observers need tensor values, so validation initializes each
        quantized module with meta tensors and compresses it without calibration.
        This verifies the same shape and compression compatibility checks as
        :meth:`process` while leaving observers to run only during the real
        conversion passes.
        """
        tensors = split_fused_moe_experts(tensors)
        output = dict(tensors)
        for module_name, name, scheme in _match_tensors_to_schemes(
            tensors, self.config
        ):
            validate_weight_for_quantization(tensors[name], scheme, name)
            module = initialize_quantized_linear(
                tensors[name], scheme, torch.device("meta")
            )
            compress_module(module)

            del output[name]
            prefix = module_name + "."
            for key, value in module.state_dict(prefix=prefix).items():
                output[key] = value

        return output

    def update_config(self, config: QuantizationConfig | None) -> QuantizationConfig:
        """
        Build or merge the final QuantizationConfig in COMPRESSED status.
        When chained after a dequantizer (config=None), creates from scratch.
        When seeded with a pre-existing checkpoint config, merges into it.
        """
        unique_formats = {
            s.format for s in self.config.config_groups.values() if s.format is not None
        }
        if len(unique_formats) == 1:
            format_ = next(iter(unique_formats))
        elif len(unique_formats) > 1:
            format_ = "mixed-precision"
        else:
            format_ = "dense"

        new_config = QuantizationConfig(
            config_groups=self.config.config_groups,
            kv_cache_scheme=self.config.kv_cache_scheme,
            ignore=list(self.config.ignore) if self.config.ignore else [],
            quantization_status=QuantizationStatus.COMPRESSED,
            format=format_,
        )
        if config is None:
            return new_config
        config.merge(new_config)
        return config

    def update_model_config(self, model_config: dict) -> dict:
        quant_config = model_config.get(QUANTIZATION_CONFIG_NAME)
        if isinstance(quant_config, dict):
            quant_config.setdefault(TRANSFORM_CONFIG_NAME, {})
        return model_config

    def _process_standard(
        self, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        device = _infer_device(tensors)
        for module_name, name, scheme in _match_tensors_to_schemes(
            tensors, self.config
        ):
            validate_weight_for_quantization(tensors[name], scheme, name)
            module = initialize_quantized_linear(tensors[name], scheme, device)
            calibrate_weight(module)
            compress_module(module)
            del tensors[name]
            prefix = module_name + "."
            for key, value in module.state_dict(prefix=prefix).items():
                tensors[key] = value
        return tensors

    def _process_microscale(
        self, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        device = _infer_device(tensors)

        fused_groups = get_fused_names(tensors.keys())
        fused_name_to_fused_index: dict[str, int] = {
            name: index
            for index, fused_group in enumerate(fused_groups)
            for name in fused_group.values()
        }
        fused_modules: dict[int, dict[str, torch.nn.Module]] = defaultdict(dict)

        for module_name, name, scheme in _match_tensors_to_schemes(
            tensors, self.config
        ):
            validate_weight_for_quantization(tensors[name], scheme, name)
            module = initialize_quantized_linear(tensors[name], scheme, device)

            if is_microscale_scheme(scheme) and name in fused_name_to_fused_index:
                fused_index = fused_name_to_fused_index[name]
                fused_modules[fused_index][name] = module
                initialize_observer(module, "weight")
                apply_calibration_status(module)
                continue

            calibrate_weight(module)
            compress_module(module)
            del tensors[name]
            prefix = module_name + "."
            for key, value in module.state_dict(prefix=prefix).items():
                tensors[key] = value

        for named_modules in fused_modules.values():
            FusionHandler.fuse(
                [(mod.weight_observer, mod) for mod in named_modules.values()]
            )
            observe(named_modules.values(), base_name="weight")
            update_qparams(named_modules.values(), base_name="weight")
            for name, module in named_modules.items():
                freeze_module_quantization(module)
                compress_module(module)
                del tensors[name]
                module_name, _ = name.rsplit(".", 1)
                prefix = module_name + "."
                for key, value in module.state_dict(prefix=prefix).items():
                    tensors[key] = value

        return tensors


def _infer_device(tensors: dict[str, torch.Tensor]) -> torch.device:
    for t in tensors.values():
        return t.device
    return torch.device("meta")


def split_fused_moe_experts(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Find fused MoE experts (with gate_up_proj/down_proj).
    Split them from 3D tensors into individual 2D expert tensors.

    Args:
        tensors: Dictionary of loaded tensors from safetensors file

    Returns:
        split_tensors: New dictionary with split expert weights
    """
    split_tensors = {}

    params_to_split = {
        # If a 3D gate_up_proj layer is found, split it into a
        # 2D gate_proj and up_proj layer for each expert
        "gate_up_proj": ["gate_proj", "up_proj"],
        # If a 3D down_proj layer is found, split it into a
        # 2D down_proj layer for each expert
        "down_proj": ["down_proj"],
    }

    for name, tensor in tensors.items():
        keys_to_split = [key for key in params_to_split if key in name]
        if len(keys_to_split) >= 2:
            raise ValueError(f"Found multiple keys matching {name}: {keys_to_split}")

        elif len(keys_to_split) == 1 and tensor.ndim == 3:
            unsplit_name = keys_to_split[0]
            split_names = params_to_split[unsplit_name]

            # Get number of experts
            num_experts = tensor.shape[0]

            if tensor.shape[1] % len(split_names) != 0:
                raise ValueError(
                    f"{unsplit_name} expects a second dimension divisible by "
                    f"{len(split_names)} but got shape: {tensor.shape}"
                )

            # Split into experts
            intermediate_size = tensor.shape[1] // len(split_names)
            for expert_idx in range(num_experts):
                expert_tensor = tensor[expert_idx]
                # Split into layers
                split_layers = expert_tensor.split(intermediate_size, dim=0)
                for split_name, split_layer in zip(split_names, split_layers):
                    key = name.replace(unsplit_name, f"{expert_idx}.{split_name}")
                    if not key.endswith(".weight"):
                        key = f"{key}.weight"
                    split_tensors[key] = split_layer

            logger.info(f"Split {name} into {num_experts} experts")

        else:
            # Non-MoE or non-3D tensors, keep as is
            split_tensors[name] = tensor

    return split_tensors
