import contextlib
from functools import wraps
from typing import Type

import torch
import tqdm
from compressed_tensors.compressors import decompress_module
from compressed_tensors.quantization.utils import is_module_quantized
from compressed_tensors.utils import patch_attr
from loguru import logger
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedConfig,
    PreTrainedModel,
)
from transformers.conversion_mapping import (
    register_checkpoint_conversion_mapping,
)
from transformers.monkey_patching import clear_patch_mapping, register_patch_mapping

from llmcompressor.modeling.moe.helpers import FusedExpertsProtocol
from llmcompressor.utils.pytorch.meta import copy_module_to_meta

from .conversion_mappings import (
    get_linearize_load_mappings,
    has_linearize_load_mappings,
    set_save_conversion_mapping,
)
from .linear_experts import LinearExperts2D


def _patch_replace_with_fp8_linear_for_moe():
    """Make replace_with_fp8_linear a no-op for MoE models.

    MoE models with pre-quantized experts don't need FP8 linear replacement.
    The quantized weights are already in the checkpoint and our layerwise
    decompression code handles dequantization.
    """
    try:
        import transformers.integrations.finegrained_fp8

        def patched_replace_with_fp8_linear(model, *args, **kwargs):
            # Return model unchanged. Quantized weights are already loaded.
            logger.info("Skipping FP8 linear replacement for MoE model")
            return model

        transformers.integrations.finegrained_fp8.replace_with_fp8_linear = (
            patched_replace_with_fp8_linear
        )
    except Exception as e:
        logger.warning(f"Could not patch replace_with_fp8_linear: {e}")


@contextlib.contextmanager
def load_quantizable_moe(model_cls: Type[PreTrainedModel] = AutoModelForCausalLM):
    """
    Context manager for loading MoE models with linearized experts for
    efficient calibration and quantization.

    This context manager patches the `from_pretrained` method of the given model class
    to automatically linearize MoE (Mixture-of-Experts) layers during model loading.
    Linearization converts 3D expert weight tensors into 2D format, enabling more
    efficient calibration and quantization of individual experts.

    Two loading pathways are supported:
    1. Direct loading: If the model checkpoint contains 2D weights and conversion
        mappings areregistered for the model type, weights are loaded directly in
        linearized format.
    2. Post-load conversion: If no conversion mappings exist, the model is loaded
        normally and then linearized via `linearize_moe`.

    :param model_cls: The model class to patch, defaults to AutoModelForCausalLM
    """
    # Patch replace_with_fp8_linear to skip expert modules
    _patch_replace_with_fp8_linear_for_moe()

    original_from_pretrained = model_cls.from_pretrained
    patched_fn_called = False

    @classmethod
    @wraps(original_from_pretrained)
    def patched(cls, *args, **kwargs):
        nonlocal patched_fn_called
        patched_fn_called = True

        config = AutoConfig.from_pretrained(*args, **kwargs)
        model_type = config.model_type

        # model is 3d (or otherwise doesn't have mappings)
        # fall back to post-load conversion
        if not has_linearize_load_mappings(model_type):
            model = original_from_pretrained(*args, **kwargs)
            linearize_moe(model)
            return model

        # prepare to load linearized weights
        experts_cls, load_map, save_map = get_linearize_load_mappings(model_type)
        linear_experts_2d_cls = LinearExperts2D.get_linear_experts_cls(experts_cls)
        register_patch_mapping({experts_cls.__name__: linear_experts_2d_cls})
        register_checkpoint_conversion_mapping(model_type, load_map, overwrite=True)

        # load model
        model: PreTrainedModel = original_from_pretrained(*args, **kwargs)

        # prepare for saving to be called later
        clear_patch_mapping()
        set_save_conversion_mapping(model, save_map)
        register_checkpoint_conversion_mapping(model_type, save_map, overwrite=True)

        for _, module in get_linearized_moes(model):
            module._source_experts_cls = experts_cls
            module._source_config = _experts_config(module, model)

        return model

    with patch_attr(model_cls, "from_pretrained", patched):
        try:
            yield
        finally:
            if not patched_fn_called:
                logger.warning(
                    f"`{model_cls.__name__}.from_pretrained` was never called. If you "
                    f"are loading with a model class other than {model_cls.__name__}, "
                    "please pass as argument to `load_quantizable_moe`"
                )


def linearize_moe(model: PreTrainedModel):
    """
    Linearize a mixture-of-experts model after it has been loaded. For more
    runtime-efficient loading, please see `load_quantizable_moe`.

    Experts modules will be replaced by either two pathways:
    1. The expert module has a registered replacement. This is required for
    2. The expert module conforms to the standard transformers MoE format
    (as designated by the `use_experts_implementation` decorator)

    :param model: model containing MoE layers to linearize
    """
    non_linearized_moes = get_non_linearized_moes(model)

    if len(non_linearized_moes) <= 0:
        return model

    logger.warning(
        "MoE is being linearized after loading in order to support efficient "
        "calibration of experts. However, this may be inefficient if the model "
        "checkpoint is already linearized (2D -> 3D -> 2D). Consider registering "
        "a load converter for faster load times. See "
        "https://docs.vllm.ai/projects/llm-compressor/en/latest/developer-tutorials/add-moe-support"  # noqa: E501
    )

    for name, module in tqdm.tqdm(non_linearized_moes, desc="Linearizing experts"):
        config = _experts_config(module, model)
        linear_experts_cls = LinearExperts2D.get_linear_experts_cls(module.__class__)
        linear_moe = linear_experts_cls.from_experts_module(module, config)
        model.set_submodule(name, linear_moe)

    return model


class MoELinearizationModuleTransform:
    """Module-scoped MoE transform for sequential meta initialization."""

    def __init__(self, fqn: str, decompress: bool = False):
        self.fqn = fqn
        self.decompress = decompress
        self.source: torch.nn.Module | None = None
        self.target: torch.nn.Module | None = None
        self.offload_kwargs: dict = {}
        self.source_offloaded = False
        self.materialized = False
        self.restored = False
        # Keep LinearExperts2D in the model through Calibration_End so modifiers
        # can clean up their target modules before the final repack.
        self.restore_after_subgraph = False

    def to_meta(
        self, source: torch.nn.Module, model: torch.nn.Module
    ) -> torch.nn.Module:
        if not isinstance(model, PreTrainedModel):
            raise TypeError("MoE linearization requires a PreTrainedModel root")
        config = _experts_config(source, model)
        linear_experts_cls = LinearExperts2D.get_linear_experts_cls(source.__class__)
        meta_source = copy_module_to_meta(source)
        if self.decompress:
            for submodule in list(meta_source.modules()):
                if is_module_quantized(submodule):
                    decompress_module(submodule, leave_decompressed=False)
        return linear_experts_cls.meta_from_experts_module(meta_source, config)

    def materialize(
        self,
        target: torch.nn.Module,
        source: torch.nn.Module,
        device: torch.device,
        offload_kwargs: dict,
    ) -> dict[str, dict]:
        if not isinstance(target, LinearExperts2D):
            raise TypeError("MoE lazy target must be a LinearExperts2D module")
        if self.decompress:
            for submodule in list(source.modules()):
                if is_module_quantized(submodule):
                    decompress_module(submodule, leave_decompressed=False)
        return target.materialize_from_experts_module(
            source,
            device=device,
            offload_kwargs=offload_kwargs,
        )

    def restore(self, model: torch.nn.Module) -> None:
        """Repack calibrated experts after modifier calibration cleanup."""
        source, target = self.source, self.target
        if not self.materialized:
            if source is not None:
                model.set_submodule(self.fqn, source)
            return
        if target is None:
            target = model.get_submodule(self.fqn)
        if not isinstance(target, LinearExperts2D):
            raise TypeError("MoE lazy target must be a LinearExperts2D module")

        model.set_submodule(self.fqn, target.to_experts_module())


def _experts_config(
    module: torch.nn.Module, model: PreTrainedModel
) -> PreTrainedConfig:
    """Prefer a nested text config so fused experts construct with the right
    constructor (e.g. InklingExperts(InklingTextConfig)).
    """
    if (config := getattr(module, "config", None)) is not None:
        return config
    config = model.config
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        return get_text_config()
    return getattr(config, "text_config", config)


def get_linearized_moes(
    model: torch.nn.Module,
) -> list[tuple[str, LinearExperts2D]]:
    """
    Return all :class:`LinearExperts2D` modules in ``model`` (post-linearization).
    """
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LinearExperts2D)
    ]


def repack_moe(model: PreTrainedModel) -> PreTrainedModel:
    """
    Explicitly pack linearized :class:`LinearExperts2D` modules back into native
    fused 3D expert modules.

    Call this after linearization. Unquantized recipes (for example
    ``REAPPruningModifier`` only) and frozen quantized experts restore
    native fused Parameters, fusing extra qparams alongside ``weight``.
    If experts are already compressed (no dense ``weight``), nested packed
    projections are written instead. For compressed 3D checkpoints:

    ```python
    compressor.compress_model(model)
    repack_moe(model)
    model.save_pretrained(output_dir)
    ```

    Dense experts restore native fused Parameters. Compressed experts replace
    ``gate_up_proj`` / ``down_proj`` with serialization-only nested modules
    that own packed tensors and qparams so Transformers conversion mappings
    can emit native 3D keys. Compressed fused modules are not intended for
    fused expert forward.

    :param model: model containing linearized MoE layers to repack
    :return: the same model with fused expert modules restored
    """
    linearized = get_linearized_moes(model)
    if len(linearized) <= 0:
        return model

    for name, module in tqdm.tqdm(linearized, desc="Repacking experts"):
        fused = module.to_experts_module()
        model.set_submodule(name, fused)

    return model


def get_non_linearized_moes(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Module]]:
    """
    Return all modules which are recognized to be experts layers. A module is recognized
    as an experts layer if it conforms to the `FusedExpertsProtocol` or is registered by
    `LinearExperts2D`.

    :param model: model with modules to check for experts
    :return: list of named modules which are recognized as experts layers
    """
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, FusedExpertsProtocol)
        or LinearExperts2D.get_registration(module.__class__) is not None
    ]
