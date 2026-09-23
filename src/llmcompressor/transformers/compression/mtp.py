"""Checkpoint handling for MTP layers supported by Transformers."""

import json
import os
import re
from collections import defaultdict

import torch
from compressed_tensors.offload import get_execution_device, set_onload_device
from compressed_tensors.quantization import QuantizationMetadata
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_weight_map,
    get_weight_mappings,
    update_safetensors_index,
)
from huggingface_hub import HfApi, hf_hub_download
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import PreTrainedModel

from llmcompressor.modeling.moe.conversion_mappings import (
    get_linearize_load_mappings,
    has_linearize_load_mappings,
)
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D

FALLBACK_EXAMPLE = "examples/model_free_ptq/mtp_fp8_fallback.py"


def targets_mtp(targets: set[str]) -> bool:
    return any("mtp" in target.lower() for target in targets)


def _mtp_patterns(model: PreTrainedModel) -> list[str]:
    """Use the same checkpoint patterns and layer-count filter as MtpModel."""
    num_layers = getattr(model.config.get_text_config(), "num_hidden_layers", None)
    patterns = getattr(model, "_keys_to_ignore_on_load_unexpected", None) or []
    return [
        pattern
        for pattern in patterns
        if (match := re.search(r"\.(\d+)", pattern)) is None
        or num_layers is None
        or int(match.group(1)) >= num_layers
    ]


def _checkpoint_weights(source: str) -> dict[str, str]:
    """Map tensor names to shards without downloading every Hub weight shard."""
    if os.path.isdir(source):
        files = get_checkpoint_files(source)
        return {name: files[shard] for name, shard in get_weight_map(files).items()}

    metadata = HfApi().get_safetensors_metadata(source)
    return {
        name: shard
        for shard, info in metadata.files_metadata.items()
        for name in info.tensors
    }


def _mtp_weights(model: PreTrainedModel) -> tuple[dict[str, str], list[str]]:
    source = model.name_or_path
    if not source:
        return {}, []
    weights = _checkpoint_weights(source)
    patterns = _mtp_patterns(model)
    try:
        decoder = model.get_decoder()
    except (AttributeError, NotImplementedError):
        decoder = None
    decoder_prefix = next(
        (name for name, module in model.named_modules() if module is decoder), None
    )
    trailing_prefix = (
        f"{decoder_prefix + '.' if decoder_prefix else ''}layers."
        if decoder_prefix is not None
        else None
    )
    num_layers = getattr(model.config.get_text_config(), "num_hidden_layers", None)

    def is_mtp(name: str) -> bool:
        if any(re.search(pattern, name) for pattern in patterns):
            return True
        if re.search(r"(?:^|\.)mtp(?:\.|_)", name):
            return True
        if trailing_prefix and name.startswith(trailing_prefix):
            index = name[len(trailing_prefix) :].split(".", 1)[0]
            return (
                num_layers is not None and index.isdigit() and int(index) >= num_layers
            )
        return False

    return {name: shard for name, shard in weights.items() if is_mtp(name)}, patterns


def load_mtp_model(model: PreTrainedModel) -> None:
    if hasattr(model, "mtp"):
        return

    try:
        from transformers.modeling_layers import MtpModel
    except ImportError as error:
        message = (
            "This Transformers version has no MtpModel. Upgrade Transformers or "
            f"use the manual FP8 route in {FALLBACK_EXAMPLE}."
        )
        logger.warning(message)
        raise ValueError(message) from error

    if not _mtp_patterns(model):
        message = (
            "Transformers' MtpModel has no registered MTP checkpoint patterns "
            f"for {type(model).__name__}. For unsupported FP8 layouts, see "
            f"{FALLBACK_EXAMPLE}."
        )
        logger.warning(message)
        raise ValueError(message)

    from transformers.monkey_patching import (
        register_patch_mapping,
        unregister_patch_mapping,
    )

    model_type = model.config.model_type
    patch_experts = has_linearize_load_mappings(model_type)
    if patch_experts:
        experts_cls, _, _ = get_linearize_load_mappings(model_type)
        register_patch_mapping(
            {experts_cls.__name__: LinearExperts2D.get_linear_experts_cls(experts_cls)}
        )
    try:
        try:
            model.mtp = MtpModel.from_pretrained(model)
        except RuntimeError as error:
            if "weights are missing" not in str(error):
                raise
            message = (
                "Transformers' MtpModel could not load this checkpoint's MTP "
                "layers. For unsupported FP8 layouts, see "
                f"{FALLBACK_EXAMPLE}."
            )
            logger.warning(message)
            raise ValueError(message) from error
    finally:
        if patch_experts:
            unregister_patch_mapping([experts_cls.__name__])

    device = torch.device(get_execution_device(model.get_input_embeddings()))
    if device.type != "cpu":
        set_onload_device(model.mtp.layers, device)
        if model.mtp.use_shared_post_norm:
            set_onload_device(model.mtp.shared_post_norm, device)


def save_mtp_tensors(
    model: PreTrainedModel,
    destination: str,
    loaded_mtp: torch.nn.Module | None = None,
    save_compressed: bool = True,
) -> None:
    """Save quantized MTP, or copy untouched MTP tensors from the checkpoint."""
    if loaded_mtp is None:
        text_config = model.config.get_text_config()
        if not any(
            getattr(text_config, name, 0)
            for name in (
                "num_mtp_layers",
                "mtp_num_hidden_layers",
                "num_nextn_predict_layers",
            )
        ):
            return
        weights, patterns = _mtp_weights(model)
        if not weights:
            message = (
                "Model config indicates MTP, but no MTP checkpoint weights were found"
            )
            if not patterns:
                message += f". For unsupported FP8 layouts, see {FALLBACK_EXAMPLE}."
            logger.warning(message)
            return
        message = (
            "MTP weights were not targeted for quantization; copying them "
            "unchanged from the source checkpoint"
        )
        if not patterns:
            message += (
                ". Transformers' MtpModel has no registered pattern for them; "
                f"for unsupported FP8 layouts, see {FALLBACK_EXAMPLE}."
            )
        logger.warning(message)
        by_shard = defaultdict(list)
        for name, shard in weights.items():
            by_shard[shard].append(name)
        tensors = {}
        for shard, names in by_shard.items():
            path = (
                shard
                if os.path.isdir(model.name_or_path)
                else hf_hub_download(model.name_or_path, shard)
            )
            with safe_open(path, framework="pt") as handle:
                tensors.update({name: handle.get_tensor(name) for name in names})
        qparams = set(QuantizationMetadata.all_qparam_names()) | {
            "weight_packed",
            "weight_scale_inv",
        }
        source_quantized = any(
            name.rpartition(".")[-1] in qparams
            or (tensor.is_floating_point() and tensor.element_size() == 1)
            for name, tensor in tensors.items()
        )
        if source_quantized:
            raise ValueError(
                "Cannot copy source-quantized MTP weights without their serving "
                f"scheme. Convert them first; see {FALLBACK_EXAMPLE}."
            )
    else:
        from transformers.core_model_loading import revert_weight_conversion

        patterns = _mtp_patterns(model)
        qparams = set(QuantizationMetadata.all_qparam_names()) | {"weight_packed"}
        state = {
            name: (
                tensor.detach().to(model.dtype).cpu().contiguous()
                if not save_compressed and tensor.is_floating_point()
                else tensor.detach().cpu().contiguous()
            )
            for name, tensor in loaded_mtp.state_dict().items()
            if name.startswith(("layers.", "shared_post_norm."))
            and (save_compressed or name.rpartition(".")[-1] not in qparams)
        }
        tensors = {
            name: tensor.contiguous()
            for name, tensor in revert_weight_conversion(loaded_mtp, state).items()
        }
        if not tensors or any(
            not any(re.search(pattern, name) for pattern in patterns)
            for name in tensors
        ):
            raise ValueError("MTP weights did not map back to checkpoint names")

    shard_name = "model_mtp.safetensors"
    save_file(tensors, os.path.join(destination, shard_name))
    weight_map = {
        name: os.path.basename(path)
        for name, path in get_weight_mappings(destination).items()
        if name not in tensors
    }
    backbone = os.path.join(destination, "model.safetensors")
    if os.path.exists(backbone):
        os.replace(backbone, os.path.join(destination, "model_backbone.safetensors"))
        weight_map = {
            name: "model_backbone.safetensors"
            if shard == "model.safetensors"
            else shard
            for name, shard in weight_map.items()
        }
    weight_map.update({name: shard_name for name in tensors})
    total_size = sum(
        os.path.getsize(os.path.join(destination, shard))
        for shard in set(weight_map.values())
    )
    update_safetensors_index(destination, total_size, weight_map)

    config_path = os.path.join(destination, "config.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    quant = config.get("quantization_config")
    if quant is None:
        return
    ignores = quant.get("ignore") or []
    if loaded_mtp is None:
        ignores.extend(f"re:^{pattern}" for pattern in patterns)
        if any(re.search(r"(?:^|\.)mtp(?:\.|_)", name) for name in tensors):
            ignores.append(r"re:.*mtp.*")
        if not patterns:
            trailing = {
                match.group(1)
                for name in tensors
                if (match := re.match(r"^(.*\.layers\.\d+)\.", name))
            }
            ignores.extend(rf"re:^{re.escape(prefix)}\." for prefix in trailing)
    else:
        for group in quant["config_groups"].values():
            if targets_mtp(set(group["targets"])):
                backbone_targets = [
                    target for target in group["targets"] if "mtp" not in target.lower()
                ]
                group["targets"] = backbone_targets + [
                    f"re:^{pattern}" for pattern in patterns
                ]
        ignores = [name for name in ignores if not name.startswith("mtp.")]
        compressed = {
            name.rpartition(".")[0]
            for name in tensors
            if name.endswith(
                (".weight_packed", ".weight_scale", ".weight_global_scale")
            )
        }
        ignores.extend(
            name.removesuffix(".weight")
            for name in tensors
            if name.endswith(".weight")
            and name.removesuffix(".weight") not in compressed
        )
    quant["ignore"] = list(dict.fromkeys(ignores))
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
