"""Save GLM trailing MTP layers alongside a compressed backbone."""

import json
import os
import re
from collections import defaultdict

import torch
from compressed_tensors.compressors import BaseCompressor
from compressed_tensors.compressors.format import infer_module_format
from compressed_tensors.entrypoints.convert import (
    CompressedTensorsDequantizer,
    FP8BlockDequantizer,
)
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationMetadata,
    preset_name_to_scheme,
)
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_weight_map,
    get_weight_mappings,
    update_safetensors_index,
)
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import PreTrainedModel


def save_mtp_tensors(
    model: PreTrainedModel,
    save_directory: str | os.PathLike,
    mtp_quant_scheme: str | None,
    loaded_mtp: torch.nn.Module | None = None,
) -> None:
    """Copy, dequantize, or save calibrated GLM MTP weights without a second pass."""
    text_config = model.config.get_text_config()
    if text_config.num_mtp_layers != 1:
        raise ValueError("This MTP save path expects one GLM trailing layer")
    prefix = f"model.layers.{text_config.num_hidden_layers}"
    destination = os.fspath(save_directory)

    if mtp_quant_scheme == "NVFP4":
        if loaded_mtp is None:
            raise ValueError(
                "NVFP4 MTP saving requires an attached, calibrated MtpModel"
            )
        tensors = {
            f"{prefix}.{name.removeprefix('mtp_block.')}": tensor.detach()
            .cpu()
            .contiguous()
            for name, tensor in loaded_mtp.layers[0].state_dict().items()
        }
        tensors[f"{prefix}.shared_head.norm.weight"] = tensors.pop(
            f"{prefix}.post_norm.weight"
        )
        if not any(name.endswith(".weight_packed") for name in tensors):
            raise ValueError("Attached MTP layer has no packed NVFP4 weights")
        _write_shard(destination, tensors)
        _update_config(destination, prefix, loaded=True)
        return

    if mtp_quant_scheme not in (None, "bf16"):
        raise ValueError(f"Unsupported MTP scheme: {mtp_quant_scheme}")

    source = model.name_or_path
    model_files = get_checkpoint_files(source)
    weight_map = get_weight_map(model_files)
    by_file = defaultdict(list)
    for name, shard in weight_map.items():
        if name.startswith(f"{prefix}."):
            by_file[model_files[shard]].append(name)
    if not by_file:
        raise ValueError(f"No GLM MTP tensors found in {source}")

    tensors = {}
    for path, names in by_file.items():
        with safe_open(path, framework="pt") as shard:
            tensors.update({name: shard.get_tensor(name) for name in names})

    config_path = model_files.get("config.json") or model_files.get("params.json")
    if config_path is None:
        raise ValueError(f"No source config found for {source}")
    with open(config_path, encoding="utf-8") as file:
        source_config = json.load(file)
    if mtp_quant_scheme == "bf16":
        tensors = _dequantize(source, source_config, tensors)
    _write_shard(destination, tensors)
    _update_config(destination, prefix, source_config=source_config, tensors=tensors)


def _dequantize(source: str, config: dict, tensors: dict) -> dict:
    tensors = {
        name: tensor
        for name, tensor in tensors.items()
        if name.rpartition(".")[-1].startswith("weight_")
        or name.rpartition(".")[-1] not in QuantizationMetadata.all_qparam_names()
    }
    names = set(tensors)
    if any(name.endswith(".weight_scale_inv") for name in names):
        targets = [
            name.removesuffix(".weight_scale_inv")
            for name in names
            if name.endswith(".weight_scale_inv")
        ]
        block_size = tuple(
            config.get("quantization_config", {}).get("weight_block_size", (128, 128))
        )
        tensors = FP8BlockDequantizer(
            targets=targets, weight_block_size=block_size
        ).validate(tensors)
    elif _source_scheme(config, names) is not None:
        tensors = CompressedTensorsDequantizer(source).validate(tensors)

    qparams = set(QuantizationMetadata.all_qparam_names()) | {
        "weight_packed",
        "weight_scale_inv",
    }
    residual = [name for name in tensors if name.rpartition(".")[-1] in qparams]
    if residual:
        raise ValueError(f"MTP dequantization left compressed tensors: {residual}")
    return {
        name: tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
        for name, tensor in tensors.items()
    }


def _write_shard(destination: str, tensors: dict[str, torch.Tensor]) -> None:
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


def _source_scheme(config: dict, names: set[str]):
    raw = config.get("quantization_config", {})
    if any(name.endswith(".weight_scale_inv") for name in names):
        scheme = preset_name_to_scheme("FP8_BLOCK", targets=[])
        block_size = raw.get("weight_block_size", (128, 128))
        scheme.weights.block_structure = list(block_size)
        scheme.input_activations.group_size = block_size[1]
        scheme.format = infer_module_format(torch.nn.Linear, scheme).value
        return scheme
    if raw.get("quant_method") != "compressed-tensors":
        return None
    config = QuantizationConfig.model_validate(raw)
    groups = list(config.config_groups.items())
    groups.sort(key=lambda item: item[0] != "mtp_group")
    for _, scheme in groups:
        scheme = scheme.model_copy(deep=True)
        scheme.format = infer_module_format(torch.nn.Linear, scheme).value
        params = BaseCompressor.get_value_from_registry(
            scheme.format
        ).compression_param_names(scheme)
        if any(
            name.endswith(f".{params[0]}")
            and all(f"{name.rpartition('.')[0]}.{param}" in names for param in params)
            for name in names
        ):
            return scheme
    return None


def _update_config(
    destination: str,
    prefix: str,
    source_config: dict | None = None,
    tensors: dict[str, torch.Tensor] | None = None,
    loaded: bool = False,
) -> None:
    path = os.path.join(destination, "config.json")
    with open(path, encoding="utf-8") as file:
        config = json.load(file)
    quant = config.get("quantization_config")
    if quant is None:
        return

    ignores = quant.get("ignore") or []
    if loaded:
        for index, name in enumerate(ignores):
            match = re.fullmatch(r"mtp\.layers\.0\.(?:mtp_block\.)?(.+)", name)
            if match:
                ignores[index] = f"{prefix}.{match[1]}"
    else:
        scheme = _source_scheme(source_config, set(tensors))
        if scheme is None:
            ignores.append(rf"re:^{re.escape(prefix)}\.")
        else:
            scheme.targets = [rf"re:^{re.escape(prefix)}\."]
            groups = quant["config_groups"]
            quant["config_groups"] = {
                "mtp_group": scheme.model_dump(mode="json", exclude_none=True),
                **{
                    name: value for name, value in groups.items() if name != "mtp_group"
                },
            }
            compressed = {
                name.rpartition(".")[0]
                for name in tensors
                if name.rpartition(".")[-1].startswith("weight_")
            }
            ignores.extend(
                name.removesuffix(".weight")
                for name in tensors
                if name.endswith(".weight")
                and name.removesuffix(".weight") not in compressed
            )
            formats = {
                group.get("format") or quant.get("format")
                for group in quant["config_groups"].values()
            }
            quant["format"] = formats.pop() if len(formats) == 1 else "mixed-precision"
    quant["ignore"] = list(dict.fromkeys(ignores))
    with open(path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
