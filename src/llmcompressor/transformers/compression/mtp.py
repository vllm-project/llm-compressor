import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, replace

import torch
from compressed_tensors import __version__ as ct_version
from compressed_tensors.base import (
    COMPRESSION_VERSION_NAME,
    QUANTIZATION_CONFIG_NAME,
    QUANTIZATION_METHOD_NAME,
    TRANSFORM_CONFIG_NAME,
)
from compressed_tensors.compressors import BaseCompressor
from compressed_tensors.compressors.format import infer_module_format
from compressed_tensors.distributed import is_source_process
from compressed_tensors.entrypoints.convert import FP8BlockDequantizer
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationMetadata,
    QuantizationScheme,
    QuantizationStatus,
    preset_name_to_scheme,
)
from compressed_tensors.utils.match import match_name
from compressed_tensors.utils.safetensors_load import (
    get_weight_mappings,
    update_safetensors_index,
)
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import PretrainedConfig, PreTrainedModel

from llmcompressor.entrypoints.model_free.converter import ModelFreePtqConverter
from llmcompressor.entrypoints.model_free.microscale import DEFAULT_FUSED_MAPPINGS
from llmcompressor.entrypoints.model_free.validate import validate_config
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    suspend_distributed_timeout,
)

__all__ = ["prepare_mtp_save", "save_mtp_tensors"]


@dataclass(frozen=True)
class _MtpLayout:
    """Describe source and runtime tensor names for one MTP architecture."""

    source_prefixes: tuple[str, ...]
    runtime_prefixes: tuple[str, ...]
    quantized_weights: tuple[str, ...]
    dense_weights: tuple[str, ...]
    targets: tuple[str, ...]
    ignores: tuple[str, ...]
    discarded_tensors: tuple[str, ...] = ()

    def owns(self, name: str) -> bool:
        """Return whether a source tensor belongs to this MTP layout."""
        return any(name.startswith(f"{prefix}.") for prefix in self.source_prefixes)

    def quantizes(self, name: str) -> bool:
        """Return whether a source weight is explicitly quantizable."""
        return any(re.fullmatch(pattern, name) for pattern in self.quantized_weights)

    def keeps_dense(self, name: str) -> bool:
        """Return whether a source weight must remain dense."""
        return any(re.fullmatch(pattern, name) for pattern in self.dense_weights)

    def discards(self, name: str) -> bool:
        """Return whether the runtime omits a source tensor."""
        return any(re.fullmatch(pattern, name) for pattern in self.discarded_tensors)

    def full_precision_ignores(self) -> tuple[str, ...]:
        """Return runtime ignore patterns for an unquantized MTP module."""
        return tuple(f"re:^{re.escape(prefix)}\\." for prefix in self.runtime_prefixes)


def _text_config(config: PretrainedConfig) -> PretrainedConfig:
    """Return the decoder text config across Transformers config variants."""
    get_text_config = getattr(config, "get_text_config", None)
    if not callable(get_text_config):
        return config
    try:
        text_config = get_text_config(decoder=True)
    except TypeError:
        text_config = get_text_config()
    return (
        PretrainedConfig.from_dict(text_config)
        if isinstance(text_config, dict)
        else text_config
    )


def _validate_layer_ids(
    names: set[str],
    pattern: str,
    expected: set[int],
    architecture: str,
    minimum: int = 0,
) -> None:
    """Ensure physical MTP layer ids agree with the architecture config."""
    actual = {
        int(match.group(1))
        for name in names
        if (match := re.match(pattern, name)) is not None
        and int(match.group(1)) >= minimum
    }
    if actual != expected:
        raise ValueError(
            f"{architecture} MTP layers do not match its config: "
            f"expected {sorted(expected)}, found {sorted(actual)}"
        )


def _qwen3_5_layout(config: PretrainedConfig, names: set[str]) -> _MtpLayout:
    """Build the Qwen3.5 MTP projection layout."""
    count = int(getattr(config, "mtp_num_hidden_layers", 0))
    expected = set(range(count))
    _validate_layer_ids(names, r"^mtp\.layers\.(\d+)\.", expected, "Qwen3.5")
    layers = "|".join(str(index) for index in sorted(expected))
    layer = rf"mtp\.layers\.(?:{layers})"
    projections = (
        r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
        r"|mlp\.(?:gate_proj|up_proj|down_proj)"
    )
    runtime_projections = (
        r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj|qkv_proj)"
        r"|mlp\.(?:gate_proj|up_proj|down_proj|gate_up_proj)"
    )
    return _MtpLayout(
        source_prefixes=("mtp",),
        runtime_prefixes=("mtp",),
        quantized_weights=(rf"^{layer}\.(?:{projections})\.weight$",),
        dense_weights=(r"^mtp\.(?:fc|embed_tokens|lm_head)\.weight$",),
        targets=(rf"re:^{layer}\.(?:{runtime_projections})$",),
        ignores=(r"re:^mtp\.(?:fc|embed_tokens|lm_head)$",),
    )


def _glm5_next_layout(config: PretrainedConfig, names: set[str]) -> _MtpLayout:
    """Build the GLM-5.3-Flash MTP projection layout."""
    start = int(config.num_hidden_layers)
    count = int(getattr(config, "num_nextn_predict_layers", 0))
    indices = tuple(range(start, start + count))
    expected = set(indices)
    _validate_layer_ids(
        names,
        r"^model\.language_model\.layers\.(\d+)\.",
        expected,
        "GLM-5.3-Flash",
        minimum=start,
    )

    source_prefixes = tuple(f"model.language_model.layers.{index}" for index in indices)
    layers = "|".join(str(index) for index in indices)
    source_layer = rf"model\.language_model\.layers\.(?:{layers})"
    runtime_layer = rf"model\.layers\.(?:{layers})"
    projections = (
        r"mlp\.(?:experts\.\d+|shared_experts)\." r"(?:gate_proj|up_proj|down_proj)"
    )
    dense = (
        r"eh_proj|mlp\.gate|self_attn\."
        r"(?:q_a_proj|kv_a_proj_with_mqa|q_b_proj|kv_b_proj|o_proj|"
        r"indexer\.(?:weights_proj|wk|wq_b))"
    )
    return _MtpLayout(
        source_prefixes=source_prefixes,
        runtime_prefixes=tuple(f"model.layers.{index}" for index in indices),
        quantized_weights=(rf"^{source_layer}\.(?:{projections})\.weight$",),
        dense_weights=(rf"^{source_layer}\.(?:{dense})\.weight$",),
        targets=(
            rf"re:^{runtime_layer}\.mlp\.experts\.\d+\."
            r"(?:gate_proj|up_proj|down_proj)$",
            rf"re:^{runtime_layer}\.mlp\.shared_experts\."
            r"(?:gate_proj|up_proj|down_proj|gate_up_proj)$",
        ),
        ignores=(
            rf"re:^{runtime_layer}\.(?:eh_proj|mlp\.gate)$",
            rf"re:^{runtime_layer}\.self_attn\..*$",
        ),
        discarded_tensors=(rf"^{source_layer}\.hc_(?:attn|ffn)_(?:base|fn|scale)$",),
    )


def _glm_moe_dsa_layout(config: PretrainedConfig, names: set[str]) -> _MtpLayout:
    """Build the GLM-5.3 MTP projection layout."""
    start = int(config.num_hidden_layers)
    count = int(getattr(config, "num_nextn_predict_layers", 0))
    indices = tuple(range(start, start + count))
    expected = set(indices)
    _validate_layer_ids(
        names,
        r"^model\.layers\.(\d+)\.",
        expected,
        "GLM5.3",
        minimum=start,
    )

    prefixes = tuple(f"model.layers.{index}" for index in indices)
    layers = "|".join(str(index) for index in indices)
    layer = rf"model\.layers\.(?:{layers})"
    projections = (
        r"self_attn\.(?:q_a_proj|kv_a_proj_with_mqa|q_b_proj|kv_b_proj|o_proj)"
        r"|self_attn\.indexer\.wq_b"
        r"|mlp\.(?:experts\.\d+|shared_experts)\."
        r"(?:gate_proj|up_proj|down_proj)"
    )
    dense = r"eh_proj|mlp\.gate|self_attn\.indexer\.(?:weights_proj|wk)"
    return _MtpLayout(
        source_prefixes=prefixes,
        runtime_prefixes=tuple(f"model.layers.{index}" for index in indices),
        quantized_weights=(rf"^{layer}\.(?:{projections})\.weight$",),
        dense_weights=(rf"^{layer}\.(?:{dense})\.weight$",),
        targets=(
            rf"re:^{layer}\.self_attn\."
            r"(?:fused_qkv_a_proj|q_b_proj|kv_b_proj|o_proj|indexer\.wq_b)$",
            rf"re:^{layer}\.mlp\.experts\.\d+\." r"(?:gate_proj|up_proj|down_proj)$",
            rf"re:^{layer}\.mlp\.shared_experts\."
            r"(?:gate_proj|up_proj|down_proj|gate_up_proj)$",
        ),
        ignores=(
            rf"re:^{layer}\.(?:eh_proj|mlp\.gate|"
            r"self_attn\.indexer\.wk_weights_proj)$",
        ),
    )


def _nemotron_h_layout(config: PretrainedConfig, names: set[str]) -> _MtpLayout:
    """Build the NemotronH hybrid MTP projection layout."""
    count = int(getattr(config, "num_nextn_predict_layers", 0))
    pattern = getattr(config, "mtp_hybrid_override_pattern", None)
    if pattern is None:
        pattern = getattr(config, "mtp_layers_block_type", ())
    expected = set(range(count * len(pattern)))
    _validate_layer_ids(names, r"^mtp\.layers\.(\d+)\.", expected, "NemotronH")

    layers = "|".join(str(index) for index in sorted(expected))
    layer = rf"mtp\.layers\.(?:{layers})"
    projections = (
        r"eh_proj|mixer\.(?:q_proj|k_proj|v_proj|o_proj)"
        r"|mixer\.(?:experts\.\d+|shared_experts)\.(?:up_proj|down_proj)"
    )
    dense = r"mixer\.gate"
    return _MtpLayout(
        source_prefixes=("mtp",),
        runtime_prefixes=("mtp",),
        quantized_weights=(rf"^{layer}\.(?:{projections})\.weight$",),
        dense_weights=(
            rf"^{layer}\.(?:{dense})\.weight$",
            r"^mtp\.(?:embed_tokens|lm_head)\.weight$",
        ),
        targets=(
            rf"re:^{layer}\.(?:eh_proj|mixer\."
            r"(?:q_proj|k_proj|v_proj|o_proj|qkv_proj))$",
            rf"re:^{layer}\.mixer\.experts\.\d+\." r"(?:gate_proj|up_proj|down_proj)$",
            rf"re:^{layer}\.mixer\.shared_experts\.(?:up_proj|down_proj)$",
        ),
        ignores=(
            rf"re:^{layer}\.(?:{dense})$",
            r"re:^mtp\.(?:embed_tokens|lm_head)$",
        ),
    )


_MTP_LAYOUTS = {
    "Qwen3_5ForConditionalGeneration": _qwen3_5_layout,
    "Qwen3_5ForCausalLM": _qwen3_5_layout,
    "Qwen3_5MoeForConditionalGeneration": _qwen3_5_layout,
    "Qwen3_5MoeForCausalLM": _qwen3_5_layout,
    "Glm5NextForConditionalGeneration": _glm5_next_layout,
    "Glm5NextForCausalLM": _glm5_next_layout,
    "GlmMoeDsaForCausalLM": _glm_moe_dsa_layout,
    "NemotronHForCausalLM": _nemotron_h_layout,
}


def _resolve_mtp_layout(config: PretrainedConfig, names: set[str]) -> _MtpLayout:
    """Resolve an explicitly supported architecture to its MTP layout."""
    architectures = getattr(config, "architectures", None) or []
    architecture = next(
        (name for name in architectures if name in _MTP_LAYOUTS),
        None,
    )
    if architecture is None:
        raise ValueError(
            "MTP checkpoint processing is supported for "
            f"{sorted(_MTP_LAYOUTS)}, found {architectures or ['unknown']}"
        )
    layout = _MTP_LAYOUTS[architecture](_text_config(config), names)
    if architecture.startswith("Qwen3_5Moe"):
        # The existing MoE save workflow preserves the complete MTP prefix.
        # Do not claim support for quantizing its expert projections yet.
        layout = replace(
            layout,
            quantized_weights=(),
            targets=(),
            dense_weights=(r"^mtp\..*\.weight$",),
        )
    return layout


def _resolve_mtp_scheme(
    mtp_scheme: str | QuantizationScheme | None,
) -> QuantizationScheme | None:
    """Resolve an MTP preset to a calibration-free quantization scheme."""
    if mtp_scheme is None:
        return None
    if isinstance(mtp_scheme, str) and mtp_scheme.strip().upper() == "BF16":
        return None
    if isinstance(mtp_scheme, QuantizationScheme):
        scheme = mtp_scheme.model_copy(deep=True)
    elif isinstance(mtp_scheme, str):
        try:
            scheme = preset_name_to_scheme(mtp_scheme.strip().upper(), targets=[])
        except KeyError as error:
            raise ValueError(
                f"Unknown MTP quantization scheme {mtp_scheme!r}; "
                'use mtp_scheme=None to preserve MTP or "BF16" to dequantize it'
            ) from error
        if scheme.weights is None:
            raise ValueError('Use mtp_scheme=None to preserve MTP or "BF16"')
    else:
        raise TypeError(
            "mtp_scheme must be None, a QuantizationScheme, or a preset name "
            f"str, got {type(mtp_scheme).__name__}"
        )

    try:
        validated = validate_config(config=None, scheme=scheme, ignore=[])
    except ValueError as error:
        logger.warning(
            f"The requested MTP scheme is not data-free: {error}. "
            "Preserving source MTP tensors instead; use an explicit data-free "
            "scheme such as FP8_DYNAMIC, FP8_BLOCK, MXFP4, or NVFP4A16."
        )
        return None
    return next(iter(validated.config_groups.values()))


def _source_weight_map(
    source_model: str, revision: str | None
) -> tuple[dict[str, str], str, bool]:
    """Load a local or Hub checkpoint weight map."""
    source_model = os.fspath(source_model)
    local = os.path.isdir(source_model)
    index_name = "model.safetensors.index.json"

    if local:
        index_path = os.path.join(source_model, index_name)
    else:
        try:
            index_path = hf_hub_download(source_model, index_name, revision=revision)
        except EntryNotFoundError:
            index_path = ""

    if index_path and os.path.exists(index_path):
        with open(index_path) as file:
            weight_map = json.load(file)["weight_map"]
        return weight_map, os.path.dirname(index_path), local

    if local:
        single_path = os.path.join(source_model, "model.safetensors")
    else:
        single_path = hf_hub_download(
            source_model, "model.safetensors", revision=revision
        )
    if not os.path.exists(single_path):
        raise FileNotFoundError(f"No safetensors checkpoint found in {source_model}")
    with safe_open(single_path, framework="pt", device="cpu") as file:
        weight_map = {name: os.path.basename(single_path) for name in file.keys()}
    return weight_map, os.path.dirname(single_path), local


@dataclass(frozen=True)
class _MtpSource:
    """Validated source metadata; never retains model-sized tensor storage.

    :param bf16: Explicit BF16 conversion, distinct from default preservation.
    """

    model: str
    revision: str | None
    directory: str
    local: bool
    weight_map: dict[str, str]
    config: PretrainedConfig
    layout: _MtpLayout
    scheme: QuantizationScheme | None
    bf16: bool = False


def _source_mtp_scheme(source: _MtpSource) -> QuantizationScheme | None:
    """Recover the source scheme for preserving or dequantizing compressed MTP."""
    names = {name for name in source.weight_map if source.layout.owns(name)}
    quantized = any(
        name.endswith((".weight_packed", ".weight_scale", ".weight_global_scale"))
        for name in names
    )
    if not quantized:
        return None
    raw_config = getattr(source.config, "quantization_config", None) or {}
    if hasattr(raw_config, "to_dict"):
        raw_config = raw_config.to_dict()
    raw_scheme = raw_config.get("config_groups", {}).get("mtp_group")
    if raw_config.get("quant_method") != "compressed-tensors" or not raw_scheme:
        raise ValueError("Compressed MTP requires source mtp_group metadata")
    scheme = QuantizationScheme.model_validate(raw_scheme)
    if (
        raw_config.get("quantization_status") != QuantizationStatus.COMPRESSED
        or scheme.format != infer_module_format(torch.nn.Linear, scheme).value
    ):
        raise ValueError("Source MTP is not in a supported compressed representation")
    if not source.layout.targets or scheme.targets != list(source.layout.targets):
        raise ValueError(
            "Source MTP quantization targets do not match this runtime layout"
        )
    return scheme


def _native_mtp_scheme(source: _MtpSource) -> QuantizationScheme | None:
    """Describe native block-FP8 without converting its weights or scales."""
    if not any(
        source.layout.owns(name) and name.endswith(".weight_scale_inv")
        for name in source.weight_map
    ):
        return None
    config = getattr(source.config, "quantization_config", None) or {}
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    block = config.get("weight_block_size") or [128, 128]
    if (
        config.get("activation_scheme", "dynamic") != "dynamic"
        or not isinstance(block, (list, tuple))
        or len(block) != 2
        or any(not isinstance(size, int) or size <= 0 for size in block)
    ):
        return None
    scheme = _resolve_mtp_scheme("FP8_BLOCK")
    scheme.weights.block_structure = block
    scheme.input_activations.group_size = block[1]
    return QuantizationScheme.model_validate(scheme.model_dump())


def _same_mtp_scheme(left: QuantizationScheme, right: QuantizationScheme) -> bool:
    """Compare quantization settings and storage, independently of target names."""
    left_format = left.format or infer_module_format(torch.nn.Linear, left).value
    right_format = right.format or infer_module_format(torch.nn.Linear, right).value
    return left_format == right_format and left.model_dump(
        exclude={"targets", "format"}
    ) == right.model_dump(exclude={"targets", "format"})


def _prepare_mtp_source(
    source_model: str,
    config: PretrainedConfig,
    revision: str | None,
    mtp_scheme: str | QuantizationScheme | None,
) -> _MtpSource:
    weight_map, directory, local = _source_weight_map(source_model, revision)
    config_path = os.path.join(directory, "config.json")
    if not local and not os.path.exists(config_path):
        config_path = hf_hub_download(source_model, "config.json", revision=revision)
    # Read the immutable checkpoint metadata, not the model config that modifiers
    # and Transformers save_pretrained can subsequently mutate.
    source_config = config
    if os.path.exists(config_path):
        with open(config_path) as file:
            source_config = PretrainedConfig.from_dict(json.load(file))
    layout_config = source_config if source_config.architectures else config
    layout = _resolve_mtp_layout(layout_config, set(weight_map))
    scheme = _resolve_mtp_scheme(mtp_scheme)
    bf16 = isinstance(mtp_scheme, str) and mtp_scheme.strip().upper() == "BF16"
    if scheme is not None and not layout.targets:
        logger.warning(
            "MTP quantization is not supported for this architecture; "
            "preserving source MTP"
        )
        scheme = None
    source = _MtpSource(
        source_model,
        revision,
        directory,
        local,
        weight_map,
        source_config,
        layout,
        scheme,
        bf16,
    )
    _source_mtp_scheme(source)
    # Header-only preflight: discover missing files and unclassified projections
    # before calibration without holding MTP tensors in memory.
    found = False
    for shard, names in _mtp_shards(source).items():
        with safe_open(shard, framework="pt", device="cpu") as file:
            for name in names:
                found = True
                shape = file.get_slice(name).get_shape()
                if name.endswith(".weight") and len(shape) == 2:
                    if not layout.quantizes(name) and not layout.keeps_dense(name):
                        raise ValueError(
                            f"Unsupported MTP projection: {name}. The checkpoint "
                            "does not match the supported runtime layout; use a "
                            "compatible checkpoint rather than extending the "
                            "quantization targets."
                        )
                    block = scheme.weights.block_structure if scheme else None
                    if (
                        block
                        and layout.quantizes(name)
                        and ".experts." in name
                        and any(size % width for size, width in zip(shape, block))
                    ):
                        raise ValueError(
                            f"MTP weight {name} with shape {shape} is not divisible "
                            f"by quantization block {block}. The runtime requires "
                            "aligned blocks; use FP8_DYNAMIC or another compatible "
                            "mtp_scheme."
                        )
    if not found:
        raise ValueError(f"No MTP tensors found in {source_model}")
    return source


def prepare_mtp_save(
    model: PreTrainedModel,
    mtp_scheme: str | QuantizationScheme | None = None,
    revision: str | None = None,
) -> _MtpSource | None:
    """Preflight unloaded MTP before calibration or a standalone backbone save."""
    config = _text_config(model.config)
    if not any(
        getattr(config, field, 0)
        for field in (
            "mtp_num_hidden_layers",
            "num_mtp_layers",
            "num_nextn_predict_layers",
        )
    ):
        return None
    source = getattr(model, "name_or_path", None) or getattr(
        model.config, "_name_or_path", None
    )
    if not source:
        raise ValueError(
            "Cannot preserve MTP tensors: no source checkpoint path or Hub ID"
        )
    revision = revision or getattr(model.config, "_commit_hash", None)
    return _prepare_mtp_source(source, model.config, revision, mtp_scheme)


def _mtp_shards(source: _MtpSource) -> dict[str, list[str]]:
    """Resolve only shards containing runtime MTP tensors."""
    by_shard = defaultdict(list)
    for name, shard in source.weight_map.items():
        if source.layout.owns(name) and not source.layout.discards(name):
            by_shard[shard].append(name)
    resolved = {}
    for shard, names in by_shard.items():
        path = os.path.join(source.directory, shard)
        if not os.path.exists(path):
            if source.local:
                raise FileNotFoundError(f"MTP shard not found: {path}")
            path = hf_hub_download(source.model, shard, revision=source.revision)
        resolved[path] = names
    return resolved


def _load_mtp_tensors(source: _MtpSource) -> dict[str, torch.Tensor]:
    """Read only tensors owned by the selected MTP layout."""
    tensors = {}
    for shard_path, names in _mtp_shards(source).items():
        with safe_open(shard_path, framework="pt", device="cpu") as file:
            for name in names:
                tensors[name] = file.get_tensor(name)
    return tensors


def _dequantize_fp8_blocks(
    tensors: dict[str, torch.Tensor], config: PretrainedConfig
) -> dict[str, torch.Tensor]:
    """Dequantize native block-FP8 MTP weights to BF16."""
    modules = sorted(
        name.removesuffix(".weight_scale_inv")
        for name in tensors
        if name.endswith(".weight_scale_inv")
    )
    if not modules:
        return tensors

    quantization_config = getattr(config, "quantization_config", None)
    if hasattr(quantization_config, "to_dict"):
        quantization_config = quantization_config.to_dict()
    block_size = (128, 128)
    if isinstance(quantization_config, dict):
        raw_block_size = quantization_config.get("weight_block_size")
        if raw_block_size is not None:
            block_size = tuple(raw_block_size)
    return FP8BlockDequantizer(targets=modules, weight_block_size=block_size).validate(
        tensors
    )


def _compress_mtp_weights(
    tensors: dict[str, torch.Tensor], scheme: QuantizationScheme
) -> dict[str, torch.Tensor]:
    """Quantize dense MTP weights through model-free PTQ."""
    conversion_scheme = scheme.model_copy(update={"targets": ["re:.*"]})
    config = QuantizationConfig(config_groups={"mtp": conversion_scheme})
    return ModelFreePtqConverter(config).process(dict(tensors))


def _dequantize_mtp_tensors(
    tensors: dict[str, torch.Tensor], source: _MtpSource
) -> dict[str, torch.Tensor]:
    """Restore source MTP weights to dense tensors without changing the source."""
    output = _dequantize_fp8_blocks(dict(tensors), source.config)
    scheme = _source_mtp_scheme(source)
    qparams = set(QuantizationMetadata.all_qparam_names())
    if scheme is not None:
        compressor = BaseCompressor.get_value_from_registry(scheme.format)
        param_names = compressor.compression_param_names(scheme)
        # Source names may differ from the fused runtime targets in mtp_group.
        # The architecture policy identifies the actual checkpoint projections.
        for name in list(output):
            module, _, param = name.rpartition(".")
            if param != param_names[0] or not source.layout.quantizes(
                f"{module}.weight"
            ):
                continue
            try:
                state = {param: output[f"{module}.{param}"] for param in param_names}
            except KeyError as error:
                raise ValueError(
                    f"Missing MTP compression parameter: {error}"
                ) from error
            weight = compressor.decompress(state, scheme)["weight"]
            for param in set(param_names) | qparams:
                output.pop(f"{module}.{param}", None)
            output[f"{module}.weight"] = weight.to(torch.bfloat16)
    for name, tensor in output.items():
        if name.rpartition(".")[-1] in qparams | {"weight_packed", "weight_scale_inv"}:
            raise ValueError(f"MTP dequantization left a compression parameter: {name}")
        if tensor.is_floating_point() and tensor.element_size() == 1:
            raise ValueError(f"MTP dequantization left an FP8 tensor: {name}")
    return output


def _preserve_mtp_tensors(
    tensors: dict[str, torch.Tensor], source: _MtpSource
) -> tuple[dict[str, torch.Tensor], QuantizationScheme | None]:
    """Keep source tensor values and express supported quantization in the output."""
    output = dict(tensors)
    scheme = _source_mtp_scheme(source)
    native = _native_mtp_scheme(source)
    if native is None and any(name.endswith(".weight_scale_inv") for name in tensors):
        raise ValueError('Cannot preserve native MTP quantization settings; use "BF16"')
    if scheme is not None and native is not None:
        raise ValueError("MTP mixes native and compressed-tensors quantization")
    scheme = scheme or native
    if scheme is None:
        if any(
            t.is_floating_point() and t.element_size() == 1 for t in output.values()
        ):
            raise ValueError("Cannot preserve an FP8 tensor without MTP scales")
        return output, None

    compressor = BaseCompressor.get_value_from_registry(
        infer_module_format(torch.nn.Linear, scheme).value
    )
    params = compressor.compression_param_names(scheme)
    qparams = set(QuantizationMetadata.all_qparam_names()) | {"weight_scale_inv"}
    for name, tensor in tensors.items():
        module, _, param = name.rpartition(".")
        quantized = source.layout.quantizes(f"{module}.weight")
        if (param in qparams or param == "weight_packed") and not quantized:
            raise ValueError(
                f'Cannot preserve quantized MTP projection {module}; use "BF16"'
            )
        if param in qparams and f"{module}.{params[0]}" not in tensors:
            raise ValueError(f"Orphan MTP compression parameter: {name}")
        if param != params[0]:
            continue
        if not quantized:
            if tensor.is_floating_point() and tensor.element_size() == 1:
                raise ValueError(
                    f'Cannot preserve FP8 MTP projection {module}; use "BF16"'
                )
            continue
        for required in params:
            key = f"{module}.{required}"
            if native is not None and required == "weight_scale":
                key = f"{module}.weight_scale_inv"
            if key not in tensors:
                raise ValueError(f"Missing MTP compression parameter: {key}")
        if native is not None:
            scale = tensors[f"{module}.weight_scale_inv"]
            block = scheme.weights.block_structure
            if (
                tensor.dtype != torch.float8_e4m3fn
                or tensor.ndim != 2
                or any(size % width for size, width in zip(tensor.shape, block))
                or tuple(scale.shape)
                != tuple(size // width for size, width in zip(tensor.shape, block))
                or not torch.isfinite(scale).all()
                or not (scale > 0).all()
            ):
                raise ValueError(
                    f"Cannot preserve native FP8 MTP shape/scales for {module}; "
                    'use "BF16"'
                )
            # Native scale_inv is a multiplier, like compressed-tensors weight_scale.
            output[f"{module}.weight_scale"] = output.pop(f"{module}.weight_scale_inv")
        for pattern, partners in DEFAULT_FUSED_MAPPINGS.items():
            if match := re.fullmatch(pattern, f"{module}.weight"):
                for partner in partners:
                    peer = partner.format(**match.groupdict()).removesuffix(".weight")
                    if f"{peer}.{params[0]}" not in tensors:
                        raise ValueError(f"Incomplete MTP fusion group: {module}")
                    for scale_name in ("weight_global_scale", "input_global_scale"):
                        key, peer_key = f"{module}.{scale_name}", f"{peer}.{scale_name}"
                        if key in tensors and (
                            peer_key not in tensors
                            or not torch.equal(tensors[key], tensors[peer_key])
                        ):
                            raise ValueError(f"Incompatible fused MTP scales: {key}")
    return output, scheme


def _partition_mtp_tensors(
    tensors: dict[str, torch.Tensor], layout: _MtpLayout
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Partition MTP tensors using the architecture's explicit policy."""
    quantized = {}
    dense = {}
    for name, tensor in tensors.items():
        if not name.endswith(".weight") or tensor.ndim != 2:
            dense[name] = tensor
        elif layout.quantizes(name):
            quantized[name] = tensor
        elif layout.keeps_dense(name):
            dense[name] = tensor
        else:
            raise ValueError(
                f"Unsupported MTP projection for this architecture: {name}"
            )
    if not quantized:
        raise ValueError("No supported MTP projections were found")
    return quantized, dense


def _update_quantization_config(
    destination: str,
    layout: _MtpLayout,
    scheme: QuantizationScheme | None,
    tensor_names: set[str],
) -> None:
    """Merge the MTP scheme and dense exclusions into config.json."""
    config_path = os.path.join(destination, "config.json")
    with open(config_path) as file:
        config = json.load(file)

    raw_quantization_config = config.get(QUANTIZATION_CONFIG_NAME)
    quantization_config = (
        QuantizationConfig.model_validate(raw_quantization_config)
        if raw_quantization_config is not None
        else None
    )
    if quantization_config is None and scheme is None:
        return

    groups = (
        {
            name: group
            for name, group in quantization_config.config_groups.items()
            if name != "mtp_group"
        }
        if quantization_config is not None
        else {}
    )
    managed_ignores = set(layout.ignores + layout.full_precision_ignores())
    managed_ignores.update(
        f"re:^{re.escape(prefix)}\\." for prefix in layout.source_prefixes
    )
    if layout.runtime_prefixes == ("mtp",):
        managed_ignores.add("re:^mtp.*")
    ignores = [
        value
        for value in ((quantization_config.ignore or []) if quantization_config else [])
        if value not in managed_ignores
    ]

    if scheme is None:
        ignores.extend(layout.full_precision_ignores())
        updated = quantization_config.model_copy(
            update={"config_groups": groups, "ignore": list(dict.fromkeys(ignores))}
        )
    else:
        # Validate both checkpoint names and fused runtime names: ignores take
        # precedence over schemes in vLLM, even with mtp_group ordered first.
        fused = {
            "q_proj": "qkv_proj",
            "k_proj": "qkv_proj",
            "v_proj": "qkv_proj",
            "gate_proj": "gate_up_proj",
            "up_proj": "gate_up_proj",
            "q_a_proj": "fused_qkv_a_proj",
            "kv_a_proj_with_mqa": "fused_qkv_a_proj",
        }
        for tensor_name in tensor_names:
            module_name = tensor_name.rsplit(".", 1)[0]
            for source_prefix, runtime_prefix in zip(
                layout.source_prefixes, layout.runtime_prefixes
            ):
                if module_name.startswith(f"{source_prefix}."):
                    module_name = runtime_prefix + module_name[len(source_prefix) :]
                    break
            parent, _, projection = module_name.rpartition(".")
            candidates = {module_name, f"{parent}.{fused.get(projection, projection)}"}
            for name in candidates:
                if any(match_name(name, target) for target in layout.targets):
                    if any(
                        value == "Linear" or match_name(name, value)
                        for value in ignores
                    ):
                        raise ValueError(
                            f"Quantized MTP module {name} conflicts with ignore list"
                        )
        format_ = infer_module_format(torch.nn.Linear, scheme).value
        mtp_group = scheme.model_copy(
            update={"targets": list(layout.targets), "format": format_}
        )
        groups = {"mtp_group": mtp_group, **groups}
        ignores.extend(layout.ignores)
        if quantization_config is None:
            updated = QuantizationConfig(
                config_groups=groups,
                format=format_,
                quantization_status=QuantizationStatus.COMPRESSED,
                ignore=list(dict.fromkeys(ignores)),
            )
        else:
            output_format = quantization_config.format
            if output_format not in (format_, "mixed-precision"):
                output_format = "mixed-precision"
            updated = quantization_config.model_copy(
                update={
                    "config_groups": groups,
                    "format": output_format,
                    "quantization_status": QuantizationStatus.COMPRESSED,
                    "ignore": list(dict.fromkeys(ignores)),
                }
            )

    metadata = {
        COMPRESSION_VERSION_NAME: ct_version,
        QUANTIZATION_METHOD_NAME: "compressed-tensors",
        TRANSFORM_CONFIG_NAME: {},
    }
    if isinstance(raw_quantization_config, dict):
        metadata.update(
            {
                key: value
                for key, value in raw_quantization_config.items()
                if key not in QuantizationConfig.model_fields
            }
        )
    config[QUANTIZATION_CONFIG_NAME] = {
        **metadata,
        **updated.model_dump(mode="json", exclude_none=True),
    }
    with open(config_path, "w") as file:
        json.dump(config, file, indent=2)


def _update_index(
    destination: str,
    shard_name: str,
    tensors: dict[str, torch.Tensor],
    layout: _MtpLayout,
) -> None:
    """Replace stale MTP index entries with the generated shard."""
    index_path = os.path.join(destination, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as file:
            weight_map = dict(json.load(file)["weight_map"])
    else:
        weight_map = {
            name: os.path.basename(path)
            for name, path in get_weight_mappings(destination).items()
        }
    weight_map = {
        name: shard for name, shard in weight_map.items() if not layout.owns(name)
    }
    weight_map.update({name: shard_name for name in tensors})
    total_size = sum(
        os.path.getsize(os.path.join(destination, shard))
        for shard in set(weight_map.values())
    )
    update_safetensors_index(destination, total_size, weight_map)


def _quantize_and_save_mtp_tensors(
    source_model: str,
    destination: str,
    config: PretrainedConfig,
    mtp_scheme: str | QuantizationScheme | None = None,
    revision: str | None = None,
    shard_name: str = "model_mtp.safetensors",
    *,
    source: _MtpSource | None = None,
) -> None:
    """Load, optionally quantize, and save one architecture's MTP tensors."""
    source = source or _prepare_mtp_source(source_model, config, revision, mtp_scheme)
    layout = source.layout
    tensors = _load_mtp_tensors(source)
    scheme = source.scheme
    shard_path = os.path.join(destination, shard_name)
    source_scheme = _source_mtp_scheme(source)
    if source.bf16:
        dense = _dequantize_mtp_tensors(tensors, source)
        output = {
            name: tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
            for name, tensor in dense.items()
        }
    elif scheme is None or (
        (existing := source_scheme or _native_mtp_scheme(source)) is not None
        and _same_mtp_scheme(existing, scheme)
    ):
        output, scheme = _preserve_mtp_tensors(tensors, source)
    else:
        # Dequantization failures remain fatal, outside optional quantization.
        dense = _dequantize_mtp_tensors(tensors, source)
        try:
            quantized, output = _partition_mtp_tensors(dense, layout)
            output.update(_compress_mtp_weights(quantized, scheme))
        except Exception as error:
            logger.warning(
                "Could not apply data-free MTP quantization; preserving source "
                f"MTP tensors instead. Reason: {error}"
            )
            output, scheme = _preserve_mtp_tensors(tensors, source)

    # Conversion may fall back, but a checkpoint write failure must remain fatal.
    output = {name: tensor.contiguous() for name, tensor in output.items()}
    save_file(output, shard_path)

    _update_index(destination, shard_name, output, layout)
    _update_quantization_config(destination, layout, scheme, set(output))


def save_mtp_tensors(
    model: PreTrainedModel,
    save_directory: str,
    mtp_scheme: str | QuantizationScheme | None = None,
    revision: str | None = None,
    *,
    source: _MtpSource | None = None,
) -> None:
    """Process unloaded MTP tensors after a compressed backbone save.

    Supported layouts are Qwen3.5, GLM-5.3 Flash and DSA, and NemotronH.
    Transformers omits these tensors from the model object, so they are read
    from the source checkpoint and always written alongside the saved backbone.
    Source precision is preserved by default. An explicit BF16 request dequantizes
    floating-point MTP tensors; a data-free scheme applies optional quantization.

    :param model: Model whose source checkpoint contains MTP tensors.
    :param save_directory: Directory containing the saved backbone checkpoint.
    :param mtp_scheme: Preset name or ``QuantizationScheme`` for MTP weights.
        ``None`` preserves source precision; ``"BF16"`` dequantizes or casts MTP.
        Matching quantization settings reuse compatible source weights and scales.
    :param revision: Optional source checkpoint revision.
    """
    source = source or prepare_mtp_save(model, mtp_scheme, revision)
    if source is None:
        return
    with suspend_distributed_timeout():
        if is_source_process():
            _quantize_and_save_mtp_tensors(
                source.model,
                save_directory,
                model.config,
                mtp_scheme=mtp_scheme,
                revision=revision,
                source=source,
            )
