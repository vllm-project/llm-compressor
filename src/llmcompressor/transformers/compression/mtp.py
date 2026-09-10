import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass

import torch
from compressed_tensors import __version__ as ct_version
from compressed_tensors.base import (
    COMPRESSION_VERSION_NAME,
    QUANTIZATION_CONFIG_NAME,
    QUANTIZATION_METHOD_NAME,
    TRANSFORM_CONFIG_NAME,
)
from compressed_tensors.compressors.format import infer_module_format
from compressed_tensors.distributed import is_source_process
from compressed_tensors.entrypoints.convert import FP8BlockDequantizer
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    preset_name_to_scheme,
)
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
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    suspend_distributed_timeout,
)

__all__ = ["save_mtp_tensors"]

_MTP_UNQUANTIZED_ALIASES = frozenset(
    {"bf16", "bfloat16", "none", "dense", "unquantized"}
)


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
        return get_text_config(decoder=True)
    except TypeError:
        return get_text_config()


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
    "Glm5NextForConditionalGeneration": _glm5_next_layout,
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
    return _MTP_LAYOUTS[architecture](_text_config(config), names)


def _resolve_mtp_scheme(
    mtp_scheme: str | QuantizationScheme | None,
) -> QuantizationScheme | None:
    """Resolve an MTP preset and remove activation schemes needing calibration."""
    if mtp_scheme is None:
        return None
    if isinstance(mtp_scheme, QuantizationScheme):
        scheme = mtp_scheme
    elif isinstance(mtp_scheme, str):
        normalized = mtp_scheme.strip()
        if normalized.lower() in _MTP_UNQUANTIZED_ALIASES:
            return None
        scheme = preset_name_to_scheme(normalized.upper(), targets=[])
    else:
        raise TypeError(
            "mtp_scheme must be None, a QuantizationScheme, or a preset name "
            f"str, got {type(mtp_scheme).__name__}"
        )

    input_activations = scheme.input_activations
    if input_activations is not None and input_activations.dynamic is not True:
        logger.warning(
            "MTP activations cannot be calibrated because Transformers does not "
            "construct these layers; quantizing their weights only."
        )
        scheme = scheme.model_copy(update={"input_activations": None})
    return scheme


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


def _load_mtp_tensors(
    source_model: str,
    revision: str | None,
    weight_map: dict[str, str],
    source_dir: str,
    local: bool,
    layout: _MtpLayout,
) -> dict[str, torch.Tensor]:
    """Read only tensors owned by the selected MTP layout."""
    names_by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        if layout.owns(name) and not layout.discards(name):
            names_by_shard[shard].append(name)
    if not names_by_shard:
        raise ValueError(f"No MTP tensors found in {source_model}")

    tensors = {}
    for shard, names in names_by_shard.items():
        shard_path = os.path.join(source_dir, shard)
        if not os.path.exists(shard_path):
            if local:
                raise FileNotFoundError(f"MTP shard not found: {shard_path}")
            shard_path = hf_hub_download(source_model, shard, revision=revision)
        with safe_open(shard_path, framework="pt", device="cpu") as file:
            for name in names:
                tensors[name] = file.get_tensor(name)
    return tensors


def _dequantize_fp8_blocks(
    tensors: dict[str, torch.Tensor], config: PretrainedConfig
) -> dict[str, torch.Tensor]:
    """Dequantize native block-FP8 tensors before applying a new scheme."""
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
    block_size = (
        tuple(quantization_config.get("weight_block_size", (128, 128)))
        if isinstance(quantization_config, dict)
        else (128, 128)
    )
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
    ignores = [
        value
        for value in (quantization_config.ignore if quantization_config else [])
        if value not in managed_ignores
    ]

    if scheme is None:
        ignores.extend(layout.full_precision_ignores())
        updated = quantization_config.model_copy(
            update={"config_groups": groups, "ignore": list(dict.fromkeys(ignores))}
        )
    else:
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
) -> None:
    """Load, optionally quantize, and save one architecture's MTP tensors."""
    weight_map, source_dir, local = _source_weight_map(source_model, revision)
    layout = _resolve_mtp_layout(config, set(weight_map))
    tensors = _load_mtp_tensors(
        source_model,
        revision,
        weight_map,
        source_dir,
        local,
        layout,
    )
    tensors = _dequantize_fp8_blocks(tensors, config)
    scheme = _resolve_mtp_scheme(mtp_scheme)

    if scheme is not None:
        quantized, output = _partition_mtp_tensors(tensors, layout)
        output.update(_compress_mtp_weights(quantized, scheme))
    else:
        output = tensors

    save_file(output, os.path.join(destination, shard_name))
    _update_index(destination, shard_name, output, layout)
    _update_quantization_config(destination, layout, scheme)


def save_mtp_tensors(
    model: PreTrainedModel,
    save_directory: str,
    mtp_scheme: str | QuantizationScheme | None = None,
    revision: str | None = None,
) -> None:
    """Process unloaded MTP tensors after a oneshot backbone save.

    Supported layouts are Qwen3.5, GLM-5.3 Flash and DSA, and NemotronH.
    Transformers omits these tensors from the model object, so they are read
    from the source checkpoint and optionally weight-quantized with model-free
    PTQ.

    :param model: Model whose source checkpoint contains MTP tensors.
    :param save_directory: Directory containing the saved backbone checkpoint.
    :param mtp_scheme: Preset name or ``QuantizationScheme`` for MTP weights.
        ``None`` preserves the MTP tensors at full precision.
    :param revision: Optional source checkpoint revision.
    """
    text_config = _text_config(model.config)
    has_mtp = any(
        getattr(text_config, field, 0)
        for field in (
            "mtp_num_hidden_layers",
            "num_mtp_layers",
            "num_nextn_predict_layers",
        )
    )
    if not has_mtp:
        return

    source_model = getattr(model, "name_or_path", None) or model.config._name_or_path
    with suspend_distributed_timeout():
        if is_source_process():
            _quantize_and_save_mtp_tensors(
                source_model,
                save_directory,
                model.config,
                mtp_scheme=mtp_scheme,
                revision=revision,
            )
