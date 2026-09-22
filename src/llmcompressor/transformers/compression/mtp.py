import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass

import torch
from compressed_tensors.compressors import BaseCompressor
from compressed_tensors.compressors.format import infer_module_format
from compressed_tensors.entrypoints.convert import (
    CompressedTensorsDequantizer,
    Converter,
    FP8BlockDequantizer,
)
from compressed_tensors.entrypoints.convert.convert_file import (
    convert_file,
    validate_file,
    write_checkpoint_quantization_config,
)
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationMetadata,
    QuantizationScheme,
    QuantizationStatus,
    preset_name_to_scheme,
)
from compressed_tensors.utils.match import match_name
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_weight_map,
    get_weight_mappings,
    update_safetensors_index,
)
from loguru import logger
from safetensors import safe_open
from transformers import PreTrainedModel

from llmcompressor.entrypoints.model_free.converter import ModelFreePtqConverter
from llmcompressor.entrypoints.model_free.validate import validate_config

__all__ = ["MtpConverter", "save_mtp_tensors"]

_APPENDED_DENSE_PATTERNS = (
    r"eh_proj$",
    r".*\.indexer\.(?:weights_proj|wk)$",
)


@dataclass(frozen=True)
class _MtpLayout:
    source_prefixes: tuple[str, ...]
    runtime_prefixes: tuple[str, ...]
    dense_patterns: tuple[str, ...] = ()

    def owns(self, name: str) -> bool:
        return any(name.startswith(f"{prefix}.") for prefix in self.source_prefixes)

    def targets(self, runtime: bool) -> list[str]:
        prefixes = self.runtime_prefixes if runtime else self.source_prefixes
        return [f"re:^{re.escape(prefix)}\\." for prefix in prefixes]

    def runtime_name(self, name: str) -> str:
        for source, runtime in zip(self.source_prefixes, self.runtime_prefixes):
            if name == source or name.startswith(f"{source}."):
                return runtime + name[len(source) :]
        return name

    def full_precision_ignores(self) -> list[str]:
        return [f"re:^{re.escape(prefix)}(?:\\.|$)" for prefix in self.runtime_prefixes]

    def quantization_ignores(self, runtime: bool) -> list[str]:
        prefixes = self.runtime_prefixes if runtime else self.source_prefixes
        root = "(?:" + "|".join(re.escape(prefix) for prefix in prefixes) + ")"
        patterns = (
            r"(?:.*\.)?(?:embed_tokens|lm_head|fc|gate)$",
            *self.dense_patterns,
        )
        return [rf"re:^{root}\.{pattern}" for pattern in patterns]

    def is_managed_ignore(self, target: str) -> bool:
        return (
            target in (self.full_precision_ignores() + self.quantization_ignores(True))
            or (self.runtime_prefixes == ("mtp",) and target == "re:^mtp.*")
            or any(target.startswith(f"{prefix}.") for prefix in self.runtime_prefixes)
        )


class MtpConverter:
    """Apply an existing converter chain to tensors omitted by Transformers."""

    def __init__(
        self,
        layout: _MtpLayout,
        converters: list[Converter],
        output_scheme: QuantizationScheme | None,
        ignores: list[str] | None = None,
    ):
        self.layout = layout
        self.converters = converters
        self.output_scheme = output_scheme
        self.ignores = ignores or []

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for converter in self.converters:
            tensors = converter.process(tensors)
        return tensors

    def validate(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not tensors:
            raise ValueError("No MTP tensors were provided")
        for converter in self.converters:
            tensors = converter.validate(tensors)
        return tensors

    def get_dependencies(self, weight_name: str) -> set[str]:
        return set().union(
            *(converter.get_dependencies(weight_name) for converter in self.converters)
        )

    def update_model_config(self, model_config: dict) -> dict:
        return model_config

    def update_config(
        self, config: QuantizationConfig | None
    ) -> QuantizationConfig | None:
        if self.output_scheme is None:
            return _ignore_mtp(config, self.layout)

        groups = {
            "mtp_group": self.output_scheme,
            **_groups_without_mtp(config),
        }
        ignores = _ignores_without_mtp(config, self.layout)
        ignores.extend(self.ignores)
        return _new_config(config, groups, ignores)


class _Bfloat16Converter:
    """Cast dense tensors after an existing source dequantizer has run."""

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            name: tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
            for name, tensor in tensors.items()
        }

    def validate(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        qparams = set(QuantizationMetadata.all_qparam_names()) | {
            "weight_packed",
            "weight_scale_inv",
        }
        residual = [name for name in tensors if name.rpartition(".")[-1] in qparams]
        if residual:
            raise ValueError(f"MTP dequantization left compression params: {residual}")
        if any(
            tensor.is_floating_point() and tensor.element_size() == 1
            for tensor in tensors.values()
        ):
            raise ValueError("MTP dequantization left an FP8 tensor")
        return self.process(tensors)

    def get_dependencies(self, weight_name: str) -> set[str]:
        return set()

    def update_config(
        self, config: QuantizationConfig | None
    ) -> QuantizationConfig | None:
        return None

    def update_model_config(self, model_config: dict) -> dict:
        return model_config


def save_mtp_tensors(
    model: PreTrainedModel,
    save_directory: str | os.PathLike,
    mtp_quant_scheme: str | QuantizationScheme | None = None,
    shard_name: str = "model_mtp.safetensors",
) -> None:
    """Copy, dequantize, or data-free quantize unloaded MTP tensors."""
    source_model = getattr(model, "name_or_path", None) or getattr(
        model.config, "_name_or_path", None
    )
    if not source_model:
        raise ValueError("Cannot save MTP tensors without a source checkpoint")

    model_files = get_checkpoint_files(source_model)
    weight_map = get_weight_map(model_files)
    config_path = model_files.get("config.json") or model_files.get("params.json")
    if config_path is None:
        raise ValueError(f"Could not find a config for {source_model}")
    with open(config_path, encoding="utf-8") as file:
        source_config = json.load(file)

    layout = _resolve_layout(source_config, set(weight_map))
    inverse_map: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        if layout.owns(name):
            inverse_map[model_files[shard]].append(name)
    if not inverse_map:
        raise ValueError(f"No MTP tensors found in {source_model}")

    names = {name for shard_names in inverse_map.values() for name in shard_names}
    source_scheme, source_group = _source_scheme(source_config, names)
    if mtp_quant_scheme is None:
        source_quantization = _quantization_config(source_config)
        if source_scheme is None:
            ignores = []
        elif source_group == "mtp_group":
            assert source_quantization is not None
            source_ignores = source_quantization.ignore or []
            modules = {layout.runtime_name(name.rsplit(".", 1)[0]) for name in names}
            ignores = [
                target
                for target in source_ignores
                if any(match_name(module, target) for module in modules)
            ]
        else:
            source_scheme = source_scheme.model_copy(
                update={"targets": layout.targets(True)}
            )
            ignores = _dense_source_ignores(names, source_scheme, layout, runtime=True)
        converter = MtpConverter(layout, [], source_scheme, ignores)
    else:
        scheme = _resolve_scheme(mtp_quant_scheme)
        converters = _source_dequantizers(
            source_model, source_config, layout, names, source_scheme
        )
        converters.append(_Bfloat16Converter())
        if scheme is not None:
            vector_ignores = _vector_weight_ignores(inverse_map)
            source_target = scheme.model_copy(update={"targets": layout.targets(False)})
            converters.append(
                ModelFreePtqConverter(
                    QuantizationConfig(
                        config_groups={"mtp_group": source_target},
                        ignore=layout.quantization_ignores(False) + vector_ignores,
                    )
                )
            )
            scheme = scheme.model_copy(
                update={
                    "targets": layout.targets(True),
                    "format": infer_module_format(torch.nn.Linear, scheme).value,
                }
            )
            ignores = layout.quantization_ignores(True) + [
                layout.runtime_name(name) for name in vector_ignores
            ]
        converter = MtpConverter(
            layout,
            converters,
            scheme,
            ignores if scheme is not None else [],
        )

    destination = os.fspath(save_directory)
    validate_file(dict(inverse_map), [converter])
    _, mtp_weight_map = convert_file(
        dict(inverse_map), os.path.join(destination, shard_name), [converter]
    )
    _update_index(destination, mtp_weight_map, layout)
    _write_config(destination, converter)
    logger.info(f"Saved MTP weights from {source_model} to {destination}")


def _resolve_layout(config: dict, names: set[str]) -> _MtpLayout:
    if any(name.startswith("mtp.") for name in names):
        return _MtpLayout(("mtp",), ("mtp",))

    text_config = config.get("text_config", config)
    start = int(text_config.get("num_hidden_layers", 0))
    count = int(text_config.get("num_nextn_predict_layers", 0))
    if count <= 0:
        raise ValueError("Checkpoint config does not describe MTP layers")

    for base, dense_patterns in (
        (
            "model.language_model.layers",
            (*_APPENDED_DENSE_PATTERNS, r"self_attn\..*$"),
        ),
        ("model.layers", _APPENDED_DENSE_PATTERNS),
    ):
        sources = tuple(f"{base}.{index}" for index in range(start, start + count))
        found_all = all(
            any(name.startswith(f"{prefix}.") for name in names) for prefix in sources
        )
        if found_all:
            runtimes = tuple(
                f"model.layers.{index}" for index in range(start, start + count)
            )
            return _MtpLayout(sources, runtimes, dense_patterns)
    raise ValueError("Could not locate the MTP layers described by the config")


def _vector_weight_ignores(shards: dict[str, list[str]]) -> list[str]:
    """Keep vector weights dense without naming architecture-specific norms."""
    ignores = set()
    for path, names in shards.items():
        with safe_open(path, framework="pt") as file:
            for name in names:
                if not name.endswith(".weight"):
                    continue
                if len(file.get_slice(name).get_shape()) < 2:
                    ignores.add(name.removesuffix(".weight"))
    return sorted(ignores)


def _resolve_scheme(
    value: str | QuantizationScheme,
) -> QuantizationScheme | None:
    if isinstance(value, str) and value.strip().lower() in {"bf16", "bfloat16"}:
        return None
    if isinstance(value, QuantizationScheme):
        scheme = value.model_copy(deep=True)
    elif isinstance(value, str):
        try:
            scheme = preset_name_to_scheme(value.strip().upper(), targets=[])
        except KeyError as error:
            raise ValueError(f"Unknown MTP quantization scheme {value!r}") from error
    else:
        raise TypeError(
            "mtp_quant_scheme must be None, 'bf16', a preset name, or a "
            f"QuantizationScheme, got {type(value).__name__}"
        )
    try:
        config = validate_config(config=None, scheme=scheme, ignore=[])
    except ValueError as error:
        raise ValueError(
            "mtp_quant_scheme must be data-free; schemes requiring activation "
            "calibration cannot be applied during save_pretrained"
        ) from error
    return next(iter(config.config_groups.values()))


def _quantization_config_data(config: dict) -> dict | None:
    return config.get("quantization_config") or config.get("text_config", {}).get(
        "quantization_config"
    )


def _quantization_config(config: dict) -> QuantizationConfig | None:
    data = _quantization_config_data(config)
    if not isinstance(data, dict) or data.get("quant_method") != "compressed-tensors":
        return None
    return QuantizationConfig.model_validate(data)


def _source_scheme(
    source_config: dict, names: set[str]
) -> tuple[QuantizationScheme | None, str | None]:
    config = _quantization_config(source_config)
    if config is not None:
        schemes = [
            (name, scheme)
            for name, scheme in config.config_groups.items()
            if name != "mtp_group"
        ]
        if mtp_scheme := config.config_groups.get("mtp_group"):
            schemes.insert(0, ("mtp_group", mtp_scheme))
        for group, candidate in schemes:
            scheme = candidate.model_copy(deep=True)
            scheme.format = infer_module_format(torch.nn.Linear, scheme).value
            compressor = BaseCompressor.get_value_from_registry(scheme.format)
            params = compressor.compression_param_names(scheme)
            for name in names:
                module, _, param = name.rpartition(".")
                if param == params[0] and all(
                    f"{module}.{required}" in names for required in params
                ):
                    return scheme, group

    if any(name.endswith(".weight_scale_inv") for name in names):
        scheme = preset_name_to_scheme("FP8_BLOCK", targets=[])
        block_size = _weight_block_size(source_config)
        scheme.weights.block_structure = list(block_size)
        scheme.input_activations.group_size = block_size[1]
        scheme.format = infer_module_format(torch.nn.Linear, scheme).value
        return scheme, None
    return None, None


def _source_dequantizers(
    source_model: str,
    source_config: dict,
    layout: _MtpLayout,
    names: set[str],
    source_scheme: QuantizationScheme | None,
) -> list[Converter]:
    ignores = (
        _dense_source_ignores(names, source_scheme, layout, runtime=False)
        if source_scheme is not None
        else []
    )
    if any(name.endswith(".weight_scale_inv") for name in names):
        return [
            FP8BlockDequantizer(
                targets=layout.targets(False),
                ignore=ignores,
                weight_block_size=_weight_block_size(source_config),
                dtype=torch.bfloat16,
            )
        ]
    if source_scheme is None:
        return []

    dequantizer = CompressedTensorsDequantizer(source_model, dtype=torch.bfloat16)
    source_scheme = source_scheme.model_copy(update={"targets": layout.targets(False)})
    dequantizer.quant_config = QuantizationConfig(
        config_groups={"mtp_group": source_scheme}, ignore=ignores
    )
    return [dequantizer]


def _dense_source_ignores(
    names: set[str],
    scheme: QuantizationScheme,
    layout: _MtpLayout,
    runtime: bool,
) -> list[str]:
    compressor = BaseCompressor.get_value_from_registry(scheme.format)
    params = compressor.compression_param_names(scheme)
    compressed = {
        module
        for name in names
        for module, _, param in [name.rpartition(".")]
        if param == params[0]
        and all(f"{module}.{required}" in names for required in params)
    }
    if any(name.endswith(".weight_scale_inv") for name in names):
        compressed = {
            name.removesuffix(".weight_scale_inv")
            for name in names
            if name.endswith(".weight_scale_inv")
        }
    dense = {
        name.removesuffix(".weight")
        for name in names
        if name.endswith(".weight") and name.removesuffix(".weight") not in compressed
    }
    return sorted(layout.runtime_name(name) if runtime else name for name in dense)


def _weight_block_size(config: dict) -> tuple[int, int]:
    raw_config = _quantization_config_data(config) or {}
    block_size = raw_config.get("weight_block_size") or (128, 128)
    if len(block_size) != 2 or any(
        not isinstance(size, int) or size <= 0 for size in block_size
    ):
        raise ValueError("FP8 weight_block_size must contain two positive ints")
    return tuple(block_size)


def _groups_without_mtp(
    config: QuantizationConfig | None,
) -> dict[str, QuantizationScheme]:
    if config is None:
        return {}
    return {
        name: scheme
        for name, scheme in config.config_groups.items()
        if name != "mtp_group"
    }


def _ignores_without_mtp(
    config: QuantizationConfig | None, layout: _MtpLayout
) -> list[str]:
    return [
        value
        for value in ((config.ignore or []) if config is not None else [])
        if not layout.is_managed_ignore(value)
    ]


def _new_config(
    config: QuantizationConfig | None,
    groups: dict[str, QuantizationScheme],
    ignores: list[str],
) -> QuantizationConfig:
    formats = {scheme.format for scheme in groups.values()}
    format_ = next(iter(formats)) if len(formats) == 1 else "mixed-precision"
    return QuantizationConfig(
        config_groups=groups,
        kv_cache_scheme=config.kv_cache_scheme if config is not None else None,
        quantization_status=QuantizationStatus.COMPRESSED,
        format=format_,
        ignore=list(dict.fromkeys(ignores)),
    )


def _ignore_mtp(
    config: QuantizationConfig | None, layout: _MtpLayout
) -> QuantizationConfig | None:
    if config is None:
        return None
    ignores = _ignores_without_mtp(config, layout) + layout.full_precision_ignores()
    return _new_config(config, _groups_without_mtp(config), ignores)


def _write_config(destination: str, converter: MtpConverter) -> None:
    write_checkpoint_quantization_config(destination, [converter])
    config_path = os.path.join(destination, "config.json")
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    groups = config.get("quantization_config", {}).get("config_groups", {})
    if "mtp_group" not in groups:
        return
    config["quantization_config"]["config_groups"] = {
        "mtp_group": groups["mtp_group"],
        **{name: scheme for name, scheme in groups.items() if name != "mtp_group"},
    }
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def _update_index(
    destination: str,
    mtp_weight_map: dict[str, str],
    layout: _MtpLayout,
) -> None:
    weight_map = {
        name: os.path.basename(path)
        for name, path in get_weight_mappings(destination).items()
        if not layout.owns(name)
    }
    weight_map.update(mtp_weight_map)
    total_size = sum(
        os.path.getsize(os.path.join(destination, shard))
        for shard in set(weight_map.values())
    )
    update_safetensors_index(destination, total_size, weight_map)
