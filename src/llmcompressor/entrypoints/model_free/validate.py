import json
from typing import Iterable

from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    preset_name_to_scheme,
)
from compressed_tensors.utils import getattr_chain
from compressed_tensors.utils.safetensors_load import (
    find_safetensors_index_file,
)
from loguru import logger

from .helpers import invert_mapping
from .microscale import get_fused_names, has_microscale_scheme

__all__ = ["validate_config", "validate_safetensors_index"]


def validate_config(
    config: QuantizationConfig | None,
    scheme: QuantizationScheme | str | None,
    ignore: Iterable[str],
) -> QuantizationConfig:
    if config is not None and scheme is not None:
        raise ValueError("Please specify either `config` or `scheme`, not both")

    if config is None and scheme is None:
        raise ValueError("Please specify either `config` or `scheme`")

    if scheme is not None:
        if isinstance(scheme, str):
            scheme_name, scheme = scheme, preset_name_to_scheme(scheme, [])
        else:
            scheme_name = "config_group_0"

        config = QuantizationConfig(
            config_groups={scheme_name: scheme},
            quantization_status=QuantizationStatus.INITIALIZED,
            ignore=list(ignore),
        )
    else:
        if ignore:
            existing = list(config.ignore) if config.ignore else []
            config = config.model_copy(update={"ignore": existing + list(ignore)})

    for scheme in config.config_groups.values():
        _validate_scheme(scheme)

    return config


def _validate_scheme(scheme: QuantizationScheme):
    if scheme.weights is None:
        raise ValueError(
            "Must provide a weights quantization scheme to perform weights-only PTQ"
        )

    input_dynamic = getattr_chain(scheme, "input_activations.dynamic", True)
    output_dynamic = getattr_chain(scheme, "output_activations.dynamic", True)
    if input_dynamic is not True or output_dynamic is not True:
        raise ValueError(
            "Model Free PTQ cannot calibrate activations. Please use `oneshot` instead."
        )

    # Remove after https://github.com/vllm-project/compressed-tensors/pull/489
    if scheme.weights.observer in ("minmax", "mse"):
        new_observer = f"static_{scheme.weights.observer}"
        logger.warning(
            f"Scheme uses {scheme.weights.observer} weight observer. "
            f"Using {new_observer} instead"
        )
        scheme.weights.observer = new_observer

    if len(scheme.targets) == 0:
        scheme.targets.append("Linear")


def validate_safetensors_index(model_files: dict[str, str], config: QuantizationConfig):
    index_file_path = find_safetensors_index_file(model_files)
    if index_file_path is None:
        return

    if has_microscale_scheme(config):
        with open(index_file_path, "r") as file:
            weights_map: dict[str, str] = json.load(file)["weight_map"]

        file_map = invert_mapping(weights_map)
        for file in sorted(file_map):
            tensor_names = file_map[file]
            _fused_sets, unmatched_sets = get_fused_names(tensor_names)
            if len(unmatched_sets) > 0:
                logger.debug(
                    f"{file} has fused weights split across shards: "
                    f"{json.dumps(unmatched_sets, indent=4)}\n"
                    "These will be resolved via precomputed inverse_weight_map."
                )
