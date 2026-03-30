import json

from compressed_tensors.quantization import (
    QuantizationScheme,
    preset_name_to_scheme,
)
from compressed_tensors.utils import getattr_chain
from loguru import logger

from .helpers import find_safetensors_index_file, invert_mapping
from .microscale import get_fused_names, is_microscale_scheme

__all__ = ["validate_scheme", "validate_safetensors_index"]


def validate_scheme(scheme: QuantizationScheme) -> tuple[str, QuantizationScheme]:
    # treat strings as preset schemes
    if isinstance(scheme, str):
        scheme_name, scheme = scheme, preset_name_to_scheme(scheme, [])
    else:
        scheme_name = "config_group_0"

    # weight quantization must be provided
    if scheme.weights is None:
        raise ValueError(
            "Must provide a weights quantization scheme to perform weights-only PTQ"
        )

    # activation quantization must be dynamic
    input_dynamic = getattr_chain(scheme, "input_activations.dynamic", True)
    output_dynamic = getattr_chain(scheme, "output_activations.dynamic", True)
    if input_dynamic is not True or output_dynamic is not True:
        raise ValueError(
            "Model Free PTQ cannot calibrate activations. Please use `oneshot` instead."
        )

    # override with static observers
    # Remove after https://github.com/vllm-project/compressed-tensors/pull/489
    if scheme.weights.observer in ("minmax", "mse"):
        new_observer = f"static_{scheme.weights.observer}"
        logger.warning(
            f"Scheme uses {scheme.weights.observer} weight observer. "
            f"Using {new_observer} instead"
        )
        scheme.weights.observer = new_observer

    # target all modules; filter by ignore list
    # technically this should be "re:.*", but vllm's
    # ct moe layer has a hard coded check for "Linear"
    if len(scheme.targets) == 0:
        scheme.targets.append("Linear")
    return scheme_name, scheme


def validate_safetensors_index(model_files: dict[str, str], scheme: QuantizationScheme):
    index_file_path = find_safetensors_index_file(model_files)
    if index_file_path is None:
        return

    if is_microscale_scheme(scheme):
        with open(index_file_path, "r") as file:
            weights_map: dict[str, str] = json.load(file)["weight_map"]

        file_map = invert_mapping(weights_map)
        for file in sorted(file_map):
            tensor_names = file_map[file]
            _fused_sets, unmatched_sets = get_fused_names(tensor_names)
            if len(unmatched_sets) > 0:
                # Cross-shard fused weights detected. model_free_ptq handles
                # this automatically via precomputed inverse_weights_map —
                # fused partner tensors are fetched via partial reads and
                # re-saved into the requesting shard's output.
                logger.debug(
                    f"{file} has fused weights split across shards: "
                    f"{json.dumps(unmatched_sets, indent=4)}\n"
                    "These will be resolved via precomputed inverse_weights_map."
                )
