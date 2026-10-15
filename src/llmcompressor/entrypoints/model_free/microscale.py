from collections import defaultdict
from typing import Iterable

from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStrategy,
)

from llmcompressor.observers.fused_mappings import (
    FUSED_MODULE_NAMES_BY_LEAF,
    resolve_fused_names,
)

__all__ = [
    "has_microscale_scheme",
    "is_microscale_scheme",
    "get_fused_names",
]


def is_microscale_scheme(scheme: QuantizationScheme) -> bool:
    assert scheme.weights is not None
    return scheme.weights.strategy == QuantizationStrategy.TENSOR_GROUP


def has_microscale_scheme(config: QuantizationConfig) -> bool:
    return any(is_microscale_scheme(scheme) for scheme in config.config_groups.values())


def get_fused_names(tensor_names: Iterable[str]) -> list[dict[str, str]]:
    """
    Find the fused weight groups among checkpoint weight names, as defined by
    FUSED_MODULE_MAPPINGS. Weights are only grouped with siblings under the same
    parent module, so weights of different layers or experts are never fused.

    :param tensor_names: checkpoint tensor names. Only ".weight" tensors are
        considered
    :return: one dict per fused group, mapping each layer name (e.g. "q_proj")
        to its weight name. The first entry is the group's primary weight
    """
    layers_by_parent: dict[str, dict[str, str]] = defaultdict(dict)
    for tensor_name in tensor_names:
        if not tensor_name.endswith(".weight"):
            continue
        module_name = tensor_name.removesuffix(".weight")
        for layer_name in FUSED_MODULE_NAMES_BY_LEAF.get(
            module_name.rsplit(".", 1)[-1], ()
        ):
            if module_name == layer_name:
                parent = ""
            elif module_name.endswith(f".{layer_name}"):
                parent = module_name.removesuffix(f".{layer_name}")
            else:
                continue
            layers_by_parent[parent][layer_name] = tensor_name

    fused_groups = []
    grouped_names: set[str] = set()
    for parent in sorted(layers_by_parent):
        layers = layers_by_parent[parent]
        for group in resolve_fused_names(layers.keys()):
            fused_group = {name: layers[name] for name in group}
            if not grouped_names.isdisjoint(fused_group.values()):
                raise ValueError(
                    f"Weights in fused group {list(fused_group.values())} "
                    "also belong to another fused group"
                )
            grouped_names.update(fused_group.values())
            fused_groups.append(fused_group)

    return fused_groups
