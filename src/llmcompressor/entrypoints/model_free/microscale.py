from collections import defaultdict
from typing import Iterable, Mapping

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
    "index_fused_weights",
]


def is_microscale_scheme(scheme: QuantizationScheme) -> bool:
    assert scheme.weights is not None
    return scheme.weights.strategy == QuantizationStrategy.TENSOR_GROUP


def has_microscale_scheme(config: QuantizationConfig) -> bool:
    return any(is_microscale_scheme(scheme) for scheme in config.config_groups.values())


def get_fused_names(
    tensor_names: Iterable[str],
    checkpoint_layers: Mapping[str, Mapping[str, str]] | None = None,
    context_names: Iterable[str] = (),
) -> list[dict[str, str]]:
    """
    Find the fused weight groups among checkpoint weight names, as defined by
    FUSED_MODULE_MAPPINGS. Weights are only grouped with siblings under the same
    parent module, so weights of different layers or experts are never fused.

    :param tensor_names: checkpoint tensor names. Only ".weight" tensors are
        considered
    :param checkpoint_layers: the whole checkpoint's fused layers by parent
        module, from `index_fused_weights` (a plain nested dict, kept untyped
        beyond its mapping shape). They only satisfy mappings'
        `context_names`, which can be in another shard or ignored
    :param context_names: other tensor names that only satisfy mappings'
        `context_names`, e.g. the ignored tensors of the same shard
    :return: one dict per fused group, mapping each layer name (e.g. "q_proj")
        to its weight name, made of `tensor_names` alone. The first entry is
        the group's primary weight
    """
    return _get_fused_groups(
        _index_fused_layers(tensor_names),
        checkpoint_layers or {},
        _index_fused_layers(context_names),
    )


def index_fused_weights(
    weight_names: Iterable[str],
) -> tuple[dict[str, dict[str, str]], dict[str, set[str]]]:
    """
    Index a whole checkpoint's fused weights once, for per-shard resolution

    The partners are resolved with every weight as a member, ignored ones
    included, while a job's groups are resolved from the weights it
    quantizes. A job may then load a partner it does not fuse. Reconciling
    the two is left for later: for every mapping and every module layout in
    the vLLM snapshot, with any one or two layers ignored, each group a job
    resolves is part of a whole-checkpoint group with the same primary.

    :param weight_names: names of all tensors in the checkpoint
    :return: the fused layers by parent module, to pass to `get_fused_names`,
        and each fused group's primary weight name mapped to its partners
    """
    checkpoint_layers = _index_fused_layers(weight_names)
    fused_partners = {}
    for fused_group in _get_fused_groups(checkpoint_layers):
        primary, *partners = fused_group.values()
        fused_partners[primary] = set(partners)
    return checkpoint_layers, fused_partners


def _index_fused_layers(tensor_names: Iterable[str]) -> dict[str, dict[str, str]]:
    """Map each parent module to its fused layer names and their weight names"""
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
    return dict(layers_by_parent)


def _get_fused_groups(
    layers_by_parent: Mapping[str, Mapping[str, str]],
    *context_indexes: Mapping[str, Mapping[str, str]],
) -> list[dict[str, str]]:
    fused_groups = []
    grouped_names: set[str] = set()
    for parent in sorted(layers_by_parent):
        layers = layers_by_parent[parent]
        context_layers = {
            name for index in context_indexes for name in index.get(parent, ())
        }
        for group in resolve_fused_names(layers.keys(), context_layers):
            fused_group = {name: layers[name] for name in group}
            if not grouped_names.isdisjoint(fused_group.values()):
                raise ValueError(
                    f"Weights in fused group {list(fused_group.values())} "
                    "also belong to another fused group"
                )
            grouped_names.update(fused_group.values())
            fused_groups.append(fused_group)

    return fused_groups
