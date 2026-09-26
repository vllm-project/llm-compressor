from collections import defaultdict
from dataclasses import dataclass
from typing import Collection

from torch.nn import Module

__all__ = [
    "FusedMapping",
    "FUSED_MODULE_MAPPINGS",
    "resolve_fused_names",
    "get_fused_layers",
]


@dataclass(frozen=True)
class FusedMapping:
    """
    Layers that vLLM loads into a single packed weight. Packed layers must be
    quantized the same way and, for TENSOR_GROUP schemes (e.g. NVFP4), must
    share one global_scale.

    Names are paths relative to the layers' common parent module. They are
    matched against transformers module attributes (oneshot) and against
    checkpoint weight names (model_free_ptq). Where the two namings differ, the
    fusion is listed once per naming.

    :param fused_name: name of the packed module in vLLM, for reference
    :param names: layers that are always packed. The mapping applies to a
        parent only when all of these are present
    :param optional_names: layers that are packed only in some model
        configurations. When the mapping applies, those present join the group
    """

    fused_name: str
    names: tuple[str, ...]
    optional_names: tuple[str, ...] = ()


# Single source of truth for linear-layer groups that vLLM packs together
# and that are relevant to llm-compressor quantization, derived from model
# `packed_modules_mapping` and `stacked_params_mapping`.
FUSED_MODULE_MAPPINGS: tuple[FusedMapping, ...] = (
    # MLP / expert gate and up projections
    FusedMapping("gate_up_proj", ("gate_proj", "up_proj")),
    FusedMapping("gate_up_proj", ("w1", "w3")),
    # Attention q/k/v. Gemma 4 `attention_k_eq_v` layers have no v_proj, and
    # vLLM loads k_proj into both the k and v shards of qkv_proj
    FusedMapping("qkv_proj", ("q_proj", "k_proj"), optional_names=("v_proj",)),
    # Kimi delta attention. g_proj is packed only with `use_full_rank_gate`
    FusedMapping(
        "in_proj_qkvgfab",
        ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj"),
        optional_names=("g_proj",),
    ),
    # Kimi delta attention as defined in transformers, which nests f_a_proj
    FusedMapping(
        "in_proj_qkvgfab",
        ("q_proj", "k_proj", "v_proj", "b_proj", "forget_gate.f_a_proj"),
        optional_names=("g_proj",),
    ),
    # Multi-latent attention (DeepSeek V2/V3/V3.2, Kimi). Kimi MLA layers with
    # `mla_use_output_gate` also pack g_proj, into fused_qkv_a_g_proj
    FusedMapping(
        "fused_qkv_a_proj",
        ("q_a_proj", "kv_a_proj_with_mqa"),
        optional_names=("g_proj",),
    ),
    # Multi-latent attention, Mistral checkpoint names
    FusedMapping("fused_qkv_a_proj", ("wq_a", "wkv_a_with_mqa")),
    # DeepSeek V4 attention, transformers names then checkpoint names
    FusedMapping("fused_wqa_wkv", ("q_a_proj", "kv_proj")),
    FusedMapping("fused_wqa_wkv", ("wq_a", "wkv")),
    # DeepSeek V4 compressors (HCA/CSA and indexer), transformers names then
    # checkpoint names. vLLM loads these unquantized
    FusedMapping("fused_wkv_wgate", ("kv_proj", "gate_proj")),
    FusedMapping("fused_wkv_wgate", ("wkv", "wgate")),
    # DeepSeek sparse attention (DSA) indexer, e.g. DeepSeek V3.2 and GLM-5.
    # vLLM loads this unquantized
    FusedMapping("wk_weights_proj", ("wk", "weights_proj")),
)

FUSED_MODULE_NAMES: frozenset[str] = frozenset(
    name
    for mapping in FUSED_MODULE_MAPPINGS
    for name in mapping.names + mapping.optional_names
)


def resolve_fused_names(present_names: Collection[str]) -> list[tuple[str, ...]]:
    """
    Resolve the fused groups among the layers present under one parent module.

    A mapping applies when all of its `names` are present, and those of its
    `optional_names` that are present join the group. This follows vLLM, which
    packs a layer into whichever fused module exists on that parent. For
    example, Kimi `g_proj` is packed into `in_proj_qkvgfab` on delta attention
    layers and into `fused_qkv_a_g_proj` on MLA layers.

    Larger groups take precedence, and a group that shares a layer with an
    already chosen group is dropped. For example, a Kimi delta attention layer
    resolves to one q/k/v/b/f_a group rather than a q/k/v group.

    :param present_names: layer names present under one parent, relative to it
    :return: disjoint groups of layer names. The first name of each group is
        always one of its mapping's required `names`
    """
    candidates: list[tuple[str, ...]] = []
    for mapping in FUSED_MODULE_MAPPINGS:
        if all(name in present_names for name in mapping.names):
            optional = tuple(n for n in mapping.optional_names if n in present_names)
            candidates.append(mapping.names + optional)

    # stable sort: larger groups first, ties keep FUSED_MODULE_MAPPINGS order
    candidates.sort(key=len, reverse=True)

    groups: list[tuple[str, ...]] = []
    claimed: set[str] = set()
    for group in candidates:
        if claimed.isdisjoint(group):
            claimed.update(group)
            groups.append(group)
    return groups


def get_fused_layers(module: Module) -> list[dict[str, Module]]:
    """
    Find the fused groups among the layers of one parent module, as defined by
    FUSED_MODULE_MAPPINGS.

    :param module: parent module of the layers, e.g. an attention module
    :return: one dict per fused group, mapping each layer name to its layer
    """
    present_layers = {}
    for name in FUSED_MODULE_NAMES:
        layer = module
        for part in name.split("."):
            layer = getattr(layer, part, None)
        if isinstance(layer, Module):
            present_layers[name] = layer

    return [
        {name: present_layers[name] for name in group}
        for group in resolve_fused_names(present_layers.keys())
    ]


# Fused layer names indexed by their final path component. This lets checkpoint
# matching consider only mapping names that could match the current tensor.
_names_by_leaf: dict[str, list[str]] = defaultdict(list)
for _name in sorted(FUSED_MODULE_NAMES):
    _names_by_leaf[_name.rsplit(".", 1)[-1]].append(_name)

FUSED_MODULE_NAMES_BY_LEAF: dict[str, tuple[str, ...]] = {
    leaf: tuple(names) for leaf, names in _names_by_leaf.items()
}
