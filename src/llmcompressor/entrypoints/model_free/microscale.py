from compressed_tensors.quantization import QuantizationScheme, QuantizationStrategy

from llmcompressor.entrypoints.model_free.helpers import (
    MatchedNamesSet,
    match_names_set_eager,
)

__all__ = [
    "is_microscale_scheme",
    "get_fused_names",
    "DEFAULT_FUSED_MAPPINGS",
]

# Mapping of primary weight pattern -> list of partner weight patterns.
# The shard owning the primary tensor is responsible for fetching its partners.
# This prevents double reads: each fused set is fetched exactly once, by the
# shard that owns the primary (e.g. q_proj fetches k_proj + v_proj).
#
# Patterns use a named group (?P<prefix>...) so partner names can be
# constructed by substituting the matched prefix via:
#   partner.format(prefix=match.group("prefix"))
DEFAULT_FUSED_MAPPINGS: dict[str, list[str]] = {
    # Attention q/k/v fusion: q_proj is primary
    r"^(?P<prefix>.+?)\.(?P<attn>attn|attention|self_attn|self_attention)"
    r"\.q_proj\.weight$": [
        r"{prefix}.{attn}.k_proj.weight",
        r"{prefix}.{attn}.v_proj.weight",
    ],
    # MLA attention fusion: wq_a is primary
    r"^(?P<prefix>.+?)\.(?P<attn>attn|attention|self_attn)\.wq_a\.weight$": [
        r"{prefix}.{attn}.wkv_a_with_mqa.weight",
    ],
    # MLP gate/up fusion: gate_proj is primary
    r"^(?P<prefix>.+?)\.(?P<mlp>mlp|feed_forward)\.gate_proj\.weight$": [
        r"{prefix}.{mlp}.up_proj.weight",
    ],
    # MoE w1/w3 fusion: w1 is primary
    r"^(?P<prefix>.+?)\.w1\.weight$": [
        r"{prefix}.w3.weight",
    ],
}

# List-of-lists format used by get_fused_names and validate.py
_DEFAULT_FUSED_MAPPINGS_LIST = [
    [
        r"re:.*(attn|attention)\.q_proj\.weight$",
        r"re:.*(attn|attention)\.k_proj\.weight$",
        r"re:.*(attn|attention)\.v_proj\.weight$",
    ],
    [
        r"re:.*(attn|attention)\.wq_a\.weight$",
        r"re:.*(attn|attention)\.wkv_a_with_mqa\.weight$",
    ],
    [r"re:.*mlp\.gate_proj\.weight$", r"re:.*mlp\.up_proj\.weight$"],
    [r"re:.*w1\.weight$", r"re:.*w3\.weight$"],
]


def is_microscale_scheme(scheme: QuantizationScheme) -> bool:
    assert scheme.weights is not None
    return scheme.weights.strategy == QuantizationStrategy.TENSOR_GROUP


def get_fused_names(
    tensor_names: set[str] | list[str],
) -> tuple[list[MatchedNamesSet], list[MatchedNamesSet]]:
    matched = []
    unmatched = []
    for mapping in _DEFAULT_FUSED_MAPPINGS_LIST:
        _matched, _unmatched = match_names_set_eager(tensor_names, mapping)
        matched.extend(_matched)
        if _unmatched is not None:
            unmatched.append(_unmatched)
    return matched, unmatched
