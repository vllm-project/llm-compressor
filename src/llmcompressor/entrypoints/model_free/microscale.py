from compressed_tensors.quantization import QuantizationScheme, QuantizationStrategy

from llmcompressor.entrypoints.model_free.helpers import (
    MatchedNamesSet,
    match_names_set_eager,
)

__all__ = ["is_microscale_scheme", "get_fused_names", "DEFAULT_FUSED_MAPPINGS"]


DEFAULT_FUSED_MAPPINGS = [
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
    for mapping in DEFAULT_FUSED_MAPPINGS:
        _matched, _unmatched = match_names_set_eager(tensor_names, mapping)

        matched.extend(_matched)
        if _unmatched is not None:
            unmatched.append(_unmatched)

    return matched, unmatched
