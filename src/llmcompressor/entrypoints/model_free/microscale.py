from compressed_tensors.quantization import QuantizationScheme, QuantizationStrategy
from compressed_tensors.utils.match import match_name

from llmcompressor.entrypoints.model_free.helpers import (
    MatchedNamesSet,
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


# def get_fused_names(
#     tensor_names: set[str] | list[str],
# ) -> tuple[list[MatchedNamesSet], list[MatchedNamesSet]]:
#     matched = []
#     unmatched = []
#     for mapping in DEFAULT_FUSED_MAPPINGS:
#         _matched, _unmatched = match_names_set_eager(tensor_names, mapping)

#         matched.extend(_matched)
#         if _unmatched is not None:
#             unmatched.append(_unmatched)

#     return matched, unmatched


def get_fused_names(
    tensor_names: set[str] | list[str],
) -> tuple[list[MatchedNamesSet], list[MatchedNamesSet]]:
    matched = []
    unmatched = []
    targets = [
        "re:.*mlp.*\.(gate_up|gate|up|down)_proj.weight$",
        "re:.*self_attn.*\.(kv_b|o|q_a|q_b)_proj.weight$",
        "re:.*self_attn.kv_a_proj_with_mqa.weight$",
        "re:.*self_attn.indexer.(wk|wq_b).weight$",
        "re:.*mlp.*\.(gate_up|gate|up|down)_proj.weight_scale_inv$",
        "re:.*self_attn.*\.(kv_b|o|q_a|q_b)_proj.weight_scale_inv$",
        "re:.*self_attn.kv_a_proj_with_mqa.weight_scale_inv$",
        "re:.*self_attn.indexer.(wk|wq_b).weight_scale_inv$",
    ]
    for tensor_name in tensor_names:
        if any([match_name(tensor_name, target) for target in targets]):
            # if "weight" is found, we want "weight_scale_inv"
            # if "weight_scale_inv" is found, we want "weight"
            desired_tensor_name = (
                tensor_name.rstrip("_scale_inv")
                if tensor_name.endswith("_scale_inv")
                else (tensor_name + "_scale_inv")
            )
            if desired_tensor_name not in tensor_names:
                unmatched.append({desired_tensor_name: tensor_name})
            else:
                matched.append({tensor_name: desired_tensor_name})
    return matched, unmatched
