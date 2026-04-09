import re

from compressed_tensors.entrypoints.convert import Converter, build_inverse_weight_maps
from compressed_tensors.quantization import QuantizationScheme, QuantizationStrategy
from compressed_tensors.utils.safetensors_load import InverseWeightMap

from llmcompressor.entrypoints.model_free.helpers import (
    MatchedNamesSet,
    match_names_set_eager,
)

__all__ = [
    "build_microscale_inverse_weight_maps",
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


def build_microscale_inverse_weight_maps(
    weight_map: dict[str, str],
    model_files: dict[str, str],
    converters: list[Converter],
) -> dict[str, InverseWeightMap]:
    """
    This function replicates the logic of
    `compressed_tensors.entrypoints.convert.build_inverse_weight_maps` including the
    case of microscale partner shards, as defined in DEFAULT_FUSED_MAPPINGS

    For a given output shard, precompute exactly which tensors to load from
    which source files — including required partner tensors from other shards.

    This is necessary because some converters require that a set of tensors are
    accessible in order for them to be processed correctly.

    :param shard_name: the shard filename this job will process and save
    :param weight_map: tensor name -> shard filename (from safetensors.index.json)
    :param model_files: shard filename -> resolved absolute path
    :return: {resolved_file_path: [tensor_names_to_load]}
    """

    # TODO move to top level, fulfill rest of Protocol contract
    #  - ideally remove the need for separate process_file and process_microscale_file
    #    functions
    #  - remove build_microscale_inverse_weight_maps entirely, replace with
    #    `compressed_tensors.entrypoints.convert.build_inverse_weight_maps`
    class MicroscaleConverter:
        def get_dependencies(self, weight_name: str) -> set[str]:
            deps = set()
            for primary_pattern, partner_templates in DEFAULT_FUSED_MAPPINGS.items():
                match = re.match(primary_pattern, weight_name)
                if match is None:
                    continue

                # Build partner names using named groups from the match
                for partner_template in partner_templates:
                    partner_name = partner_template.format(**match.groupdict())

                    deps.add(partner_name)
            return deps

    converters.append(MicroscaleConverter())
    return build_inverse_weight_maps(weight_map, model_files, converters)
