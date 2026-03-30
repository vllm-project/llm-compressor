import re
from collections import defaultdict

from compressed_tensors.quantization import QuantizationScheme, QuantizationStrategy

from llmcompressor.entrypoints.model_free.helpers import (
    MatchedNamesSet,
    match_names_set_eager,
)

__all__ = [
    "build_inverse_weights_map",
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


def build_microscale_inverse_weights_map(
    shard_name: str,
    weight_map: dict[str, str],
    model_files: dict[str, str],
) -> dict[str, list[str]]:
    """
    For a given output shard, precompute exactly which tensors need to be
    loaded from which source files in order to run model_free_ptq with microscale
    scheme — including fused partner tensors that live in other shards.

    This moves fused partner discovery out of the per-process runtime and
    into the job-building phase, avoiding redundant re-discovery and enabling
    cleaner process function signatures.

    For example, given:
        shard0: [q_proj.weight, ...]
        shard1: [k_proj.weight, v_proj.weight, ...]

    The inverse_weights_map for shard0's job would be:
        {
            "/path/to/shard0.safetensors": ["q_proj.weight", ...],
            "/path/to/shard1.safetensors": ["k_proj.weight", "v_proj.weight"],
        }

    :param shard_name: the shard filename this job will process and save
    :param weight_map: mapping of tensor name -> shard filename (from index.json)
    :param model_files: mapping of shard filename -> resolved absolute path
    :return: dict mapping resolved source file path -> list of tensor names to load
    """
    # These are now module-level since function is in microscale.py
    # DEFAULT_FUSED_MAPPINGS and get_fused_names are available at module scope

    # Tensors natively belonging to this shard
    native_tensors = [t for t, s in weight_map.items() if s == shard_name]

    # Check if all fused sets are already complete within this shard
    _, unmatched_sets = get_fused_names(native_tensors)

    # Start with native tensors grouped by their source file
    result: dict[str, list[str]] = defaultdict(list)
    own_resolved = model_files[shard_name]
    result[own_resolved] = list(native_tensors)

    if not unmatched_sets:
        return dict(result)

    # For each unmatched fused set, find partner tensors in other shards
    all_patterns = [p for mapping in DEFAULT_FUSED_MAPPINGS for p in mapping]

    for unmatched in unmatched_sets:
        present_names = {v for v in unmatched.values() if v is not None}
        layer_prefixes = {name.rsplit(".", 2)[0] for name in present_names}

        for tensor_name, tensor_shard in weight_map.items():
            if tensor_shard == shard_name:
                continue  # already in native tensors
            resolved = model_files.get(tensor_shard)
            if resolved is None:
                continue
            candidate_prefix = tensor_name.rsplit(".", 2)[0]
            if candidate_prefix not in layer_prefixes:
                continue
            if any(_match_regex_glob(tensor_name, p) for p in all_patterns):
                if tensor_name not in result[resolved]:
                    result[resolved].append(tensor_name)

    return dict(result)


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


def build_inverse_weights_map(
    shard_name: str,
    weight_map: dict[str, str],
    model_files: dict[str, str],
) -> dict[str, list[str]]:
    """
    For a given output shard, precompute exactly which tensors to load from
    which source files — including fused partner tensors from other shards.

    Uses DEFAULT_FUSED_MAPPINGS with primary->partners structure to ensure
    only the shard owning the primary tensor fetches its partners, preventing
    double reads when fused weights span multiple shards.

    Example — given:
        shard0: [q_proj.weight, ...]   <- primary owner
        shard1: [k_proj.weight, v_proj.weight, ...]   <- partners

    Only shard0's inverse_weights_map will include shard1's tensors.
    Shard1's job loads only its own native tensors.

    :param shard_name: the shard filename this job will process and save
    :param weight_map: tensor name -> shard filename (from safetensors.index.json)
    :param model_files: shard filename -> resolved absolute path
    :return: {resolved_file_path: [tensor_names_to_load]}
    """
    own_resolved = model_files[shard_name]
    native_tensors = [t for t, s in weight_map.items() if s == shard_name]

    inverse_weights_map: dict[str, list[str]] = defaultdict(list)
    inverse_weights_map[own_resolved] = list(native_tensors)

    # For each native tensor that matches a primary pattern, fetch its partners
    for name in native_tensors:
        for primary_pattern, partner_templates in DEFAULT_FUSED_MAPPINGS.items():
            match = re.match(primary_pattern, name)
            if match is None:
                continue

            # Build partner names using named groups from the match
            for partner_template in partner_templates:
                partner_name = partner_template.format(**match.groupdict())

                partner_shard = weight_map.get(partner_name)
                if partner_shard is None or partner_shard == shard_name:
                    continue  # same shard or not found

                partner_resolved = model_files.get(partner_shard)
                if partner_resolved is None:
                    continue

                if partner_name not in inverse_weights_map[partner_resolved]:
                    inverse_weights_map[partner_resolved].append(partner_name)

    return dict(inverse_weights_map)
