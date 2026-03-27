from collections import defaultdict
import re
import fnmatch

from compressed_tensors.quantization import QuantizationScheme, QuantizationStrategy

from llmcompressor.entrypoints.model_free.helpers import (
    MatchedNamesSet,
    match_names_set_eager,
)

__all__ = [
    "DEFAULT_FUSED_MAPPINGS",
    "build_microscale_inverse_weights_map",
    "is_microscale_scheme",
    "get_fused_names",
]

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
    for mapping in DEFAULT_FUSED_MAPPINGS:
        _matched, _unmatched = match_names_set_eager(tensor_names, mapping)

        matched.extend(_matched)
        if _unmatched is not None:
            unmatched.append(_unmatched)

    return matched, unmatched


def _match_regex_glob(name: str, pattern: str) -> bool:
    """
    Pattern matching for tensor names.
    Handles either 're:' prefix for regex patterns or glob patterns.

    :param name: string to check
    :param pattern: either a regex (with "re:" prefix) or a glob pattern
    :returns: True if name matches regex or glob pattern, otherwise False
    """

    if pattern.startswith("re:"):
        # Regex pattern - strip 're:' prefix and match
        regex = pattern[3:]
        return re.match(regex, name) is not None
    else:
        # Glob-style pattern

        return fnmatch.fnmatch(name, pattern)
