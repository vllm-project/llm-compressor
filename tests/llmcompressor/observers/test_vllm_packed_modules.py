"""
Check FUSED_MODULE_MAPPINGS against the layer groups that vLLM packs into one
weight. The groups come from vllm_packed_modules.json, which
tools/generate_vllm_packed_modules.py collects from the `packed_modules_mapping`
and stacked params mappings of vLLM's model definitions, with the linear layers
of each vLLM module that packs some of them. To refresh it, run the script on a
vLLM checkout, then triage the groups and modules this test reports.
"""

import json
from pathlib import Path

import pytest

from llmcompressor.observers.fused_mappings import (
    FUSED_MODULE_MAPPINGS,
    resolve_fused_names,
)

SNAPSHOT = json.loads(
    (Path(__file__).parent / "vllm_packed_modules.json").read_text(encoding="utf-8")
)


def _key(packed: str, *names: str) -> tuple[str, frozenset[str]]:
    # the generator dedupes groups this way, so a name-order change keeps entries
    return packed, frozenset(names)


# vLLM groups that a mapping fuses only when context layers are present. The
# test adds these siblings before resolving the group
COVERED_WITH_SIBLINGS = {
    _key("in_proj_qkvg", "q_proj", "k_proj", "v_proj", "g_proj"): (
        "a_proj",
        "b_proj",
    ),
}

# vLLM groups that llm-compressor does not need to fuse, or fuses in a way this
# test cannot check (as part of a larger group)
OUT_OF_SCOPE = {
    _key("conv1d", "q_conv1d", "k_conv1d", "v_conv1d"): (
        "delta attention short convs: nn.Conv1d in transformers, unquantized in vLLM"
    ),
    _key("fc13", "fc1", "fc3"): "AIMv2 and dots.ocr vision encoder MLP",
    _key("gate_up_proj", "fc1", "fc3"): "Voxtral audio encoder MLP",
    _key("gate_up_proj", "gate_proj", "up_proj", "w1", "w3"): (
        "Pixtral vision encoder, HF and Mistral checkpoint names in one list"
    ),
    _key(
        "input_mix_weight_down_block_inject",
        "input_mix_weight_down",
        "block_inject_weight",
    ): "Qwen4Exp hyper-connection mixer, built with quant_config=None",
    _key(
        "input_mix_weight_down_block_inject",
        "input_mix_weight_down",
        "block_inject_weight",
        "_input_mix_padding",
    ): "Qwen4Exp hyper-connection mixer, built with quant_config=None",
    _key("merged_kv", "k_proj", "v_proj"): "Molmo2 image pooling (vision)",
    _key("merged_proj", "gate_proj", "dense_h_to_4h"): "GLM-4V vision projector",
    _key("merged_qkv", "wq", "wk", "wv"): "Molmo2 vision backbone attention",
    _key(
        "qkv",
        "attention.query",
        "attention.key",
        "attention.value",
        "q_proj",
        "k_proj",
        "v_proj",
    ): "RADIO vision encoder",
    _key("qkv", "q", "k", "v"): (
        "vision tower attention (Qwen2.5-VL, Qwen3-VL, GLM-4.1V and others)"
    ),
    _key(
        "qkv_indexer_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "sparse_indexer_q",
        "sparse_indexer_k",
        "sparse_indexer_z",
        "g_proj",
    ): "Step5 DSA layers: vLLM rejects a quantized fused layer",
    _key("qkv_proj", "q_proj", "k_proj", "v_proj", "gate_proj"): (
        "MuseSpark1x Omni: vLLM accepts only ModelOpt NVFP4 checkpoints"
    ),
    _key("qkv_proj", "q_proj", "k_proj", "v_proj", "wq", "wk", "wv"): (
        "Pixtral vision encoder, HF and Mistral checkpoint names in one list"
    ),
    _key("qkv_proj", "query_key_value"): (
        "Param2MoE stores q/k/v pre-fused, vLLM only renames the layer"
    ),
    _key("qkvg_proj", "q_proj", "k_proj", "v_proj", "g_proj"): (
        "Step5 packs these only when q/k/v and g_proj are unquantized"
    ),
    _key("wk_weights_proj", "weights_proj"): (
        "HY-V4 indexer: load_weights renames wk, covered by wk_weights_proj"
    ),
}

# Whisper and Cohere ASR decoders, also used by Canary and Nemotron Parse
CROSS_ATTENTION = (
    "cross-attention: vLLM packs k/v into kv_proj and keeps q_proj separate, "
    "but the attention mapping fuses q/k/v"
)

# vLLM groups that llm-compressor should fuse but does not yet
NOT_YET_COVERED = {
    _key("kv_proj", "k_proj", "v_proj"): CROSS_ATTENTION,
    _key(
        "in_proj_qkvbfg_a",
        "q_proj",
        "k_proj",
        "v_proj",
        "b_proj",
        "f_a_proj",
        "g_a_proj",
    ): "GLM-5-Next delta attention also packs g_a_proj, which Kimi does not",
    _key("kv_proj", "key_net", "value_net"): (
        "Cohere ASR NeMo checkpoint names (model_free_ptq; oneshot is covered)"
    ),
    _key("kv_proj", "key_proj", "value_proj"): (
        "Qwen4Exp per-layer embedding (ple) key and value projections"
    ),
    _key("qkg_proj", "q_proj", "k_proj", "g_proj"): (
        "Ricursive packs q/k with g_proj, and keeps v_proj separate"
    ),
    _key("qkv_gate_proj", "q_proj", "k_proj", "v_proj", "output_gate_proj"): (
        "Beam packs the attention gate (checkpoint self_attn.gate_proj) with q/k/v"
    ),
    _key("qkv_proj", "indexer.index_qk_proj"): (
        "Qwen4Exp QSA packs indexer.index_qk_proj with q/k/v when quantized alike"
    ),
    _key("qkv_proj", "q_proj", "k_proj", "v_proj", "index_q_proj", "index_k_proj"): (
        "MiniMax-M3 sparse attention packs the indexer q/k with q/k/v"
    ),
    _key("qkv_proj", "q_proj", "k_proj", "v_proj", "o_gate_proj"): (
        "Muse Spark DSpark draft packs o_gate_proj with q/k/v"
    ),
    _key("qkv_proj", "query", "key", "value"): (
        "BERT and RoBERTa self-attention, TeleChat2 query with fused key_value"
    ),
    _key("qkv_proj", "query_net", "key_net", "value_net"): (
        "Cohere ASR NeMo checkpoint names (model_free_ptq; oneshot is covered)"
    ),
    _key("qkvr", "wq_du", "wk_dv", "wv_dv", "wr_du"): (
        "Inkling packs q/k/v with r_proj (checkpoint wq_du/wk_dv/wv_dv/wr_du)"
    ),
    _key("qkvb_proj", "q_proj", "k_proj", "v_proj", "b_proj"): (
        "Bailing MoE V3 delta attention packs b_proj with q/k/v"
    ),
}

MODELS = "vllm/model_executor/models"

# vLLM modules (file, class) that llm-compressor does not need to fuse
MODULES_OUT_OF_SCOPE = {
    (f"{MODELS}/molmo2.py", "ImagePoolingAttention"): "Molmo2 image pooling (vision)",
    (f"{MODELS}/radio.py", "RadioAttention"): "RADIO vision encoder",
}

# vLLM modules whose layers llm-compressor resolves to other groups than vLLM
# packs there
MODULES_NOT_YET_COVERED = {
    (f"{MODELS}/bailing_moe_v3.py", "BailingMoeV3MLAAttention"): (
        "Bailing MoE V3 MLA keeps g_proj out of fused_qkv_a_proj, but the MLA "
        "mapping fuses its optional g_proj, which Kimi K3 packs"
    ),
    (f"{MODELS}/cohere_asr.py", "CohereASRCrossAttention"): CROSS_ATTENTION,
    (f"{MODELS}/whisper.py", "WhisperCrossAttention"): CROSS_ATTENTION,
    (f"{MODELS}/ricursive.py", "RicursiveAttention"): (
        "Ricursive packs q/k with g_proj, and keeps v_proj separate"
    ),
    (f"{MODELS}/sarvam.py", "SarvamMLAAttention"): (
        "Sarvam MLA loads q_a_proj and kv_a_proj_with_mqa as separate layers, "
        "but the MLA mapping fuses them"
    ),
}

# FUSED_MODULE_MAPPINGS groups that vLLM declares under another naming
NOT_IN_VLLM = {
    _key(
        "in_proj_qkvgfab",
        "q_proj",
        "k_proj",
        "v_proj",
        "b_proj",
        "forget_gate.f_a_proj",
    ): "transformers names of Kimi delta attention; vLLM uses checkpoint names",
    _key(
        "in_proj_qkvgfab",
        "q_proj",
        "k_proj",
        "v_proj",
        "b_proj",
        "forget_gate.f_a_proj",
        "g_proj",
    ): "transformers names of Kimi delta attention; vLLM uses checkpoint names",
    _key("fused_qkv_a_proj", "wq_a", "wkv_a_with_mqa"): (
        "Mistral checkpoint names, which vLLM renames before packing"
    ),
    _key("fused_wqa_wkv", "q_a_proj", "kv_proj"): (
        "transformers names of DeepSeek V4 attention; vLLM uses wq_a/wkv"
    ),
    _key("fused_wkv_wgate", "kv_proj", "gate_proj"): (
        "transformers names of DeepSeek V4 compressors; vLLM uses wkv/wgate"
    ),
}

# A group whose only layer is the packed one is stored fused in the checkpoint
# (identity packing), so there is nothing to fuse
GROUPS = [
    (group["packed"], tuple(group["names"]))
    for group in SNAPSHOT["groups"]
    if group["names"] != [group["packed"]]
]


def _param(packed: str, names: tuple[str, ...]):
    key = _key(packed, *names)
    marks = ()
    if key in OUT_OF_SCOPE:
        marks = pytest.mark.skip(reason=OUT_OF_SCOPE[key])
    elif key in NOT_YET_COVERED:
        marks = pytest.mark.xfail(reason=NOT_YET_COVERED[key], strict=True)
    return pytest.param(packed, names, marks=marks, id=f"{packed}:{'+'.join(names)}")


def test_tables_match_snapshot():
    tables = (COVERED_WITH_SIBLINGS, OUT_OF_SCOPE, NOT_YET_COVERED)
    keys = [key for table in tables for key in table]
    assert len(keys) == len(set(keys))
    assert set(keys) <= {_key(packed, *names) for packed, names in GROUPS}

    modules = [*MODULES_OUT_OF_SCOPE, *MODULES_NOT_YET_COVERED]
    assert len(modules) == len(set(modules))
    assert set(modules) <= {(m["file"], m["class"]) for m in SNAPSHOT["modules"]}


@pytest.mark.parametrize("packed,names", [_param(*group) for group in GROUPS])
def test_vllm_packed_group_is_fused(packed, names):
    """
    The layers of a vLLM packed group, with no siblings other than those in
    COVERED_WITH_SIBLINGS, resolve to exactly that group: a mapping's required
    names are all in it, its optional names may join, and mappings whose
    context layers are absent do not apply
    """
    siblings = COVERED_WITH_SIBLINGS.get(_key(packed, *names), ())
    groups = [set(group) for group in resolve_fused_names({*names, *siblings})]
    assert set(names) in groups, f"vLLM packs {names} into {packed}"


@pytest.mark.parametrize(
    "mapping", FUSED_MODULE_MAPPINGS, ids=lambda mapping: mapping.fused_name
)
def test_mapping_is_packed_by_vllm(mapping):
    """
    Each mapping, with and without its optional names, is a group that vLLM
    packs, unless NOT_IN_VLLM gives a reason
    """
    vllm_groups = {frozenset(names) for _, names in GROUPS}
    for names in {mapping.names, mapping.names + mapping.optional_names}:
        key = _key(mapping.fused_name, *names)
        assert frozenset(names) in vllm_groups or key in NOT_IN_VLLM, key


def test_not_in_vllm_matches_mappings():
    keys = {
        _key(mapping.fused_name, *names)
        for mapping in FUSED_MODULE_MAPPINGS
        for names in (mapping.names, mapping.names + mapping.optional_names)
    }
    assert set(NOT_IN_VLLM) <= keys


def test_not_in_vllm_is_absent_from_snapshot():
    """An entry is stale once vLLM packs its names, and the reason with it"""
    vllm_groups = {frozenset(names) for _, names in GROUPS}
    assert not [key for key in NOT_IN_VLLM if key[1] in vllm_groups]


def _module_param(module: dict):
    key = module["file"], module["class"]
    marks = ()
    if key in MODULES_OUT_OF_SCOPE:
        marks = pytest.mark.skip(reason=MODULES_OUT_OF_SCOPE[key])
    elif key in MODULES_NOT_YET_COVERED:
        marks = pytest.mark.xfail(reason=MODULES_NOT_YET_COVERED[key], strict=True)
    name = f"{module['file'].rsplit('/', 1)[-1]}:{module['class']}"
    return pytest.param(module, marks=marks, id=name)


@pytest.mark.parametrize("module", [_module_param(m) for m in SNAPSHOT["modules"]])
def test_vllm_module_is_fused_as_packed(module):
    """
    The linear layers of a vLLM module, with its packed layers split into their
    sub layers, resolve only to groups that the module packs. This catches a
    mapping that fuses more layers than vLLM packs on that module, e.g. through
    an optional name that vLLM keeps separate there. Layers that the module
    builds under mutually exclusive conditions are merged, so a branch on its
    own is not checked here, e.g. Bailing MoE V3 delta attention with b_proj
    excluded, which packs qkv_proj instead of qkvb_proj. A packed layer is
    split into the sub layers of every group its file declares under that
    packed name, so two such groups in one file are not told apart
    """
    file_groups = [
        (group["packed"], frozenset(group["names"]))
        for group in SNAPSHOT["groups"]
        if module["file"] in group["files"] and group["names"] != [group["packed"]]
    ]
    packed = {names for name, names in file_groups if name in module["layers"]}
    layers = set(module["layers"])
    for name, names in file_groups:
        if name in layers:
            layers = layers - {name} | names

    groups = {frozenset(group) for group in resolve_fused_names(layers)}
    assert groups <= packed, f"{sorted(map(sorted, groups - packed))} not in {packed}"
