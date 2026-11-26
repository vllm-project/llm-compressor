import json
from collections import Counter

import pytest
import torch
from compressed_tensors.entrypoints.convert import build_inverse_weight_maps
from compressed_tensors.quantization import preset_name_to_scheme
from compressed_tensors.quantization.quant_config import QuantizationConfig
from safetensors.torch import load_file, save_file

from llmcompressor import model_free_ptq
from llmcompressor.entrypoints.model_free.converter import ModelFreePtqConverter
from llmcompressor.entrypoints.model_free.microscale import (
    get_fused_names,
    index_fused_weights,
)

KDA_ATTN = "model.layers.0.self_attn"
KDA_TRANSFORMERS_ATTN = "model.layers.4.self_attn"
MLA_ATTN = "model.layers.1.self_attn"
K_EQ_V_ATTN = "model.layers.2.self_attn"
EXPERT = "model.layers.1.mlp.experts"
V4_ATTN = "layers.3.attn"
GDN = "model.language_model.layers.5.linear_attn"
EXAONE_MLP = "transformer.h.6.mlp"
BAILING_ATTN = "model.layers.7.attention"

# fused groups of a checkpoint mixing the layer types that vLLM packs differently
FUSED_GROUPS = [
    # kimi delta attention with use_full_rank_gate
    [f"{KDA_ATTN}.{n}" for n in ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj")]
    + [f"{KDA_ATTN}.g_proj"],
    # kimi delta attention with transformers checkpoint naming
    [f"{KDA_TRANSFORMERS_ATTN}.{n}" for n in ("q_proj", "k_proj", "v_proj", "b_proj")]
    + [f"{KDA_TRANSFORMERS_ATTN}.forget_gate.f_a_proj"],
    # kimi mla with output gate
    [f"{MLA_ATTN}.{n}" for n in ("q_a_proj", "kv_a_proj_with_mqa", "g_proj")],
    # gemma 4 attention_k_eq_v layer
    [f"{K_EQ_V_ATTN}.q_proj", f"{K_EQ_V_ATTN}.k_proj"],
    # qwen3.5 gated delta net
    [f"{GDN}.in_proj_qkv", f"{GDN}.in_proj_z"],
    [f"{GDN}.in_proj_b", f"{GDN}.in_proj_a"],
    # moe experts
    [f"{EXPERT}.0.gate_proj", f"{EXPERT}.0.up_proj"],
    [f"{EXPERT}.1.gate_proj", f"{EXPERT}.1.up_proj"],
    # deepseek v4 attention and compressor, checkpoint names
    [f"{V4_ATTN}.wq_a", f"{V4_ATTN}.wkv"],
    [f"{V4_ATTN}.compressor.wkv", f"{V4_ATTN}.compressor.wgate"],
    # exaone 3 and 3.5 mlp
    [f"{EXAONE_MLP}.c_fc_0", f"{EXAONE_MLP}.c_fc_1"],
    # bailing moe v3 delta attention
    [f"{BAILING_ATTN}.{n}" for n in ("q_proj", "k_proj", "v_proj", "b_proj")],
]
UNFUSED = [
    f"{KDA_ATTN}.f_b_proj",
    f"{KDA_ATTN}.o_proj",
    f"{MLA_ATTN}.q_b_proj",
    f"{MLA_ATTN}.o_proj",
    f"{EXPERT}.0.down_proj",
    f"{V4_ATTN}.wo_b",
    f"{GDN}.out_proj",
    f"{EXAONE_MLP}.c_proj",
    f"{BAILING_ATTN}.g_proj",
    f"{BAILING_ATTN}.f_proj",
    f"{BAILING_ATTN}.dense",
]
MODULE_NAMES = [name for group in FUSED_GROUPS for name in group] + UNFUSED


def _weight_names():
    return [f"{name}.weight" for name in MODULE_NAMES]


def test_get_fused_names():
    tensor_names = _weight_names() + [
        f"{KDA_ATTN}.q_proj.weight_scale_inv",
        f"{KDA_ATTN}.A_log",
        f"{EXPERT}.0.gate_proj.bias",
    ]
    fused_groups = get_fused_names(tensor_names)
    assert sorted(sorted(group.values()) for group in fused_groups) == sorted(
        sorted(f"{name}.weight" for name in group) for group in FUSED_GROUPS
    )
    # the primary of each group is always a required layer
    primaries = {next(iter(group.values())) for group in fused_groups}
    assert f"{KDA_ATTN}.q_proj.weight" in primaries
    assert f"{MLA_ATTN}.q_a_proj.weight" in primaries


def test_inverse_weight_maps_colocate_fused_groups(tmp_path):
    """
    Each fused group is loaded by exactly one job, even when split across
    shards, and every tensor is loaded exactly once
    """
    weight_map = {
        name: f"shard-{index % 3}.safetensors"
        for index, name in enumerate(_weight_names())
    }
    model_files = {shard: str(tmp_path / shard) for shard in set(weight_map.values())}
    converter = ModelFreePtqConverter(
        QuantizationConfig(config_groups={"g": preset_name_to_scheme("NVFP4", [])}),
        weight_names=weight_map.keys(),
    )

    inverse_weight_maps = build_inverse_weight_maps(
        weight_map, model_files, [converter]
    )

    jobs = [
        {name for names in iwm.values() for name in names}
        for iwm in inverse_weight_maps.values()
    ]
    counts = Counter(name for job in jobs for name in job)
    assert set(counts) == set(weight_map) and set(counts.values()) == {1}
    for group in FUSED_GROUPS:
        weights = {f"{name}.weight" for name in group}
        assert any(weights <= job for job in jobs), group


def test_model_free_ptq_shares_global_scale(tmp_path):
    """Fused groups split across shards are quantized with one global scale"""
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()

    generator = torch.Generator().manual_seed(0)
    shards = {"model-00001.safetensors": {}, "model-00002.safetensors": {}}
    for index, name in enumerate(MODULE_NAMES):
        # distinct absmax per layer, so that unfused layers get distinct global scales
        weight = torch.randn(32, 32, generator=generator)
        weight = weight / weight.abs().max() * (index + 1)
        shard = list(shards)[index % 2]
        shards[shard][f"{name}.weight"] = weight.to(torch.bfloat16)
    for shard, tensors in shards.items():
        save_file(tensors, src / shard)
    weight_map = {name: shard for shard, ts in shards.items() for name in ts}
    with open(src / "model.safetensors.index.json", "w") as file:
        json.dump({"metadata": {}, "weight_map": weight_map}, file)
    with open(src / "config.json", "w") as file:
        json.dump({"architectures": ["TestModel"]}, file)

    model_free_ptq(src, out, scheme="NVFP4A16", device="cpu")

    with open(out / "model.safetensors.index.json") as file:
        output_map = json.load(file)["weight_map"]
    global_scales = {}
    for shard in set(output_map.values()):
        for key, value in load_file(out / shard).items():
            if key.endswith(".weight_global_scale"):
                global_scales[key.removesuffix(".weight_global_scale")] = value

    assert set(global_scales) == set(MODULE_NAMES)
    for group in FUSED_GROUPS:
        scales = [global_scales[name] for name in group]
        assert all(torch.equal(scales[0], scale) for scale in scales), group
    assert len({global_scales[name].item() for name in UNFUSED}) == len(UNFUSED)


@pytest.mark.parametrize(
    "names,expected",
    [
        # Gemma 4 attention_k_eq_v: q/k is a valid packed group without v_proj.
        (
            [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
            ],
            [
                {
                    "q_proj": "model.layers.0.self_attn.q_proj.weight",
                    "k_proj": "model.layers.0.self_attn.k_proj.weight",
                }
            ],
        ),
        # Standard q/k/v attention: optional v_proj joins when present.
        (
            [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
            ],
            [
                {
                    "q_proj": "model.layers.0.self_attn.q_proj.weight",
                    "k_proj": "model.layers.0.self_attn.k_proj.weight",
                    "v_proj": "model.layers.0.self_attn.v_proj.weight",
                }
            ],
        ),
        # Kimi KDA is the larger overlapping group and must win over q/k/v.
        (
            [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
                "model.layers.0.self_attn.b_proj.weight",
                "model.layers.0.self_attn.f_a_proj.weight",
            ],
            [
                {
                    "q_proj": "model.layers.0.self_attn.q_proj.weight",
                    "k_proj": "model.layers.0.self_attn.k_proj.weight",
                    "v_proj": "model.layers.0.self_attn.v_proj.weight",
                    "b_proj": "model.layers.0.self_attn.b_proj.weight",
                    "f_a_proj": "model.layers.0.self_attn.f_a_proj.weight",
                }
            ],
        ),
    ],
)
def test_get_fused_names_resolution_behavior(names, expected):
    assert get_fused_names(names) == expected


def test_get_fused_names_keeps_parent_modules_separate():
    names = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.self_attn.k_proj.weight",
        "model.layers.1.self_attn.v_proj.weight",
    ]

    groups = get_fused_names(names)

    assert groups == [
        {
            "q_proj": "model.layers.0.self_attn.q_proj.weight",
            "k_proj": "model.layers.0.self_attn.k_proj.weight",
            "v_proj": "model.layers.0.self_attn.v_proj.weight",
        },
        {
            "q_proj": "model.layers.1.self_attn.q_proj.weight",
            "k_proj": "model.layers.1.self_attn.k_proj.weight",
            "v_proj": "model.layers.1.self_attn.v_proj.weight",
        },
    ]


def test_microscale_groups_see_ignored_context_without_weight_names():
    """
    Ignored tensors of the job are context too, so an OLMo-Hybrid q/k/v/g_proj
    group with a_proj and b_proj ignored is fused without weight_names, as on
    main
    """
    attn = "model.layers.0.linear_attn"
    group = [f"{attn}.{n}.weight" for n in ("q_proj", "k_proj", "v_proj", "g_proj")]
    context = [f"{attn}.a_proj", f"{attn}.b_proj"]
    scheme = preset_name_to_scheme("NVFP4A16", ["Linear"])
    converter = ModelFreePtqConverter(
        QuantizationConfig(config_groups={"g": scheme}, ignore=context)
    )

    generator = torch.Generator().manual_seed(0)
    tensors = {
        name: torch.randn(32, 32, generator=generator) * (index + 1)
        for index, name in enumerate(group + [f"{n}.weight" for n in context])
    }
    tensors = converter.process(tensors)

    scales = [tensors[name + "_global_scale"] for name in group]
    assert all(torch.equal(scales[0], scale) for scale in scales)


def test_microscale_groups_see_context_in_other_shards():
    """
    A fused group is resolved on the whole checkpoint, so context layers that
    decide it can be stored in another shard than the group
    """
    attn = "model.layers.0.linear_attn"
    group = [f"{attn}.{n}.weight" for n in ("q_proj", "k_proj", "v_proj", "g_proj")]
    context = [f"{attn}.a_proj.weight", f"{attn}.b_proj.weight"]
    scheme = preset_name_to_scheme("NVFP4A16", ["Linear"])
    converter = ModelFreePtqConverter(
        QuantizationConfig(config_groups={"g": scheme}),
        weight_names=group + context,
    )

    generator = torch.Generator().manual_seed(0)
    tensors = {
        name: torch.randn(32, 32, generator=generator) * (index + 1)
        for index, name in enumerate(group)
    }
    tensors = converter.process(tensors)

    scales = [tensors[name + "_global_scale"] for name in group]
    assert all(torch.equal(scales[0], scale) for scale in scales)


MLA_LAYERS = ("q_a_proj", "kv_a_proj_with_mqa", "g_proj", "q_b_proj", "dense")
BAILING_KDA_LAYERS = ("q_proj", "k_proj", "v_proj", "b_proj", "g_proj", "f_proj")
KIMI_KDA_LAYERS = ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "f_b_proj")


@pytest.mark.parametrize(
    "checkpoint_layers, job_layers, expected",
    [
        # an optional layer outside the job does not join the group
        (
            MLA_LAYERS,
            ("q_a_proj", "kv_a_proj_with_mqa"),
            [("q_a_proj", "kv_a_proj_with_mqa")],
        ),
        (MLA_LAYERS, ("kv_a_proj_with_mqa", "g_proj"), []),
        # nor does a required layer: Bailing MoE V3 delta attention without
        # b_proj gives q/k/v, as on main. vLLM rejects a partly ignored packed
        # NVFP4 layer either way
        (
            BAILING_KDA_LAYERS,
            ("q_proj", "k_proj", "v_proj", "g_proj", "f_proj"),
            [("q_proj", "k_proj", "v_proj")],
        ),
        (
            KIMI_KDA_LAYERS,
            ("q_proj", "k_proj", "v_proj", "b_proj", "f_b_proj"),
            [("q_proj", "k_proj", "v_proj")],
        ),
        # a context layer outside the job still applies its mapping
        (
            ("q_proj", "k_proj", "v_proj", "g_proj", "a_proj", "b_proj"),
            ("q_proj", "k_proj", "v_proj", "g_proj"),
            [("q_proj", "k_proj", "v_proj", "g_proj")],
        ),
    ],
)
def test_get_fused_names_with_checkpoint_layers(
    checkpoint_layers, job_layers, expected
):
    """
    Checkpoint layers outside the job only satisfy context names: groups are
    made of the job's layers
    """
    attn = "model.layers.0.attention"
    index, _ = index_fused_weights(f"{attn}.{n}.weight" for n in checkpoint_layers)

    groups = get_fused_names([f"{attn}.{n}.weight" for n in job_layers], index)

    assert groups == [{n: f"{attn}.{n}.weight" for n in group} for group in expected]


def test_microscale_groups_leave_out_ignored_layers():
    """
    As in HIGGS, groups are resolved from the quantized weights. With Kimi
    f_a_proj ignored, q/k/v_proj share a global scale, and b_proj does not
    join them through the q/k/v/b/f_a group the ignored f_a_proj would select
    """
    attn = "model.layers.0.self_attn"
    names = [f"{attn}.{n}.weight" for n in KIMI_KDA_LAYERS]
    scheme = preset_name_to_scheme("NVFP4A16", ["Linear"])
    config = QuantizationConfig(
        config_groups={"g": scheme}, ignore=[f"{attn}.f_a_proj", f"{attn}.f_b_proj"]
    )
    converter = ModelFreePtqConverter(config, weight_names=names)

    generator = torch.Generator().manual_seed(0)
    tensors = {
        name: torch.randn(32, 32, generator=generator) * (index + 1)
        for index, name in enumerate(names)
    }
    tensors = converter.process(tensors)

    scale = {
        n: tensors[f"{attn}.{n}.weight_global_scale"] for n in ("q_proj", "v_proj")
    }
    b_scale = tensors[f"{attn}.b_proj.weight_global_scale"]
    assert torch.equal(scale["q_proj"], scale["v_proj"])
    assert not torch.equal(scale["q_proj"], b_scale)
