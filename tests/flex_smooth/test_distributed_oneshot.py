# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The real three-modifier sequential path, with CT shared backing on two ranks."""

import json
import os
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from compressed_tensors.offload import disable_offloading, offload_module
from compressed_tensors.offload.cache import DistributedCPUCache
from compressed_tensors.offload.module import remove_module_offload
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    CompressedTensorsConfig,
    GlmMoeDsaConfig,
    GlmMoeDsaForCausalLM,
    PreTrainedTokenizerFast,
)

from examples.glm52_precision.oneshot_profile import (
    Profile,
    calibration_preflight,
    model_preflight,
    quantized_tiles,
)
from llmcompressor import oneshot
from llmcompressor.modeling.moe.linearize import linearize_moe
from llmcompressor.modifiers.transform import FlexSmoothModifier, QuaRotModifier
from llmcompressor.modifiers.transform.flex_smooth import base as flex_base
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.sequential import pipeline as sequential_pipeline
from llmcompressor.pipelines.sequential.helpers import Subgraph
from llmcompressor.recipe import Recipe
from tests.flex_smooth.compression_trace import CompressionTrace, state_signature
from tests.flex_smooth.oneshot_trace import OneshotTrace
from tests.flex_smooth.test_transformers import _LinearizedGlm


def _fixture(directory):
    config = GlmMoeDsaConfig.from_pretrained(directory / "input", local_files_only=True)
    config._name_or_path = str(directory / "input")
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = GlmMoeDsaForCausalLM(config).float().eval()
        linearize_moe(model)
    with torch.no_grad():
        for module in model.modules():
            if type(module).__name__.endswith("RMSNorm"):
                module.weight.copy_(torch.linspace(0.6, 1.4, module.weight.numel()))
    backend = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "[PAD]": 1, **{f"w{i}": i + 2 for i in range(90)}},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    return model, tokenizer


def _snapshot(model):
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _run(model, tokenizer, loader):
    recipe = Recipe.create_instance(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/glm52_precision/mixed_mxfp.yaml"
        )
    ).modifiers
    recipe[0].precision = torch.float32
    recipe[1].max_tokens = None
    # SequentialPipeline overwrites the model's onload device using host detection;
    # sequential_offload_device only controls activation backing. Pin this test's
    # pipeline lookup (also in spawned workers), leaving global detection untouched.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            sequential_pipeline, "get_main_device", lambda: torch.device("cpu")
        )
        oneshot(
            model=model,
            processor=tokenizer,
            dataset=loader,
            recipe=recipe,
            pipeline="sequential",
            sequential_offload_device="cpu",
            sequential_targets_per_subgraph=1,
            sequential_targets=[
                r"re:.*\.input_layernorm$",
                "ExpertMLP",
                "GlmMoeDsaMLP",
            ],
            propagate_error=True,
            moe_calibrate_all_experts=False,
        )
    return recipe[1].diagnostics


def _data(directory, model, tokenizer, report, samples=4):
    args = SimpleNamespace(
        dataset=str(directory / "prompts.json"),
        samples=samples,
        sequence_length=16,
        batch_size=2,
    )
    model_preflight(model, FlexSmoothModifier(max_tokens=None), report)
    loader, _ = calibration_preflight(
        args, report, tokenizer=tokenizer, generation_config=model.generation_config
    )
    return loader


@torch.no_grad()
def _worker(rank, store, directory, kind):
    torch.set_num_threads(1)
    directory = Path(directory)
    expected = torch.load(directory / "reference.pt", weights_only=True)
    (directory / "offload").mkdir(exist_ok=True)
    tokens = expected["tokens"]
    dist.init_process_group(
        "gloo",
        init_method=store,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    assert dist.get_backend() == "gloo"
    model, tokenizer = _fixture(directory)
    for module in model.modules():
        remove_module_offload(module, onload_tensors=True)
        offload_module(
            module,
            onload_device="cpu",
            offload_device=kind,
            **({"offload_dir": str(directory / "offload")} if kind == "disk" else {}),
        )
    rotation_calls = []
    with pytest.MonkeyPatch.context() as patch:
        compression = CompressionTrace(patch, model)
        trace = OneshotTrace(patch)
        rotate = QuaRotModifier._rotate
        start = QuaRotModifier.on_calibration_start

        def record_rotate(modifier, *args, **kwargs):
            rotation_calls.append(1)
            return rotate(modifier, *args, **kwargs)

        def check_rotated(modifier, state, event, **kwargs):
            start(modifier, state, event, **kwargs)
            actual = _snapshot(state.model)
            for name, value in expected["rotated"].items():
                if name.endswith((".weight", ".bias")):
                    torch.testing.assert_close(
                        actual[name], value, atol=0, rtol=0, msg=name
                    )

        patch.setattr(QuaRotModifier, "_rotate", record_rotate)
        patch.setattr(QuaRotModifier, "on_calibration_start", check_rotated)
        # Stand in for private accelerator residents using CPU copies of weights.
        # Quantization scale tensors retain native CPU behavior (shared backing).
        onload = DistributedCPUCache.onload

        def separate_weight_resident(cache, value):
            result = onload(cache, value)
            return (
                result.clone()
                if any(
                    cache.offloaded_values.get(key) is value
                    for key in ("weight", "bias")
                )
                else result
            )

        if kind == "cpu":
            patch.setattr(DistributedCPUCache, "onload", separate_weight_resident)
        profile = Profile("cpu", attribution=True)
        profile.report.update(rank=rank, world_size=2, device_map="auto_offload")
        uneven = {}
        _data(directory, model, tokenizer, uneven, samples=3)
        assert uneven["samples_local"] == rank + 1
        assert uneven["samples_global"] == 3
        assert uneven["tokens_global_valid"] == 9
        with pytest.raises(ValueError, match="requires max_tokens=None"):
            model_preflight(
                model,
                FlexSmoothModifier(max_tokens=128 if rank else None),
                {},
            )
        loader = _data(directory, model, tokenizer, profile.report)
        # The finite distributed cap must fail before the QuaRot callback runs.
        with pytest.raises(ValueError, match="requires max_tokens=None"):
            FlexSmoothModifier(max_tokens=128)._preflight(model)
        assert not rotation_calls
        rows = [row["input_ids"] for row in loader.dataset]
        partitions = [None, None]
        dist.all_gather_object(partitions, rows)
        assert not set(map(tuple, partitions[0])) & set(map(tuple, partitions[1]))
        assert partitions[0] + partitions[1] == expected["rows"]
        assert profile.report["samples_local"] == 2
        assert profile.report["tokens_local_valid"] == 7
        assert profile.report["tokens_global_valid"] == 14

        forwards = Counter()
        forward = Subgraph.forward

        def record_forward(subgraph, model, **kwargs):
            phase = "propagation" if HooksMixin._HOOKS_DISABLED else "calibration"
            result = forward(subgraph, model, **kwargs)
            forwards[(id(subgraph), phase)] += 1
            return result

        patch.setattr(Subgraph, "forward", record_forward)
        # Observe the real activation-derived collective; never replace its result.
        maxima = []
        search = flex_base.search_alpha_beta_distributed

        def record_search(activations, weights, **kwargs):
            assert dist.get_backend() == "gloo"
            assert activations.device.type == "cpu", activations.device
            assert weights.device.type == "cpu", weights.device
            local = activations.abs().amax(0).cpu()
            peer_maxima = [None, None]
            dist.all_gather_object(peer_maxima, local)
            reductions = Counter()
            reduce = dist.all_reduce

            def record_reduce(tensor, op=dist.ReduceOp.SUM, **options):
                reductions[str(op)] += 1
                return reduce(tensor, op=op, **options)

            with pytest.MonkeyPatch.context() as reduction_patch:
                reduction_patch.setattr(dist, "all_reduce", record_reduce)
                result = search(activations, weights, **kwargs)
            assert reductions[str(dist.ReduceOp.MAX)] == 1
            assert reductions[str(dist.ReduceOp.SUM)] > 0
            torch.testing.assert_close(
                result.activation_max.cpu(),
                torch.stack(peer_maxima).amax(0),
                atol=0,
                rtol=0,
            )
            maxima.append(
                {
                    "local": local.tolist(),
                    "global": result.activation_max.tolist(),
                    "reductions": dict(reductions),
                }
            )
            return result

        patch.setattr(flex_base, "search_alpha_beta_distributed", record_search)
        with profile.instrument():
            diagnostics = _run(model, tokenizer, loader)
        assert len(forwards) == 14  # Every local batch runs all seven subgraphs twice.
        assert set(forwards.values()) == {len(loader)}
        assert profile.stages["flex_search"]["calls"] == 6
        assert profile.stages["flex_apply"]["calls"] == 6
        assert profile.stages["calibration_forward"]["calls"] == 7
        assert profile.stages["propagation"]["calls"] == 7
        assert bool(rotation_calls) == (rank == 0)
        assert trace.events.index("quarot.applied") < trace.events.index("flex.start")
        assert trace.events.index("flex.start") < trace.events.index("quant.start")
        assert trace.captures > 0
        # OneshotTrace asserts post-Flex resident weights inside weight observe.
        observed = [None, None]
        dist.all_gather_object(observed, sorted(trace.observed))
        assert len(set(observed[0]) | set(observed[1])) >= 6
        actual = _snapshot(model)
        for name, value in expected["final"].items():
            torch.testing.assert_close(actual[name], value, atol=2e-6, rtol=2e-5)
        for name, value in diagnostics.items():
            baseline = expected["diagnostics"][name]
            assert (value["alpha"], value["beta"]) == (
                baseline["alpha"],
                baseline["beta"],
            )
            torch.testing.assert_close(
                value["scale"], baseline["scale"], atol=2e-6, rtol=2e-5
            )
            # Flex's reference capture includes padded rows; profile counts valid only.
            assert value["tokens_used"] == 20
        shared = [None, None]
        dist.all_gather_object(shared, (diagnostics, state_signature(model)))
        assert shared[0][1] == shared[1][1]
        for name in diagnostics:
            left, right = shared[0][0][name], shared[1][0][name]
            assert (left["alpha"], left["beta"]) == (right["alpha"], right["beta"])
            torch.testing.assert_close(left["scale"], right["scale"], atol=0, rtol=0)
        with disable_offloading():
            logits = model(tokens).logits
        tiles = quantized_tiles(model)
        torch.testing.assert_close(logits, expected["logits"], atol=2e-6, rtol=2e-5)
        rotary = model.model.rotary_emb.inv_freq.detach().clone()
        dist.barrier()
        with profile.attribution_scope("save"):
            model.save_pretrained(directory / "compressed", save_compressed=True)
        attribution = profile.report["attribution"]
        assert not attribution["unavailable_hooks"]
        operations = attribution["operations"]
        assert operations["initialization/initialize_qparams"]["calls"] > 0
        assert any(
            key.startswith("initialization/")
            and key.endswith(".offload")
            and value["calls"] > 0
            for key, value in attribution["cache_by_phase_backend"].items()
        )
        assert operations["save.compress/total"]["calls"] == 1
        assert operations["save.restore_dispatch/total"]["calls"] == 1
        assert operations.get("save/write_shard", {}).get("calls", 0) == int(rank == 0)
        assert bool(attribution["quarot_segments"]) == (rank == 0)
        signatures = [None, None]
        dist.all_gather_object(signatures, state_signature(model))
        assert signatures[0] == signatures[1]
        assert compression.parallel_calls == 1
        assert compression.real and compression.meta
        calls = [None, None]
        dist.all_gather_object(calls, compression.real)
        assert set(calls[0]).isdisjoint(calls[1])
        assert sorted(calls[0] + calls[1]) == expected["compressed_modules"]
        assert compression.writes == int(rank == 0)
        torch.testing.assert_close(
            model.model.rotary_emb.inv_freq, rotary, atol=0, rtol=0
        )
    dist.destroy_process_group()
    if rank == 0:
        _reload(directory, tiles, tokens, logits)
    report = {
        "rank": rank,
        "backing": kind,
        "search_device": "cpu",
        "backend": "gloo",
        "real_weights": False,
        "rotation_calls": len(rotation_calls),
        "events": trace.events,
        "sample_rows": rows,
        "uneven_partition": uneven,
        "subgraphs_per_rank": len(forwards) // 2,
        "batches_per_subgraph_per_phase": len(loader),
        "activation_max": maxima,
        "global_flex_parameters_match_single_rank": True,
        "cross_rank_flex_parameters_and_weights_exact": True,
        "flex_parameters": {
            name: {
                key: value[key].tolist() if key == "scale" else value[key]
                for key in ("alpha", "beta", "scale", "tokens_used")
            }
            for name, value in diagnostics.items()
        },
        "post_flex_observer": True,
        "shared_weights_match_single_rank": True,
        "compression": compression.report(),
        "compressed_state_equal_across_ranks": True,
        "compressed_reload_exact": rank == 0,
        "profile": profile.report,
    }
    (directory / f"rank-{rank}.json").write_text(json.dumps(report, indent=2))


def _reload(directory, tiles, tokens, logits):
    assert not dist.is_initialized()
    restored, loading = _LinearizedGlm.from_pretrained(
        directory / "compressed",
        local_files_only=True,
        attn_implementation="eager",
        dtype=torch.float32,
        quantization_config=CompressedTensorsConfig(dequantize=True),
        output_loading_info=True,
    )
    assert not any(loading.values()), loading
    restored.float().eval()
    actual_tiles = quantized_tiles(restored, dequantized=True)
    for name in tiles:
        for key in ("weight", "scale"):
            torch.testing.assert_close(
                actual_tiles[name][key], tiles[name][key], atol=0, rtol=0
            )
    torch.testing.assert_close(restored(tokens).logits, logits, atol=0, rtol=0)


def _config(tmp_path):
    GlmMoeDsaConfig(
        vocab_size=96,
        hidden_size=64,
        intermediate_size=96,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=32,
        first_k_dense_replace=1,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        index_topk=2,
        index_head_dim=64,
        index_n_heads=2,
        indexer_types=["full", "shared"],
        max_position_embeddings=64,
        use_cache=False,
        attn_implementation="eager",
    ).save_pretrained(tmp_path / "input")


@pytest.mark.parametrize("kind", ["cpu", "disk"])
@torch.no_grad()
def test_distributed_sequential_roundtrip(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    _config(tmp_path)
    model, tokenizer = _fixture(tmp_path)
    (tmp_path / "prompts.json").write_text(
        json.dumps(["w1 w2", "w3 w4 w5 w6 w7", "w8 w9", "w10 w11 w12 w13 w14"])
    )
    loader = _data(tmp_path, model, tokenizer, {})
    tokens = torch.tensor([[1, 7, 4, 5, 9], [9, 3, 5, 1, 7]])
    rotated = {}
    start = QuaRotModifier.on_calibration_start

    def record_rotated(modifier, state, event, **kwargs):
        start(modifier, state, event, **kwargs)
        rotated.update(_snapshot(state.model))

    compression = CompressionTrace(monkeypatch, model)
    with monkeypatch.context() as patch:
        patch.setattr(QuaRotModifier, "on_calibration_start", record_rotated)
        diagnostics = _run(model, tokenizer, loader)
    with disable_offloading():
        logits = model(tokens).logits
    reference = {
        "rotated": rotated,
        "final": _snapshot(model),
        "diagnostics": diagnostics,
        "tokens": tokens,
        "logits": logits,
        "rows": [row["input_ids"] for row in loader.dataset],
    }
    model.save_pretrained(tmp_path / "single-compressed", save_compressed=True)
    assert compression.parallel_calls == 0 and not compression.meta
    assert compression.writes == 1
    assert len(compression.real) == len(set(compression.real)) > 0
    reference["compressed_modules"] = sorted(compression.real)
    torch.save(reference, tmp_path / "reference.pt")
    context = mp.spawn(
        _worker,
        args=((tmp_path / "store").as_uri(), str(tmp_path), kind),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 180
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Tiny distributed oneshot exceeded 180s")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    reports = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)
    ]
    if output := os.environ.get("FLEXSMOOTH_REPORT_DIR"):
        Path(output, f"distributed-oneshot-{kind}.json").write_text(
            json.dumps(
                {"sequential_compression": compression.report(), "ranks": reports},
                indent=2,
            )
        )
