# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Replay ONE complete GLM decoder layer through QuaRot -> FlexSmooth on CPU.

Research handoff, not an inference/export pipeline. Only explicit selected-layer
weights are read. Synthetic identity boundaries expose the public Modifier API;
they are not evidence about the checkpoint embedding, final norm or LM head.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import torch
from compressed_tensors.utils import patch_attr
from safetensors import safe_open
from torch import nn
from transformers import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as glm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from llmcompressor.core import EventType  # noqa: E402
from llmcompressor.core.lifecycle import CompressionLifecycle  # noqa: E402
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D  # noqa: E402
from llmcompressor.modifiers.transform import (  # noqa: E402
    FlexSmoothModifier,
    QuaRotModifier,
)
from llmcompressor.modifiers.transform.flex_smooth.mappings import (  # noqa: E402
    glm_mappings,
)
from llmcompressor.modifiers.transform.flex_smooth.search import (  # noqa: E402
    search_alpha_beta,
)
from tests.flex_smooth.oracle import FlexOracle  # noqa: E402
from tests.quarot.modelslim_oracle import ModelSlimOracle  # noqa: E402
from tools.glm52_l4 import (  # noqa: E402
    all_passed,
    check_implementation_location,
    checkpoint_plan,
    compare,
    digest_tensor,
    reference_rotate,
    safe_output,
    stats,
)
from tools.glm52_precision_probe import local_file, read_header  # noqa: E402
from tools.run_quarot_checks import git_info  # noqa: E402

CACHE_KEY = "glm52_layer_l4"


class Gain(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width), requires_grad=False)

    def forward(self, x):
        return x * self.weight


class LayerHarness(nn.Module):
    """Identity input/output adapters around an unmodified HF decoder forward."""

    def __init__(self, config, layer):
        super().__init__()
        self.config = config
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([layer])
        width = config.hidden_size
        self.model.embed_tokens = nn.Embedding(width, width, device="meta")
        self.model.embed_tokens.weight = nn.Parameter(
            torch.eye(width), requires_grad=False
        )
        self.model.norm = Gain(width)
        self.lm_head = nn.Linear(width, width, bias=False, device="meta")
        self.lm_head.weight = nn.Parameter(torch.eye(width), requires_grad=False)

    def forward(self, values):
        x = values["hidden_states"] @ self.model.embed_tokens.weight
        output, topk = self.model.layers[0](x, **forward_kwargs(values))
        return self.lm_head(self.model.norm(output)), topk


def one_layer_config(config, layer):
    original = GlmMoeDsaConfig.from_dict(deepcopy(config))
    dense = layer < original.first_k_dense_replace
    if (original.mlp_layer_types[layer] == "dense") != dense:
        raise ValueError("MLP layer types disagree with QuaRot dense/MoE mapping")
    result = original.to_dict()
    result.update(
        num_hidden_layers=1,
        first_k_dense_replace=int(dense),
        mlp_layer_types=["dense" if dense else "sparse"],
        indexer_types=[original.indexer_types[layer]],
        layer_types=[original.layer_types[layer]],
        num_nextn_predict_layers=0,
        tie_word_embeddings=False,
        use_cache=False,
        attention_dropout=0.0,
    )
    result = GlmMoeDsaConfig.from_dict(result)
    result._attn_implementation = "eager"
    return result


def empty_layer(config):
    expert_type = LinearExperts2D.get_linear_experts_cls(glm.GlmMoeDsaExperts)
    with torch.device("meta"), patch_attr(glm, "GlmMoeDsaExperts", expert_type):
        return glm.GlmMoeDsaDecoderLayer(config, 0).eval()


def layer_plan(root, layer, max_weight_gib=2):
    """Header-only allowlist, including every expert and router correction buffer."""
    config, attention = checkpoint_plan(root, layer, max_weight_gib)
    cfg = one_layer_config(config, layer)
    expected = empty_layer(cfg).state_dict()
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(local_file(root, index.name).read_bytes())["weight_map"]
    else:
        weight_map = {
            key: "model.safetensors"
            for key in read_header(local_file(root, "model.safetensors"))
            if key != "__metadata__"
        }
    prefix = f"model.layers.{layer}."
    selected = {key[len(prefix) :] for key in weight_map if key.startswith(prefix)}
    if selected != set(expected):
        raise ValueError(
            "Complete layer requires explicit 2D expert checkpoint keys; "
            f"missing={sorted(set(expected) - selected)}, "
            f"unexpected={sorted(selected - set(expected))}"
        )
    entries, headers, size = {}, {}, 0
    for name, tensor in expected.items():
        key = prefix + name
        shard = weight_map[key]
        if shard not in headers:
            headers[shard] = read_header(local_file(root, shard))
        entry = headers[shard][key]
        if tuple(entry["shape"]) != tuple(tensor.shape) or entry["dtype"] not in (
            "F32",
            "BF16",
            "F16",
        ):
            raise ValueError(f"Invalid complete-layer baseline tensor: {key}")
        entries[name] = {"key": key, "shard": shard, **entry}
        size += math.prod(entry["shape"]) * 4
    if size > max_weight_gib * 2**30:
        raise ValueError(
            f"Complete FP32 layer needs {size / 2**30:.3f} GiB; "
            "increase --max-weight-gib explicitly; no payload read"
        )
    return cfg, {
        "layer": layer,
        "config_sha256": attention["config_sha256"],
        "selected_weights": entries,
        "selected_fp32_bytes": size,
        "indexer_type": cfg.indexer_types[0],
        "mlp_type": cfg.mlp_layer_types[0],
        "scope": "complete selected decoder layer; all routed/shared experts",
    }


def load_layer(root, cfg, plan):
    layer = empty_layer(cfg)
    parameters = set(dict(layer.named_parameters()))
    hashes = {}
    for name, entry in plan["selected_weights"].items():
        with safe_open(
            str(local_file(root, entry["shard"])), framework="pt", device="cpu"
        ) as stream:
            tensor = stream.get_tensor(entry["key"])
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite baseline: {name}")
        hashes[name] = digest_tensor(tensor)
        parent, leaf = name.rsplit(".", 1)
        module = layer.get_submodule(parent)
        tensor = tensor.float().clone()
        if name in parameters:
            tensor = nn.Parameter(tensor, requires_grad=False)
        setattr(module, leaf, tensor)
    return LayerHarness(cfg, layer).eval(), hashes


def forward_kwargs(values):
    return {
        "attention_mask": values["attention_mask"],
        "position_ids": values["position_ids"],
        "position_embeddings": (values["cos"], values["sin"]),
        "prev_topk_indices": values.get("prev_topk_indices"),
        "past_key_values": None,
        "use_cache": False,
    }


def validate_values(values, config):
    required = {"hidden_states", "attention_mask", "position_ids", "cos", "sin"}
    allowed = required | {"prev_topk_indices", "baseline_output", "baseline_topk"}
    if not required <= set(values) or set(values) - allowed:
        raise ValueError("Incomplete or unknown full-layer cache tensors")
    x = values["hidden_states"]
    if x.ndim != 3 or x.shape[0] != 1 or not 1 <= x.shape[1] <= 4096:
        raise ValueError("Capture exactly one prefill, batch=1, 1..4096 tokens")
    b, seq, h = x.shape
    if h != config.hidden_size:
        raise ValueError("Hidden width differs from checkpoint config")
    shapes = {
        "attention_mask": (b, 1, seq, seq),
        "position_ids": (b, seq),
        "cos": (b, seq, config.qk_rope_head_dim),
        "sin": (b, seq, config.qk_rope_head_dim),
        "baseline_output": (b, seq, h),
    }
    for name, value in values.items():
        if name in shapes and tuple(value.shape) != shapes[name]:
            raise ValueError(f"Unsupported prefill tensor shape: {name}")
        if name in ("position_ids", "prev_topk_indices", "baseline_topk"):
            if value.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"Integer indices required: {name}")
            if name != "position_ids" and (
                tuple(value.shape) != (b, seq, min(seq, config.index_topk))
                or value.min() < 0
                or value.max() >= seq
            ):
                raise ValueError(f"Invalid top-k indices: {name}")
        elif not value.is_floating_point() or (
            not torch.isfinite(value).all()
            if name != "attention_mask"
            else (torch.isnan(value).any() or torch.isposinf(value).any())
        ):
            raise ValueError(f"Invalid floating cache: {name}")
    # Support unpadded, standard causal prefill only. Do not silently replay an
    # unfamiliar mask or truncate context: either can change attention/top-k.
    mask = values["attention_mask"][0, 0]
    future = torch.ones(seq, seq, dtype=torch.bool).triu(1)
    if (mask[~future] != 0).any() or (mask[future] > -1e4).any():
        raise ValueError("Only an unpadded additive causal mask is supported")
    if config.indexer_types[0] == "shared" and "prev_topk_indices" not in values:
        raise ValueError("Shared indexer requires captured prev_topk_indices")


def read_cache(path, cfg, plan, hashes=None):
    header = read_header(path)
    if (
        sum(math.prod(v["shape"]) * 8 for k, v in header.items() if k != "__metadata__")
        > 2**29
    ):
        raise ValueError("Cache exceeds bounded 512 MiB allocation allowance")
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        metadata = json.loads((stream.metadata() or {}).get(CACHE_KEY, "{}"))
        if (
            metadata.get("schema_version") != 1
            or metadata.get("stage") != "baseline_complete_layer"
            or metadata.get("input_kind") not in ("real", "synthetic")
            or metadata.get("layer") != plan["layer"]
            or metadata.get("config_sha256") != plan["config_sha256"]
            or (hashes is not None and metadata.get("weight_sha256") != hashes)
        ):
            raise ValueError("Full-layer cache provenance does not match baseline")
        values = {name: stream.get_tensor(name) for name in stream.keys()}
    validate_values(values, cfg)
    if not {"baseline_output", "baseline_topk"} <= set(values):
        raise ValueError("Capture did not complete the selected layer")
    return {
        name: value.float() if value.is_floating_point() else value
        for name, value in values.items()
    }, metadata


@torch.no_grad()
def observe(model, values):
    """Collect comparable outputs and fresh Flex inputs from a genuine forward."""
    cache, trace, hooks = {}, {}, []
    for item in glm_mappings(model, ("norm-linear", "ov")):

        def capture(module, args, key=item.source):
            cache[key] = args[0].detach().reshape(-1, args[0].shape[-1]).clone()

        hooks.append(
            model.get_submodule(item.consumers[0]).register_forward_pre_hook(capture)
        )
    decoder = model.model.layers[0]

    def attention(module, args, result):
        trace["attention_output"] = result[0].detach().clone()
        if result[1] is not None:
            trace["attention_probabilities"] = result[1].detach().clone()

    def mlp(module, args, result):
        trace["mlp_output"] = result.detach().clone()

    def router(module, args, result):
        for key, value in zip(
            ("router_logits", "router_weights", "router_indices"), result, strict=True
        ):
            trace[key] = value.detach().clone()

    hooks.append(decoder.self_attn.register_forward_hook(attention))
    hooks.append(decoder.mlp.register_forward_hook(mlp))
    if hasattr(decoder.mlp, "gate"):
        hooks.append(decoder.mlp.gate.register_forward_hook(router))
    try:
        output, topk = model(values)
        trace.update(output=output, topk=topk)
    finally:
        for hook in hooks:
            hook.remove()
    return trace, cache


def trace_checks(actual, expected, hidden_rotation=None):
    checks = {}
    for name, value in expected.items():
        other = actual[name]
        if name in ("topk", "router_indices"):
            checks[name] = compare(
                other.sort(-1).values, value.sort(-1).values, exact=True
            )
        elif name == "router_weights":
            # topk(sorted=False) can change order without changing expert/weight pairs.
            ia = actual["router_indices"].argsort(-1)
            ib = expected["router_indices"].argsort(-1)
            checks[name] = compare(other.gather(-1, ia), value.gather(-1, ib), 1e-4)
        else:
            if hidden_rotation is not None and name in (
                "attention_output",
                "mlp_output",
            ):
                rotation = hidden_rotation.to(
                    device=value.device,
                    dtype=value.dtype,
                )
                value = value @ rotation
            checks[name] = compare(other, value, 1e-4)
    return checks


def state_checks(actual, reference):
    expected = reference.state_dict()
    return {
        name: compare(value, expected[name], 1e-6)
        for name, value in actual.state_dict().items()
    }


# A sparse MoE router introduces a discontinuity at top-k selection. A tiny,
# acceptable floating-point change in router logits can exchange two experts
# around the cutoff even when the continuous router computation is equivalent.
#
# These downstream traces remain valuable diagnostics, but an intermediate
# routing flip is not by itself evidence that the transform implementation is
# wrong when:
#   * router logits still satisfy the normal floating-point tolerance,
#   * every non-routing-sensitive intermediate trace passes,
#   * the independent ModelSlim differential passes, and
#   * the final composed transform remains strictly equivalent.
#
# Final composition routing is NOT relaxed.
_ROUTING_SENSITIVE_DOWNSTREAM = frozenset(
    {
        "router_indices",
        "router_weights",
        "mlp_output",
        "output",
    }
)


def intermediate_algebra_status(checks):
    """Classify intermediate sparse-MoE algebra without widening tolerances.

    A raw pass stays a pass. Otherwise the only tolerated pattern is a top-k
    routing flip with passing router logits and no failure outside the
    routing-dependent downstream traces.

    ModelSlim differential and final composition checks are separate hard gates;
    this function deliberately does not weaken them.
    """
    raw_passed = all_passed(checks)

    router_logits = checks.get("router_logits")
    router_indices = checks.get("router_indices")

    routing_sensitive = bool(
        not raw_passed
        and isinstance(router_logits, dict)
        and router_logits.get("passed") is True
        and isinstance(router_indices, dict)
        and router_indices.get("passed") is False
    )

    strict_checks = {
        name: value
        for name, value in checks.items()
        if name not in _ROUTING_SENSITIVE_DOWNSTREAM
    }
    strict_passed = all_passed(strict_checks)

    nonblocking_failures = sorted(
        name
        for name in _ROUTING_SENSITIVE_DOWNSTREAM
        if name in checks and not all_passed(checks[name])
    )

    passed = raw_passed or (
        routing_sensitive
        and strict_passed
        and "router_indices" in nonblocking_failures
    )

    return {
        "passed": passed,
        "raw_passed": raw_passed,
        "strict_passed": strict_passed,
        "routing_sensitive": routing_sensitive,
        "nonblocking_failures": nonblocking_failures,
        "criterion": (
            "raw intermediate algebra passes, or sparse top-k routing changes "
            "while router logits and all non-routing-dependent traces pass; "
            "final composition and ModelSlim differential remain hard gates"
        ),
    }


@torch.no_grad()
def reference_quarot(model, oracle, block):
    fusions, pre, stages, matrices = oracle.plan(model, block)
    for norm, targets in fusions.items():
        oracle.utils.fuse_ln_linear(
            [model.get_submodule(norm)], [model.get_submodule(name) for name in targets]
        )
    for pair in (pre, *stages.values()):
        for axis, mapping in ((0, pair.left_rot), (1, pair.right_rot)):
            for name, rotation in mapping.items():
                module = model.get_submodule(name)
                parts = rotation if isinstance(rotation, (tuple, list)) else [rotation]
                stride = sum(part.shape[0] for part in parts)
                offset = 0
                for part in parts:
                    module.weight = nn.Parameter(
                        reference_rotate(
                            oracle, module.weight, part, axis, stride, offset
                        ),
                        requires_grad=False,
                    )
                    offset += part.shape[0]
    return matrices["rot"]


@torch.no_grad()
def reference_flex(model, cache, oracle, max_tokens):
    details = {}
    for item in glm_mappings(model, ("norm-linear", "ov")):
        x = cache[item.source][:max_tokens]
        consumers = [model.get_submodule(name) for name in item.consumers]
        w = torch.cat([module.weight for module in consumers])
        searcher = oracle.search.FlexSmoothAlphaBetaSearcher()
        alpha, beta, loss = searcher.search_alpha_beta(x, w)
        candidate_losses = [
            searcher.evaluate_alpha_beta(x, w, round(i / 20, 2), 1 - round(i / 20, 2))
            for i in range(21)
        ] + [
            searcher.evaluate_alpha_beta(x, w, alpha, round(i / 20, 2))
            for i in range(21)
        ]
        same_input_search = search_alpha_beta(x, w)
        calculator = oracle.scales.FlexSmoothScaleCalculator(alpha, beta, "max")
        maxima = oracle.scales.compute_multi_weight_scale(
            [m.weight for m in consumers], w.dtype
        )
        if item.kind == "norm-linear":
            scale = calculator.compute_smooth_scale(x.abs().amax(0), maxima)
            subgraph = oracle.subgraphs.NormLinearSubgraph(
                model.get_submodule(item.source), consumers
            )
            oracle.fusion.NormLinearSubgraphFusion().apply_fusion(
                subgraph, {"scales": scale}
            )
        else:
            scale, v_scale = calculator.compute_ov_scales(
                x.abs().amax(0), maxima, item.heads, item.heads
            )
            source = model.get_submodule(item.source)
            view = source.weight.view(item.heads, item.key_dim + item.value_dim, -1)
            virtual = nn.Linear(
                source.weight.shape[1],
                item.heads * item.value_dim,
                bias=False,
                device="meta",
            )
            virtual.weight = nn.Parameter(
                view[:, item.key_dim :].reshape(-1, source.weight.shape[1]).clone(),
                requires_grad=False,
            )
            subgraph = oracle.subgraphs.OVSubgraph(
                consumers[0], virtual, item.heads, item.heads
            )
            oracle.fusion.OVSubgraphFusion().apply_fusion(
                subgraph, {"o_scales": scale, "v_scales": v_scale}
            )
            view[:, item.key_dim :].copy_(
                virtual.weight.reshape(item.heads, item.value_dim, -1)
            )
        details[item.source] = {
            "alpha": alpha,
            "beta": beta,
            "loss": loss,
            "scale": scale,
            "candidate_losses": candidate_losses,
            "same_input_losses": compare(
                torch.tensor(
                    same_input_search.alpha_losses + same_input_search.beta_losses
                ),
                torch.tensor(candidate_losses),
                exact=True,
            ),
        }
    return details


@torch.no_grad()
def run_validation(model, values, source, block=32, max_tokens=128):
    """Full-layer algebra plus independent original-source composed transforms."""
    reference = deepcopy(model)
    baseline, baseline_cache = observe(model, values)
    qoracle, foracle = ModelSlimOracle(source), FlexOracle(source)
    print("Original ModelSlim QuaRot: all complete-layer weights", flush=True)
    rotation = reference_quarot(reference, qoracle, block)
    modifier = FlexSmoothModifier(max_tokens=max_tokens)
    life = CompressionLifecycle()
    life.initialize(
        model=model,
        recipe=[
            QuaRotModifier(block_size=block, seed=1234, precision="float32"),
            modifier,
        ],
    )
    try:
        print("LLMC QuaRot -> fresh FlexSmooth calibration forward", flush=True)
        life.event(EventType.CALIBRATION_START)
        rotated, fresh = observe(model, values)
        reference_rotated, reference_cache = observe(reference, values)
        checks = {
            "quarot_algebra": trace_checks(rotated, baseline, rotation),
            "quarot_reference_weights": state_checks(model, reference),
            "quarot_reference_outputs": trace_checks(rotated, reference_rotated),
            "fresh_calibration": {
                key: compare(
                    torch.cat(modifier._cache[key]), value[:max_tokens], exact=True
                )
                for key, value in fresh.items()
            },
            "reference_calibration": {
                key: compare(
                    value[:max_tokens], reference_cache[key][:max_tokens], 1e-6
                )
                for key, value in fresh.items()
            },
        }
        print("Original ModelSlim FlexSmooth: post-QuaRot activations", flush=True)
        actual_search = {
            item.source: search_alpha_beta(
                fresh[item.source][:max_tokens],
                torch.cat(
                    [model.get_submodule(name).weight for name in item.consumers]
                ),
            )
            for item in glm_mappings(model, ("norm-linear", "ov"))
        }
        expected = reference_flex(reference, reference_cache, foracle, max_tokens)
        life.event(EventType.SEQUENTIAL_EPOCH_END, modules=list(model.modules()))
        life.event(EventType.CALIBRATION_END)
        life.finalize()
    except Exception:
        life.reset()
        raise
    final, _ = observe(model, values)
    reference_final, _ = observe(reference, values)
    checks.update(
        composition_algebra=trace_checks(final, baseline, rotation),
        flex_algebra=trace_checks(final, rotated),
        composition_reference_outputs=trace_checks(final, reference_final),
        composition_reference_weights=state_checks(model, reference),
        flex_parameters={
            key: {
                "alpha_beta": {
                    "passed": (
                        modifier.diagnostics[key]["alpha"],
                        modifier.diagnostics[key]["beta"],
                    )
                    == (value["alpha"], value["beta"])
                },
                "scale": compare(
                    modifier.diagnostics[key]["scale"], value["scale"], 1e-6
                ),
                "candidate_losses": compare(
                    torch.tensor(
                        actual_search[key].alpha_losses + actual_search[key].beta_losses
                    ),
                    torch.tensor(value["candidate_losses"]),
                    1e-5,
                ),
                "same_input_candidate_losses": value["same_input_losses"],
                "applied_search": {
                    "passed": modifier.diagnostics[key]["loss"]
                    == actual_search[key].loss
                },
            }
            for key, value in expected.items()
        },
        lifecycle={
            "passed": not modifier._hooks
            and not modifier._cache
            and model.config.quarot_config["status"] == "applied"
            and model.config.flex_smooth_config["status"] == "applied"
        },
    )
    intermediate_algebra = {
        "quarot": intermediate_algebra_status(checks["quarot_algebra"]),
        "flex_smooth": intermediate_algebra_status(checks["flex_algebra"]),
    }

    # Intermediate sparse-routing sensitivity is adjudicated above. Everything
    # else remains a hard gate, especially ModelSlim differential and the final
    # QuaRot -> FlexSmooth composition.
    blocking_checks = {
        name: value
        for name, value in checks.items()
        if name not in ("quarot_algebra", "flex_algebra")
    }
    numerical_passed = all_passed(blocking_checks) and all(
        status["passed"] for status in intermediate_algebra.values()
    )

    seq = values["hidden_states"].shape[1]
    return {
        "checks": checks,
        "intermediate_algebra": intermediate_algebra,
        "numerical_passed": numerical_passed,
        "source_sha256": {"quarot": qoracle.hashes, "flex_smooth": foracle.hashes},
        "calibration": {
            key: {
                "baseline_sha256": digest_tensor(baseline_cache[key][:max_tokens]),
                "post_quarot_sha256": digest_tensor(value[:max_tokens]),
                "post_quarot_statistics": stats(value[:max_tokens]),
                **{k: v for k, v in modifier.diagnostics[key].items() if k != "scale"},
                "scale_statistics": stats(modifier.diagnostics[key]["scale"]),
                "alpha_losses": list(actual_search[key].alpha_losses),
                "beta_losses": list(actual_search[key].beta_losses),
                "reference_candidate_losses": expected[key]["candidate_losses"],
            }
            for key, value in fresh.items()
        },
        "indexer_selective": model.config.index_topk < seq,
        "indexer_origin": model.config.indexer_types[0],
        "routed_experts_exercised": sorted(final["router_indices"].unique().tolist())
        if "router_indices" in final
        else [],
        "baseline_capture_diagnostic": compare(
            baseline["output"], values["baseline_output"], 1e-4
        )
        if "baseline_output" in values
        else None,
        "limits": [
            "CPU FP32 replay; captured BF16 output is diagnostic only",
            "Synthetic boundaries; real embedding/final norm/LM head/MTP excluded",
            "One layer and one prefill; shared-indexer producer is outside the replay",
            "All expert weights compared; only listed experts exercised by input",
            "No MXFP, NPU, decode, deployment or task accuracy validation",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--activation-cache", type=Path, required=True)
    parser.add_argument("--modelslim-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-weight-gib", type=float, default=2)
    parser.add_argument("--max-estimated-memory-gib", type=float, default=16)
    parser.add_argument("--calibration-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    destination = safe_output(
        args.output, [args.model_dir, args.modelslim_source, args.activation_cache]
    )
    report = {
        "schema_version": 1,
        "full_layer_l4_passed": False,
        "dry_run": args.dry_run,
    }
    code = 2
    try:
        if not 1 <= args.calibration_tokens <= 128 or not 1 <= args.threads <= 64:
            raise ValueError("calibration-tokens must be 1..128; threads 1..64")
        if (
            not math.isfinite(args.max_estimated_memory_gib)
            or args.max_estimated_memory_gib <= 0
        ):
            raise ValueError("Memory budget must be finite and positive")
        torch.set_num_threads(args.threads)
        check_implementation_location()
        cfg, plan = layer_plan(
            args.model_dir.resolve(), args.layer, args.max_weight_gib
        )
        values, metadata = read_cache(args.activation_cache, cfg, plan)
        seq = values["hidden_states"].shape[1]
        estimate = (
            5 * plan["selected_fp32_bytes"]
            + 12 * cfg.hidden_size**2 * 4
            + 8 * cfg.num_attention_heads * seq**2 * 4
        )
        report.update(
            plan=plan,
            cache_metadata=metadata,
            cache_sha256=hashlib.sha256(args.activation_cache.read_bytes()).hexdigest(),
            runtime_versions={
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "compressed-tensors", "numpy")
            },
            git={"llmc": git_info(REPO), "modelslim": git_info(args.modelslim_source)},
            estimated_peak_gib=estimate / 2**30,
            memory_estimate_is_hard_limit=False,
        )
        if estimate > args.max_estimated_memory_gib * 2**30:
            raise ValueError(
                "Estimated CPU memory exceeds explicit budget; no weight payload read"
            )
        report["implementation_sha256"] = {
            str(path.relative_to(REPO)): hashlib.sha256(
                path.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for path in [
                Path(__file__),
                REPO / "tools/glm52_l4.py",
                REPO / "tools/glm52_layer_l4_capture.py",
                REPO / "tools/glm52_precision_probe.py",
                REPO / "tests/quarot/modelslim_oracle.py",
                REPO / "tests/quarot/source_loader.py",
                REPO / "tests/flex_smooth/oracle.py",
                REPO / "src/llmcompressor/modeling/fuse.py",
                REPO / "src/llmcompressor/modeling/moe/linear_experts.py",
                *sorted(
                    (REPO / "src/llmcompressor/modifiers/transform/quarot").glob("*.py")
                ),
                *sorted(
                    (REPO / "src/llmcompressor/modifiers/transform/flex_smooth").glob(
                        "*.py"
                    )
                ),
            ]
            if path.is_relative_to(REPO)
        }
        report["transformers_model_sha256"] = hashlib.sha256(
            Path(inspect.getfile(glm)).read_bytes()
        ).hexdigest()
        if args.dry_run:
            code = 0
        else:
            model, hashes = load_layer(args.model_dir.resolve(), cfg, plan)
            values, metadata = read_cache(args.activation_cache, cfg, plan, hashes)
            report.update(
                run_validation(
                    model,
                    values,
                    args.modelslim_source.resolve(),
                    max_tokens=args.calibration_tokens,
                )
            )
            report["full_layer_l4_passed"] = (
                report["numerical_passed"] and metadata["input_kind"] == "real"
            )
            code = (
                0
                if report["full_layer_l4_passed"]
                else (3 if report["numerical_passed"] else 1)
            )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        f"Report: {destination}; full_layer_l4_passed={report['full_layer_l4_passed']}",
        flush=True,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
