# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded CPU L4 diagnostics for selected GLM attention subgraphs, not a model run."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from llmcompressor.modeling import fuse_norm_linears  # noqa: E402
from llmcompressor.modifiers.transform.flex_smooth.mappings import (  # noqa: E402
    SmoothMapping,
    apply_scales,
)
from llmcompressor.modifiers.transform.flex_smooth.search import (  # noqa: E402
    ov_scales,
    search_alpha_beta,
    smooth_scale,
)
from llmcompressor.modifiers.transform.quarot.rotation import (  # noqa: E402
    make_hadamard_rotation,
    rotate_axis,
)
from tests.flex_smooth.oracle import FlexOracle  # noqa: E402
from tests.quarot.modelslim_oracle import ModelSlimOracle, preserve_rng  # noqa: E402
from tools.glm52_precision_probe import local_file, read_header  # noqa: E402

ATTN = "self_attn."
NORMS = ("input_layernorm", ATTN + "q_a_layernorm", ATTN + "kv_a_layernorm")
PROJECTIONS = tuple(
    ATTN + name
    for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
)
INDEXER = tuple(ATTN + "indexer." + name for name in ("wk", "weights_proj", "wq_b"))
CACHE_MODULES = {
    "input_norm": "input_layernorm",
    "q_norm": ATTN + "q_a_layernorm",
    "kv_norm": ATTN + "kv_a_layernorm",
    "o": ATTN + "o_proj",
    "kv_b": ATTN + "kv_b_proj",
}
DIMENSIONS = (
    "hidden_size",
    "q_lora_rank",
    "kv_lora_rank",
    "num_attention_heads",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
)


def digest_tensor(value):
    value = value.detach().cpu().contiguous()
    signature = f"{value.dtype}:{tuple(value.shape)}:".encode()
    return hashlib.sha256(
        signature + value.view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def check_implementation_location():
    for function, relative in (
        (fuse_norm_linears, "modeling/fuse.py"),
        (make_hadamard_rotation, "modifiers/transform/quarot/rotation.py"),
        (search_alpha_beta, "modifiers/transform/flex_smooth/search.py"),
        (apply_scales, "modifiers/transform/flex_smooth/mappings.py"),
    ):
        actual = Path(inspect.getfile(inspect.unwrap(function))).resolve()
        expected = (REPO / "src/llmcompressor" / relative).resolve()
        if actual != expected:
            raise ValueError(
                f"LLMC imported from {actual}; "
                "use this checkout's editable installation"
            )


def checkpoint_plan(root, layer, max_weight_gib=2.0):
    """Read headers; select only one layer's attention and three norm weights."""
    root = root.resolve(strict=True)
    config_bytes = local_file(root, "config.json").read_bytes()
    config = json.loads(config_bytes)
    if config.get("quantization_config") or any(
        config.get(key) for key in ("quarot_config", "flex_smooth_config")
    ):
        raise ValueError("Use an untransformed floating-point checkpoint")
    if (
        not isinstance(config.get("num_hidden_layers"), int)
        or isinstance(config["num_hidden_layers"], bool)
        or not 0 <= layer < config["num_hidden_layers"]
    ):
        raise ValueError("Layer must be an existing decoder layer, not MTP")
    for key in DIMENSIONS:
        if (
            not isinstance(config.get(key), int)
            or isinstance(config[key], bool)
            or config[key] <= 0
        ):
            raise ValueError(f"Invalid config dimension: {key}")
    h, q, c, heads, k, r, v = (config[key] for key in DIMENSIONS)
    if config.get("num_key_value_heads", heads) != heads:
        raise ValueError("Only expanded MLA KV-B heads are supported")
    eps = config.get("rms_norm_eps", 1e-6)
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Invalid rms_norm_eps")
    headers = {}
    if (root / "model.safetensors.index.json").is_file():
        index = json.loads(
            local_file(root, "model.safetensors.index.json").read_bytes()
        )
        weight_map = index["weight_map"]
    else:
        headers["model.safetensors"] = read_header(
            local_file(root, "model.safetensors")
        )
        weight_map = {
            name: "model.safetensors"
            for name in headers["model.safetensors"]
            if name != "__metadata__"
        }
    prefix = f"model.layers.{layer}."
    indexed = [prefix + name + ".weight" in weight_map for name in INDEXER]
    if any(indexed) and not all(indexed):
        raise ValueError("Incomplete indexer projections")
    kinds = config.get("indexer_types")
    if kinds:
        if kinds[layer] not in ("full", "shared"):
            raise ValueError("Unknown indexer type; return the header probe first")
        if (kinds[layer] == "full") != all(indexed):
            raise ValueError("Indexer config and selected checkpoint weights disagree")
    shapes = dict(
        zip(
            (*NORMS, *PROJECTIONS),
            (
                (h,),
                (q,),
                (c,),
                (q, h),
                (heads * (k + r), q),
                (c + r, h),
                (heads * (k + v), c),
                (h, heads * v),
            ),
            strict=True,
        )
    )
    selected, fp32_bytes = {}, 0
    for name in (*NORMS, *PROJECTIONS, *(INDEXER if all(indexed) else ())):
        full = prefix + name + ".weight"
        if prefix + name + ".bias" in weight_map:
            raise ValueError(
                f"This diagnostic requires bias-free projections/norms: {name}"
            )
        if full not in weight_map:
            raise ValueError(f"Missing selected weight: {full}")
        shard = weight_map[full]
        path = local_file(root, shard)
        if shard not in headers:
            headers[shard] = read_header(path)
        entry = headers[shard][full]
        shape = tuple(entry["shape"])
        if entry["dtype"] not in ("BF16", "F16", "F32"):
            raise ValueError(f"Non-floating baseline: {full}")
        if name in shapes and shape != shapes[name]:
            raise ValueError(f"Shape mismatch: {full}: {shape} != {shapes[name]}")
        if name in INDEXER and (
            len(shape) != 2
            or shape[0] <= 0
            or shape[1] != (q if name.endswith("wq_b") else h)
        ):
            raise ValueError(f"Invalid indexer shape: {full}")
        fp32_bytes += math.prod(shape) * 4
        selected[name] = {"key": full, "shard": shard, **entry}
    if not math.isfinite(max_weight_gib) or max_weight_gib <= 0:
        raise ValueError("max-weight-gib must be finite and positive")
    if fp32_bytes > max_weight_gib * 2**30:
        raise ValueError(
            "Selected FP32 weights exceed --max-weight-gib; no payload read"
        )
    return config, {
        "layer": layer,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "selected_weights": selected,
        "selected_fp32_bytes": fp32_bytes,
        "payload_scope": (
            "one decoder layer: attention projections and three norms only"
        ),
    }


def load_selected(root, plan):
    weights, fingerprints = {}, {}
    for name, entry in plan["selected_weights"].items():
        with safe_open(
            str(local_file(root.resolve(), entry["shard"])),
            framework="pt",
            device="cpu",
        ) as f:
            value = f.get_tensor(entry["key"])
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite weight: {name}")
        fingerprints[name] = digest_tensor(value)
        weights[name] = value.float()
    return weights, fingerprints


def shell(weights):
    """Only selected modules; no Transformers model construction or random weights."""
    model = nn.Module()
    for name, value in weights.items():
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            if not hasattr(parent, part):
                parent.add_module(part, nn.Module())
            parent = getattr(parent, part)
        leaf = (
            nn.Module()
            if value.ndim == 1
            else nn.Linear(value.shape[1], value.shape[0], bias=False, device="meta")
        )
        leaf.weight = nn.Parameter(value.clone(), requires_grad=False)
        parent.add_module(parts[-1], leaf)
    return model


def compare(actual, expected, tolerance=1e-5, *, exact=False):
    if actual.shape != expected.shape:
        raise ValueError(
            f"Comparison shape mismatch: {actual.shape} vs {expected.shape}"
        )
    a, b = actual.detach().double(), expected.detach().double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    delta = a - b
    maximum = delta.abs().max().item() if finite else None
    relative = (
        (
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(b).clamp_min(1e-30)
        ).item()
        if finite
        else None
    )
    passed = finite and (
        torch.equal(actual, expected) if exact else relative <= tolerance
    )
    return {
        "passed": passed,
        "max_abs": maximum,
        "relative_l2": relative,
        "nonfinite_actual": int((~torch.isfinite(a)).sum()),
        "nonfinite_reference": int((~torch.isfinite(b)).sum()),
        "criterion": "exact" if exact else f"relative_l2 <= {tolerance}",
    }


def stats(value):
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "abs_max": value.abs().max().item(),
        "rms": value.double().square().mean().sqrt().item(),
    }


def norm(x, gain, eps):
    return x * (x.square().mean(-1, keepdim=True) + eps).rsqrt() * gain


def groups(weights, config):
    heads, k, v = (
        config[key] for key in ("num_attention_heads", "qk_nope_head_dim", "v_head_dim")
    )
    inputs, queries = list(PROJECTIONS[:1]) + [PROJECTIONS[2]], [PROJECTIONS[1]]
    if INDEXER[0] in weights:
        inputs += list(INDEXER[:2])
        queries += [INDEXER[2]]
    return (
        SmoothMapping("ov", PROJECTIONS[3], (PROJECTIONS[4],), heads, v, k),
        SmoothMapping("norm-linear", NORMS[0], tuple(inputs)),
        SmoothMapping("norm-linear", NORMS[1], tuple(queries)),
    )


def synthetic_inputs(config, weights, seed, tokens):
    generator = torch.Generator().manual_seed(seed)
    h, q, c, heads, _, _, v = (config[key] for key in DIMENSIONS)
    result = {
        key: torch.randn(tokens, width, generator=generator)
        for key, width in (
            ("input_norm.input", h),
            ("q_norm.input", q),
            ("kv_norm.input", c),
            ("kv_b.input", c),
            ("o.input", heads * v),
        )
    }
    for key, name in zip(("input_norm", "q_norm", "kv_norm"), NORMS, strict=True):
        result[key + ".output"] = norm(
            result[key + ".input"], weights[name], config.get("rms_norm_eps", 1e-6)
        )
    return result


def load_cache(path, plan, fingerprints, config, tokens):
    expected_widths = {
        "input_norm": config["hidden_size"],
        "q_norm": config["q_lora_rank"],
        "kv_norm": config["kv_lora_rank"],
        "kv_b": config["kv_lora_rank"],
        "o": config["num_attention_heads"] * config["v_head_dim"],
    }
    values = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = json.loads((f.metadata() or {}).get("glm52_l4", "{}"))
        if (
            metadata.get("schema_version") != 1
            or metadata.get("stage") != "baseline"
            or metadata.get("input_kind") not in ("real", "synthetic")
            or metadata.get("layer") != plan["layer"]
            or metadata.get("config_sha256") != plan["config_sha256"]
            or metadata.get("weight_sha256") != fingerprints
        ):
            raise ValueError(
                "Cache provenance/stage/selected weights do not match baseline"
            )
        for key, width in expected_widths.items():
            for suffix in ("input", "output") if key.endswith("norm") else ("input",):
                name = key + "." + suffix
                view = f.get_slice(name)
                shape = view.get_shape()
                if len(shape) != 2 or shape[0] < 1 or shape[1] != width:
                    raise ValueError(f"Cache shape mismatch: {name}: {shape}")
                value = view[:tokens].float()
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite cache: {name}")
                values[name] = value
    return values, metadata


@torch.no_grad()
def check_flex(weights, config, inputs, oracle):
    results = {}
    eps = config.get("rms_norm_eps", 1e-6)
    for mapping in groups(weights, config):
        print(f"FlexSmooth: {mapping.source}", flush=True)
        names = mapping.targets
        original = {name: weights[name] for name in names}
        actual, reference = shell(original), shell(original)
        cache_key = (
            "o.input"
            if mapping.kind == "ov"
            else (
                "input_norm.output" if mapping.source == NORMS[0] else "q_norm.output"
            )
        )
        x = inputs[cache_key]
        consumers = [reference.get_submodule(name) for name in mapping.consumers]
        w = torch.cat([module.weight for module in consumers])
        search = search_alpha_beta(x, w)
        searcher = oracle.search.FlexSmoothAlphaBetaSearcher()
        alpha, beta, loss = searcher.search_alpha_beta(x, w)
        calculator = oracle.scales.FlexSmoothScaleCalculator(alpha, beta, "max")
        maxima = oracle.scales.compute_multi_weight_scale(
            [m.weight for m in consumers], w.dtype
        )
        if mapping.kind == "norm-linear":
            scale = smooth_scale(
                x.abs().amax(0), w.abs().amax(0), search.alpha, search.beta
            )
            ref_scale = calculator.compute_smooth_scale(x.abs().amax(0), maxima)
            subgraph = oracle.subgraphs.NormLinearSubgraph(
                reference.get_submodule(mapping.source), consumers
            )
            oracle.fusion.NormLinearSubgraphFusion().apply_fusion(
                subgraph, {"scales": ref_scale}
            )
        else:
            scale, _ = ov_scales(
                x.abs().amax(0),
                w.abs().amax(0),
                search.alpha,
                search.beta,
                mapping.heads,
                mapping.heads,
            )
            ref_scale, v_scale = calculator.compute_ov_scales(
                x.abs().amax(0), maxima, mapping.heads, mapping.heads
            )
            source = reference.get_submodule(mapping.source)
            view = source.weight.view(
                mapping.heads, mapping.key_dim + mapping.value_dim, -1
            )
            virtual = shell(
                {"v": view[:, mapping.key_dim :].reshape(-1, source.weight.shape[1])}
            ).v
            subgraph = oracle.subgraphs.OVSubgraph(
                consumers[0], virtual, mapping.heads, mapping.heads
            )
            oracle.fusion.OVSubgraphFusion().apply_fusion(
                subgraph, {"o_scales": ref_scale, "v_scales": v_scale}
            )
            view[:, mapping.key_dim :].copy_(
                virtual.weight.reshape(mapping.heads, mapping.value_dim, -1)
            )
        apply_scales(actual, mapping, scale)
        ref_losses = [
            searcher.evaluate_alpha_beta(x, w, round(i / 20, 2), 1 - round(i / 20, 2))
            for i in range(21)
        ]
        ref_losses += [
            searcher.evaluate_alpha_beta(x, w, alpha, round(i / 20, 2))
            for i in range(21)
        ]
        checks = {
            "alpha_beta": {
                "passed": (search.alpha, search.beta) == (alpha, beta),
                "actual": [search.alpha, search.beta],
                "reference": [alpha, beta],
            },
            "candidate_losses": compare(
                torch.tensor(search.alpha_losses + search.beta_losses),
                torch.tensor(ref_losses),
            ),
            "scale": compare(scale, ref_scale),
        }
        for name in names:
            checks["weight:" + name] = compare(
                actual.get_submodule(name).weight, reference.get_submodule(name).weight
            )
        for name in mapping.consumers:
            before = F.linear(x, weights[name])
            after = actual.get_submodule(name)(x / scale)
            checks["cached_output:" + name] = compare(after, before, 1e-4)
            checks["reference_output:" + name] = compare(
                after, reference.get_submodule(name)(x / ref_scale), 1e-4
            )
            if mapping.kind == "norm-linear":
                raw = inputs[cache_key.replace("output", "input")]
                before = F.linear(
                    norm(raw, weights[mapping.source], eps), weights[name]
                )
                after = actual.get_submodule(name)(
                    norm(raw, actual.get_submodule(mapping.source).weight, eps)
                )
                checks["norm_linear_output:" + name] = compare(after, before, 1e-4)
        if mapping.kind == "ov":
            shape = (mapping.heads, mapping.key_dim + mapping.value_dim, -1)
            a, b = (
                actual.get_submodule(mapping.source).weight.view(shape),
                weights[mapping.source].view(shape),
            )
            checks["K_rows_unchanged"] = compare(
                a[:, : mapping.key_dim], b[:, : mapping.key_dim], exact=True
            )
            z = inputs["kv_b.input"]
            before = F.linear(z, weights[mapping.source]).view(-1, *shape[:2])[
                :, :, mapping.key_dim :
            ]
            after = F.linear(z, actual.get_submodule(mapping.source).weight).view(
                -1, *shape[:2]
            )[:, :, mapping.key_dim :]
            checks["V_output"] = compare(
                after, before / scale.view(mapping.heads, mapping.value_dim), 1e-4
            )
        results[mapping.source] = {
            "checks": checks,
            "transformed_weight_statistics": {
                name: stats(actual.get_submodule(name).weight) for name in names
            },
            "scale_stats": stats(scale),
            "activation_stats": stats(x),
            "alpha_losses": list(search.alpha_losses),
            "beta_losses": list(search.beta_losses),
            "reference_losses": ref_losses,
            "best_loss": loss,
        }
    return results


def reference_rotate(oracle, weight, rotation, axis, stride=None, offset=0):
    """Execute original rotate_linear per segment, avoiding a huge dense head matrix."""
    result = weight.clone()
    size = rotation.shape[0]
    stride = size if stride is None else stride
    for start in range(0, weight.shape[axis], stride):
        indices = [slice(None), slice(None)]
        indices[axis] = slice(start + offset, start + offset + size)
        indices = tuple(indices)
        module = shell({"linear": weight[indices]}).linear
        oracle.utils.rotate_linear(module, rotation, right_rotate=axis == 1)
        result[indices] = module.weight
    return result


@torch.no_grad()
def check_quarot(weights, config, inputs, oracle, seed, block):
    h, q, c, _, k, r, v = (config[key] for key in DIMENSIONS)
    matrices, rotations, checks = {}, {}, {}
    for name, size, shifted in (
        ("H", h, False),
        ("A", q, True),
        ("C", c, False),
        ("V", v, False),
    ):
        print(f"QuaRot matrix: {name} ({size})", flush=True)
        rotation = make_hadamard_rotation(
            size, block_size=block, shifted=shifted, seed=seed
        )
        # Dense materialization is reference-only; weights use the structured path.
        matrix = rotation.to_dense()
        rotations[name] = rotation
        mode = (
            oracle.utils.QuaRotMode.BLOCK_HADAMARD_SHIFTED
            if shifted
            else oracle.utils.QuaRotMode.HADAMARD
        )
        with preserve_rng():
            ref = oracle.utils.create_rot(mode, size, block, seed=seed)
        checks["matrix:" + name] = compare(matrix, ref, exact=True)
        # A bounded row sample supplements exact matrix parity, not a full Q.T Q test.
        support = matrix.ne(0)
        selected = torch.arange(min(size, 2 * block))
        gram = matrix[selected] @ matrix.T
        expected = torch.zeros_like(gram)
        expected[torch.arange(len(selected)), selected] = 1
        checks["orthogonal_sample:" + name] = {
            "passed": bool((gram - expected).abs().max() <= 1e-5),
            "max_abs": (gram - expected).abs().max().item(),
            "rows_checked": selected.tolist(),
            "max_nonzeros_per_row": int(support.sum(1).max()),
        }
        matrices[name] = matrix
    actual, reference = shell(weights), shell(weights)
    smooth = groups(weights, config)
    fusions = [
        (item.source, item.consumers) for item in smooth if item.kind == "norm-linear"
    ]
    fusions += [(NORMS[2], (PROJECTIONS[3],))]
    for source, consumers in fusions:
        fuse_norm_linears(
            actual.get_submodule(source),
            [actual.get_submodule(name) for name in consumers],
            precision=torch.float32,
        )
        oracle.utils.fuse_ln_linear(
            [reference.get_submodule(source)],
            [reference.get_submodule(name) for name in consumers],
        )
        for name in (source, *consumers):
            checks["fusion:" + name] = compare(
                actual.get_submodule(name).weight, reference.get_submodule(name).weight
            )
    operations = [(name, "H", 1, h, 0) for name in smooth[1].consumers]
    operations += [(PROJECTIONS[4], "H", 0, h, 0), (PROJECTIONS[0], "A", 0, q, 0)]
    operations += [(name, "A", 1, q, 0) for name in smooth[2].consumers]
    operations += [
        (PROJECTIONS[3], "V", 0, k + v, k),
        (PROJECTIONS[4], "V", 1, v, 0),
        (PROJECTIONS[2], "C", 0, c + r, 0),
        (PROJECTIONS[3], "C", 1, c, 0),
    ]
    for name, space, axis, stride, offset in operations:
        print(f"QuaRot weight: {name}, {space}", flush=True)
        module, ref = actual.get_submodule(name), reference.get_submodule(name)
        module.weight = nn.Parameter(
            rotate_axis(
                module.weight,
                rotations[space],
                axis=axis,
                stride=stride,
                offset=offset,
                precision=torch.float32,
            ),
            requires_grad=False,
        )
        ref.weight = nn.Parameter(
            reference_rotate(oracle, ref.weight, matrices[space], axis, stride, offset),
            requires_grad=False,
        )
        checks[f"rotation:{space}:{name}"] = compare(module.weight, ref.weight)
        checks[f"rotation:{space}:{name}"]["transformed_statistics"] = stats(
            module.weight
        )
    eps = config.get("rms_norm_eps", 1e-6)
    raw = inputs["input_norm.input"]
    x0 = norm(raw, weights[NORMS[0]], eps)
    x1 = norm(raw @ matrices["H"], actual.get_submodule(NORMS[0]).weight, eps)
    qa0 = norm(F.linear(x0, weights[PROJECTIONS[0]]), weights[NORMS[1]], eps)
    qa1 = norm(
        actual.get_submodule(PROJECTIONS[0])(x1),
        actual.get_submodule(NORMS[1]).weight,
        eps,
    )
    for name in smooth[2].consumers:
        checks["q_chain:" + name] = compare(
            actual.get_submodule(name)(qa1), F.linear(qa0, weights[name]), 1e-4
        )
    for name in smooth[1].consumers[2:]:
        checks["indexer_input:" + name] = compare(
            actual.get_submodule(name)(x1), F.linear(x0, weights[name]), 1e-4
        )
    kv0, kv1 = (
        F.linear(x0, weights[PROJECTIONS[2]]),
        actual.get_submodule(PROJECTIONS[2])(x1),
    )
    checks["rope_projection"] = compare(kv1[:, c:], kv0[:, c:], 1e-4)
    kv0 = F.linear(norm(kv0[:, :c], weights[NORMS[2]], eps), weights[PROJECTIONS[3]])
    kv1 = actual.get_submodule(PROJECTIONS[3])(
        norm(kv1[:, :c], actual.get_submodule(NORMS[2]).weight, eps)
    )
    kv0, kv1 = (
        value.view(-1, config["num_attention_heads"], k + v) for value in (kv0, kv1)
    )
    checks["K_chain"] = compare(kv1[:, :, :k], kv0[:, :, :k], 1e-4)
    checks["V_chain"] = compare(kv1[:, :, k:], kv0[:, :, k:] @ matrices["V"], 1e-4)
    output = inputs["o.input"]
    rotated = (
        output.view(-1, config["num_attention_heads"], v) @ matrices["V"]
    ).flatten(1)
    checks["O_output_rotated_basis"] = compare(
        actual.get_submodule(PROJECTIONS[4])(rotated),
        F.linear(output, weights[PROJECTIONS[4]]) @ matrices["H"],
        1e-4,
    )
    return {
        "checks": checks,
        "reference_execution": (
            "original fusion/rotation functions; per-segment rotate_linear, "
            "not full ModelSlim processor"
        ),
        "mapping_scope": (
            "explicit selected attention projection pairs; "
            "whole-model mapping remains outside this probe"
        ),
    }


def all_passed(value):
    if isinstance(value, dict):
        return value.get("passed", True) and all(
            all_passed(item) for item in value.values()
        )
    if isinstance(value, list):
        return all(all_passed(item) for item in value)
    return True


def safe_output(path, roots):
    path = path.resolve()
    if path.exists() or any(path.is_relative_to(root.resolve()) for root in roots):
        raise ValueError(
            "Output must be new, outside checkpoint/reference, and not the cache file"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--modelslim-source", type=Path)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--activation-cache", type=Path)
    source.add_argument("--synthetic-activations", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--tokens", type=int, default=128, choices=range(1, 129), metavar="1..128"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-weight-gib", type=float, default=2.0)
    parser.add_argument("--max-estimated-memory-gib", type=float, default=16.0)
    args = parser.parse_args(argv)
    roots = [args.model_dir]
    if args.modelslim_source:
        roots.append(args.modelslim_source)
    if args.activation_cache:
        roots.append(args.activation_cache)
    try:
        output = safe_output(args.output, roots)
    except ValueError as error:
        parser.error(str(error))
    report = {
        "schema_version": 1,
        "scope": "selected_attention_subgraphs_fp32",
        "scoped_l4_passed": False,
        "full_layer_l4_passed": False,
        "full_model_accuracy_verified": False,
        "errors": [],
        "excludes": [
            "full attention softmax/RoPE/indexer top-k",
            "MLP/experts/router/lm_head",
            "QuaRot then FlexSmooth composition on real activations",
            "BF16/NPU kernel numerics",
            "MXFP parity",
            "full Modifier lifecycle on the real model",
        ],
    }
    started = time.perf_counter()
    code = 2
    try:
        check_implementation_location()
        if (
            args.threads < 1
            or args.block_size <= 0
            or args.block_size & (args.block_size - 1)
        ):
            raise ValueError("Positive threads and power-of-two block-size required")
        torch.set_num_threads(args.threads)
        config, plan = checkpoint_plan(args.model_dir, args.layer, args.max_weight_gib)
        report["checkpoint"] = plan
        report["model_dimensions"] = {key: config[key] for key in DIMENSIONS} | {
            "rms_norm_eps": config.get("rms_norm_eps", 1e-6)
        }
        # Conservative planning estimate, not an allocator-enforced RAM limit.
        rotation_bytes = 4 * sum(
            config[key] ** 2
            for key in ("hidden_size", "q_lora_rank", "kv_lora_rank", "v_head_dim")
        )
        estimate = plan["selected_fp32_bytes"] * 16 + rotation_bytes * 6
        report["estimated_working_memory_gib"] = estimate / 2**30
        if (
            not math.isfinite(args.max_estimated_memory_gib)
            or args.max_estimated_memory_gib <= 0
            or estimate > args.max_estimated_memory_gib * 2**30
        ):
            raise ValueError("Estimated RAM exceeds --max-estimated-memory-gib")
        report["settings"] = {
            "tokens": args.tokens,
            "seed": args.seed,
            "block_size": args.block_size,
            "threads": args.threads,
            "arithmetic": "float32 CPU",
        }
        for key in ("hidden_size", "q_lora_rank", "kv_lora_rank", "v_head_dim"):
            if config[key] % args.block_size:
                raise ValueError(f"block-size does not divide {key}")
        if args.dry_run:
            report["status"] = "headers_only_no_payload_read"
            code = 0
        else:
            if not args.modelslim_source or not (
                args.activation_cache or args.synthetic_activations
            ):
                raise ValueError(
                    "Numerical run requires --modelslim-source "
                    "and an explicit activation source"
                )
            quarot, flex = (
                ModelSlimOracle(args.modelslim_source.resolve()),
                FlexOracle(args.modelslim_source.resolve()),
            )
            weights, fingerprints = load_selected(args.model_dir, plan)
            report["weight_sha256"] = fingerprints
            real_inputs = False
            if args.activation_cache:
                inputs, metadata = load_cache(
                    args.activation_cache, plan, fingerprints, config, args.tokens
                )
                report["activation_cache_metadata"] = metadata
                real_inputs = metadata.get("input_kind") == "real"
            else:
                inputs = synthetic_inputs(config, weights, args.seed, args.tokens)
            report["activation_source"] = (
                "captured_baseline" if args.activation_cache else "synthetic"
            )
            report["weight_statistics"] = {
                name: stats(value) for name, value in weights.items()
            }
            report["activation_statistics"] = {
                name: stats(value) for name, value in inputs.items()
            }
            report["activation_sha256"] = {
                name: digest_tensor(value) for name, value in inputs.items()
            }
            report["quarot"] = check_quarot(
                weights, config, inputs, quarot, args.seed, args.block_size
            )
            report["flexsmooth"] = check_flex(weights, config, inputs, flex)
            passed = all_passed(report["quarot"]) and all_passed(report["flexsmooth"])
            report["numerical_checks_passed"] = passed
            report["scoped_l4_passed"] = passed and real_inputs
            report["reference_source_sha256"] = {
                "quarot": quarot.hashes,
                "flexsmooth": flex.hashes,
            }
            report["status"] = (
                "scoped_pass"
                if report["scoped_l4_passed"]
                else "precheck_only"
                if passed
                else "numerical_failure"
            )
            code = (0 if real_inputs else 3) if passed else 1
    except Exception as error:
        report["status"] = "error"
        report["errors"].append(f"{type(error).__name__}: {error}")
    report["elapsed_seconds"] = time.perf_counter() - started
    report["versions"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "compressed-tensors", "llmcompressor")
    }
    report["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["implementation_sha256"] = {
        name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
        for name in (
            "src/llmcompressor/modeling/fuse.py",
            "src/llmcompressor/modifiers/transform/quarot/rotation.py",
            "src/llmcompressor/modifiers/transform/flex_smooth/search.py",
            "src/llmcompressor/modifiers/transform/flex_smooth/mappings.py",
            "tests/quarot/source_loader.py",
            "tests/quarot/modelslim_oracle.py",
            "tests/flex_smooth/oracle.py",
            "tools/glm52_l4_capture.py",
        )
    }

    # Nonfinite search losses are explicit strings, never invalid JSON or a pass.
    def serializable(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {key: serializable(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [serializable(item) for item in value]
        return value

    with output.open("x", encoding="utf-8") as stream:
        json.dump(serializable(report), stream, indent=2, allow_nan=False)
    print(
        f"{report['status']}: {output}; full-layer L4 and L5 remain unverified",
        flush=True,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
