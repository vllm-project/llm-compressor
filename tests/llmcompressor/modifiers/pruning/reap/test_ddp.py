"""
Multi-GPU tests for REAP expert pruning with DDP.

Verifies that running REAP with DDP (data partitioned across ranks) produces
the same pruning decisions as running on a single GPU (full dataset).

Run with:
    pytest tests/llmcompressor/modifiers/pruning/reap/test_ddp.py \
        -m multi_gpu -v
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from functools import partial

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.offload import init_dist
from pydantic import PrivateAttr
from transformers import AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modifiers.pruning.expert_sparsity.utils import get_experts_router
from llmcompressor.modifiers.pruning.reap import REAPPruningModifier
from llmcompressor.utils import load_context
from llmcompressor.utils.helpers import numeric_sorted_names
from tests.testing_utils import requires_gpu, torchrun

QWEN_MODEL = "inference-optimization/Qwen3.8-1.0B-A0.6B"
NUM_SAMPLES = 16
# PerfectBlend only has a train split; compare the next 16 training samples.
SAMPLE_OFFSET = 16
MAX_SEQ_LENGTH = 512
SALIENCY_RTOL = 1e-5
SALIENCY_ATOL = 1e-3


class _DiagnosticREAPPruningModifier(REAPPruningModifier):
    """Capture router decisions and local accumulators for this comparison."""

    _routing_trace: dict = PrivateAttr(default_factory=dict)
    _router_logits_trace: dict = PrivateAttr(default_factory=dict)
    _pre_reduce_saliency: dict = PrivateAttr(default_factory=dict)
    _calibration_samples: list = PrivateAttr(default_factory=list)
    _layer_names: list = PrivateAttr(default_factory=list)

    def experts_block_hook(self, layer_name, module, args, output):
        indices = args[1].detach().to(device="cpu").clone()
        weights = args[2].detach().to(device="cpu", dtype=torch.float32).clone()
        self._routing_trace.setdefault(layer_name, []).append(
            {"indices": indices, "weights": weights}
        )
        return super().experts_block_hook(layer_name, module, args, output)

    def router_forward_hook(self, layer_name, module, args, output):
        # Qwen's top-k router returns logits before softmax and selection as the
        # first element. Save those exact logits so mismatched routes can be
        # explained without re-running the router with a separate linear op.
        if isinstance(output, torch.Tensor):
            logits = output
        elif (
            isinstance(output, (tuple, list))
            and output
            and isinstance(output[0], torch.Tensor)
        ):
            logits = output[0]
        else:
            raise TypeError(
                f"Expected tensor router logits from {layer_name}, got "
                f"{type(output).__name__}"
            )

        self._router_logits_trace.setdefault(layer_name, []).append(
            logits.detach()
            .reshape(-1, logits.shape[-1])
            .to(device="cpu", dtype=torch.float32)
            .clone()
        )

    def on_calibration_start(self, state, event, **kwargs):
        # shuffle_calibration_samples=False uses LengthAwareSampler, which sorts
        # each rank's partition by length. Preserve the original dataset index
        # for every forward so reference and DDP routes can be aligned by sample.
        dataloader = state.data.calib
        dataset = getattr(dataloader, "dataset", None)
        sampler = getattr(dataloader, "sampler", None)
        if dataset is not None and "input_ids" in getattr(dataset, "column_names", []):
            sample_order = list(sampler) if sampler is not None else range(len(dataset))
            self._calibration_samples = [
                {
                    "dataset_index": int(dataset_idx),
                    "token_count": len(token_ids := dataset[dataset_idx]["input_ids"]),
                    "token_ids_sha256": hashlib.sha256(
                        json.dumps(token_ids, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "input_ids": token_ids,
                }
                for dataset_idx in sample_order
            ]
        result = super().on_calibration_start(state, event, **kwargs)
        # The base now discovers layers into _moe_stats; the old _moe_attrs
        # list was removed when expert pruning was moved into its shared base.
        self._layer_names = numeric_sorted_names(self._moe_stats)
        for layer_name in self._layer_names:
            moe = state.model.get_submodule(layer_name)
            _, router = get_experts_router(layer_name, moe)
            if router is None:
                raise RuntimeError(f"Could not find router for {layer_name}")
            self.register_hook(
                router,
                partial(self.router_forward_hook, layer_name),
                "forward",
            )
        return result

    def reduce_moe_stats(self, layer_name):
        # Capture local accumulators before the source-rank reduce and count
        # all-reduce. REAP now stores these in _moe_stats rather than trackers.
        stats = self._moe_stats[layer_name]
        self._pre_reduce_saliency[layer_name] = {
            "sum": stats["sum_saliency"].detach().cpu().clone(),
            "count": stats["count"].detach().cpu().clone(),
        }
        return super().reduce_moe_stats(layer_name)


def _modifier_diagnostics(modifier, rank, sample_range):
    sample_start, _ = sample_range
    samples = [
        {
            "sample_index": sample_start + sample["dataset_index"],
            "token_count": sample["token_count"],
            "token_ids_sha256": sample["token_ids_sha256"],
            "input_ids": sample["input_ids"],
        }
        for sample in modifier._calibration_samples
    ]
    routing = {}
    for layer_name, events in modifier._routing_trace.items():
        routing[layer_name] = []
        router_logits_events = modifier._router_logits_trace.get(layer_name, [])
        for call_idx, event in enumerate(events):
            sample = samples[call_idx] if call_idx < len(samples) else {}
            router_logits = (
                router_logits_events[call_idx].tolist()
                if call_idx < len(router_logits_events)
                else None
            )
            routing[layer_name].append(
                {
                    **{
                        key: sample.get(key)
                        for key in (
                            "sample_index",
                            "token_count",
                            "token_ids_sha256",
                        )
                    },
                    "indices": event["indices"].tolist(),
                    "weights": event["weights"].tolist(),
                    "router_logits": router_logits,
                }
            )

    return {
        "rank": rank,
        "sample_range": list(sample_range),
        "calibration_samples": samples,
        "routing": routing,
        "pre_reduce_saliency": {
            name: {
                "sum": stats["sum"].tolist(),
                "count": stats["count"].tolist(),
            }
            for name, stats in modifier._pre_reduce_saliency.items()
        },
    }


def _route_counts(events, num_experts):
    counts = torch.zeros(num_experts, dtype=torch.int64)
    for event in events:
        indices = torch.as_tensor(event["indices"], dtype=torch.int64).reshape(-1)
        counts += torch.bincount(indices, minlength=num_experts)[:num_experts]
    return counts.tolist()


def _router_token_diagnostics(event, token_idx, top_k):
    logits = event.get("router_logits")
    if logits is None:
        return None

    logits = torch.as_tensor(logits, dtype=torch.float32)
    if logits.ndim != 2 or token_idx >= logits.shape[0]:
        return None
    logits = logits[token_idx]
    probabilities = torch.softmax(logits, dim=-1)
    candidate_count = min(top_k + 1, logits.numel())
    candidate_probs, candidate_experts = torch.topk(probabilities, candidate_count)
    candidate_logits = logits[candidate_experts]
    candidates = [
        {
            "expert": int(expert_idx),
            "logit": float(logits[expert_idx]),
            "probability": float(probability),
        }
        for expert_idx, probability in zip(
            candidate_experts.tolist(), candidate_probs.tolist()
        )
    ]
    boundary_gap = (
        float(candidate_probs[top_k - 1] - candidate_probs[top_k])
        if candidate_count > top_k
        else None
    )
    logit_boundary_gap = (
        float(candidate_logits[top_k - 1] - candidate_logits[top_k])
        if candidate_count > top_k
        else None
    )
    return {
        "logits_by_expert": logits.tolist(),
        "probabilities_by_expert": probabilities.tolist(),
        "top_candidates": candidates,
        "top_k_boundary_probability_gap": boundary_gap,
        "top_k_boundary_logit_gap": logit_boundary_gap,
    }


def _route_diagnostics(reference, ddp_ranks, layer_names, num_experts):
    layers = {}
    all_comparable = all_match = True
    reference_tokens = {
        sample["sample_index"]: sample["input_ids"]
        for sample in reference["calibration_samples"]
    }

    for name in layer_names:
        ref_events = reference["routing"].get(name, [])
        ddp_by_rank = {
            str(rank["rank"]): rank["routing"].get(name, [])
            for rank in sorted(ddp_ranks, key=lambda item: item["rank"])
        }
        ddp_events = [event for events in ddp_by_rank.values() for event in events]
        ref_by_sample = {
            event["sample_index"]: event
            for event in ref_events
            if event["sample_index"] is not None
        }
        ddp_by_sample = {}
        for event in ddp_events:
            if event["sample_index"] is not None:
                ddp_by_sample.setdefault(event["sample_index"], []).append(event)

        unique_indices = (
            len(ref_by_sample) == len(ref_events)
            and len(ddp_by_sample) == len(ddp_events)
            and all(len(events) == 1 for events in ddp_by_sample.values())
        )
        ref_ids, ddp_ids = set(ref_by_sample), set(ddp_by_sample)
        missing_from_ddp = sorted(ref_ids - ddp_ids)
        missing_from_reference = sorted(ddp_ids - ref_ids)
        pairs = (
            [
                (idx, ref_by_sample[idx], ddp_by_sample[idx][0])
                for idx in sorted(ref_ids & ddp_ids)
            ]
            if unique_indices
            else []
        )

        slot_mismatches = expert_set_mismatches = 0
        shape_mismatches = identity_mismatches = 0
        router_logits_missing = []
        router_logits_shape_mismatches = []
        identity_known = True
        max_gate_weight_error = 0.0
        route_examples, shape_examples, identity_examples = [], [], []
        for sample_idx, ref_event, ddp_event in pairs:
            ref_hash = ref_event["token_ids_sha256"]
            ddp_hash = ddp_event["token_ids_sha256"]
            if ref_hash is None or ddp_hash is None:
                identity_known = False
            elif ref_hash != ddp_hash:
                identity_mismatches += 1
                if len(identity_examples) < 20:
                    identity_examples.append(
                        {
                            "sample_index": sample_idx,
                            "reference_token_count": ref_event["token_count"],
                            "ddp_token_count": ddp_event["token_count"],
                            "reference_token_ids_sha256": ref_hash,
                            "ddp_token_ids_sha256": ddp_hash,
                        }
                    )
                continue

            ref_indices = torch.as_tensor(ref_event["indices"], dtype=torch.int64)
            ddp_indices = torch.as_tensor(ddp_event["indices"], dtype=torch.int64)
            if ref_indices.shape != ddp_indices.shape:
                shape_mismatches += 1
                if len(shape_examples) < 20:
                    shape_examples.append(
                        {
                            "sample_index": sample_idx,
                            "reference_shape": list(ref_indices.shape),
                            "ddp_shape": list(ddp_indices.shape),
                        }
                    )
                continue

            for side, event, indices in (
                ("reference", ref_event, ref_indices),
                ("ddp", ddp_event, ddp_indices),
            ):
                router_logits = event.get("router_logits")
                if router_logits is None:
                    router_logits_missing.append(
                        {"sample_index": sample_idx, "side": side}
                    )
                    continue
                router_logits = torch.as_tensor(router_logits)
                if (
                    router_logits.ndim != 2
                    or router_logits.shape[0] != indices.shape[0]
                    or router_logits.shape[1] != num_experts
                ):
                    router_logits_shape_mismatches.append(
                        {
                            "sample_index": sample_idx,
                            "side": side,
                            "router_logits_shape": list(router_logits.shape),
                            "route_shape": list(indices.shape),
                        }
                    )

            slot_diff = ref_indices != ddp_indices
            set_diff = (
                torch.sort(ref_indices, dim=-1).values
                != torch.sort(ddp_indices, dim=-1).values
            ).any(dim=-1)
            slot_mismatches += int(slot_diff.sum().item())
            expert_set_mismatches += int(set_diff.sum().item())
            for token_idx in torch.where(slot_diff.any(dim=-1))[0].tolist():
                if len(route_examples) == 20:
                    break
                example = {
                    "sample_index": sample_idx,
                    "token_index": token_idx,
                    "token_id": reference_tokens[sample_idx][token_idx],
                    "reference_experts": ref_indices[token_idx].tolist(),
                    "ddp_experts": ddp_indices[token_idx].tolist(),
                    "expert_set_differs": bool(set_diff[token_idx]),
                    "reference_router": _router_token_diagnostics(
                        ref_event, token_idx, ref_indices.shape[-1]
                    ),
                    "ddp_router": _router_token_diagnostics(
                        ddp_event, token_idx, ddp_indices.shape[-1]
                    ),
                }
                ref_router = example["reference_router"]
                ddp_router = example["ddp_router"]
                if ref_router is not None and ddp_router is not None:
                    example["ddp_minus_reference_router_logit_by_expert"] = [
                        ddp - ref
                        for ref, ddp in zip(
                            ref_router["logits_by_expert"],
                            ddp_router["logits_by_expert"],
                        )
                    ]
                    example["ddp_minus_reference_router_probability_by_expert"] = [
                        ddp - ref
                        for ref, ddp in zip(
                            ref_router["probabilities_by_expert"],
                            ddp_router["probabilities_by_expert"],
                        )
                    ]
                route_examples.append(example)

            ref_weights = torch.as_tensor(ref_event["weights"], dtype=torch.float64)
            ddp_weights = torch.as_tensor(ddp_event["weights"], dtype=torch.float64)
            if ref_weights.shape == ddp_weights.shape and ref_weights.numel():
                max_gate_weight_error = max(
                    max_gate_weight_error,
                    float((ref_weights - ddp_weights).abs().max().item()),
                )

        comparable = (
            unique_indices
            and len(ref_events) == len(ddp_events)
            and not missing_from_ddp
            and not missing_from_reference
            and identity_known
            and identity_mismatches == 0
            and shape_mismatches == 0
            and not router_logits_missing
            and not router_logits_shape_mismatches
        )
        # Top-k slot ordering is not semantically relevant when the same
        # experts are selected. Compare each token's expert set for parity.
        routes_match = expert_set_mismatches == 0 if comparable else None
        all_comparable &= comparable
        if routes_match is False:
            all_match = False

        ref_counts = _route_counts(ref_events, num_experts)
        ddp_counts_by_rank = {
            rank: _route_counts(events, num_experts)
            for rank, events in ddp_by_rank.items()
        }
        ddp_counts = [
            sum(counts[idx] for counts in ddp_counts_by_rank.values())
            for idx in range(num_experts)
        ]
        layers[name] = {
            "comparable": comparable,
            "routes_match": routes_match,
            "reference_forward_calls": len(ref_events),
            "ddp_forward_calls": len(ddp_events),
            "unique_sample_indices": unique_indices,
            "samples_missing_from_ddp": missing_from_ddp,
            "samples_missing_from_reference": missing_from_reference,
            "sample_identity_mismatches": identity_mismatches,
            "sample_identity_mismatch_examples": identity_examples,
            "shape_mismatches": shape_mismatches,
            "shape_mismatch_examples": shape_examples,
            "router_logits_missing": router_logits_missing,
            "router_logits_shape_mismatches": router_logits_shape_mismatches,
            "ordered_expert_slot_mismatches": slot_mismatches,
            "different_expert_set_tokens": expert_set_mismatches,
            "expert_sets_match": routes_match,
            "route_mismatch_examples": route_examples,
            "reference_route_counts_by_expert": ref_counts,
            "ddp_route_counts_by_rank": ddp_counts_by_rank,
            "ddp_route_counts_total_by_expert": ddp_counts,
            "route_counts_match": ref_counts == ddp_counts,
            "max_gate_weight_absolute_error": max_gate_weight_error,
        }

    return {
        "all_layers_comparable": all_comparable,
        "all_layers_match": all_match if all_comparable else None,
        "layers": layers,
    }


def _local_expert_stats(payload, layer_name, expert_idx):
    stats = payload["pre_reduce_saliency"].get(layer_name)
    if stats is None:
        return {
            "pre_reduce_sum_saliency": 0.0,
            "count": 0.0,
            "mean_saliency": 0.0,
        }
    contribution = float(stats["sum"][expert_idx])
    count = float(stats["count"][expert_idx])
    return {
        "pre_reduce_sum_saliency": contribution,
        "count": count,
        "mean_saliency": contribution / count if count else 0.0,
    }


def _saliency_diagnostics(
    reference_report, ddp_report, reference_debug, ddp_ranks, names
):
    arrays = [
        reference_report["saliency"],
        ddp_report["saliency"],
        reference_report["count"],
        ddp_report["count"],
    ]
    layers = {}
    all_close = all_counts_match = True
    for layer_idx in range(max(map(len, arrays))):
        name = names[layer_idx] if layer_idx < len(names) else str(layer_idx)
        if any(layer_idx >= len(array) for array in arrays):
            layers[name] = {"missing_report_data": True}
            all_close = all_counts_match = False
            continue

        ref, ddp, ref_count, ddp_count = [
            torch.as_tensor(array[layer_idx], dtype=torch.float64) for array in arrays
        ]
        if not (ref.shape == ddp.shape == ref_count.shape == ddp_count.shape):
            layers[name] = {
                "reference_shapes": [list(ref.shape), list(ref_count.shape)],
                "ddp_shapes": [list(ddp.shape), list(ddp_count.shape)],
            }
            all_close = all_counts_match = False
            continue

        close = torch.isclose(ref, ddp, rtol=SALIENCY_RTOL, atol=SALIENCY_ATOL)
        counts_match = ref_count == ddp_count
        all_close &= bool(close.all())
        all_counts_match &= bool(counts_match.all())
        experts = []
        for idx, (ref_value, ddp_value) in enumerate(zip(ref.tolist(), ddp.tolist())):
            error = abs(ref_value - ddp_value)
            expert = {
                "expert": idx,
                "reference_saliency": ref_value,
                "ddp_saliency": ddp_value,
                "absolute_error": error,
                "relative_error": error / abs(ref_value) if ref_value else None,
                "within_tolerance": bool(close[idx]),
                "reference_count": float(ref_count[idx]),
                "ddp_count": float(ddp_count[idx]),
            }
            if error or not bool(counts_match[idx]):
                expert["pre_reduce_contributions"] = {
                    "single_gpu_reference": _local_expert_stats(
                        reference_debug, name, idx
                    ),
                    "ddp_ranks": [
                        {
                            "rank": rank["rank"],
                            "sample_range": rank["sample_range"],
                            **_local_expert_stats(rank, name, idx),
                        }
                        for rank in sorted(ddp_ranks, key=lambda item: item["rank"])
                    ],
                }
            experts.append(expert)

        layers[name] = {
            "within_tolerance": bool(close.all()),
            "max_absolute_error": float((ref - ddp).abs().max()),
            "count_mismatches": int((~counts_match).sum()),
            "experts": experts,
        }

    return {
        "all_layers_within_tolerance": all_close,
        "all_expert_counts_match": all_counts_match,
        "layers": layers,
    }


def _sample_summary(samples):
    return [
        {
            key: sample[key]
            for key in ("sample_index", "token_count", "token_ids_sha256")
        }
        for sample in samples
    ]


def _build_diagnostics(reference_report, ddp_report, reference_debug, ddp_ranks, names):
    num_experts = (
        len(reference_report["saliency"][0]) if reference_report["saliency"] else 0
    )
    routing = _route_diagnostics(reference_debug, ddp_ranks, names, num_experts)
    saliency = _saliency_diagnostics(
        reference_report, ddp_report, reference_debug, ddp_ranks, names
    )
    return {
        "dataset": "perfectblend",
        "sample_range": [SAMPLE_OFFSET, SAMPLE_OFFSET + NUM_SAMPLES],
        "calibration_samples": {
            "reference": _sample_summary(reference_debug["calibration_samples"]),
            "ddp_ranks": [
                {
                    "rank": rank["rank"],
                    "samples": _sample_summary(rank["calibration_samples"]),
                }
                for rank in sorted(ddp_ranks, key=lambda item: item["rank"])
            ],
        },
        "routing": routing,
        "saliency": saliency,
        "passed": (
            routing["all_layers_comparable"]
            and routing["all_layers_match"]
            and saliency["all_layers_within_tolerance"]
            and saliency["all_expert_counts_match"]
        ),
    }


def _load_report(path):
    with open(path) as f:
        return json.load(f)


@pytest.mark.integration
@pytest.mark.multi_gpu
@requires_gpu(2)
@torchrun(world_size=2)
def test_reap_ddp_qwen3():
    """REAP with DDP on Qwen3.8-1.0B-A0.6B retains the same experts as single-GPU."""
    torch.use_deterministic_algorithms(True)
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_report = f"{tmpdir}/ref_report.json"

        torch.manual_seed(42)
        torch.get_device_module().manual_seed_all(42)

        # Single-GPU reference (before init_dist)
        with load_context():
            model_ref = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL, dtype=torch.bfloat16, device_map="auto_offload"
            )

        ref_modifier = _DiagnosticREAPPruningModifier(
            sparsity=0.25, report_path=ref_report
        )
        oneshot(
            model=model_ref,
            dataset="perfectblend",
            splits=f"train[{SAMPLE_OFFSET}:{SAMPLE_OFFSET + NUM_SAMPLES}]",
            recipe=ref_modifier,
            num_calibration_samples=NUM_SAMPLES,
            max_seq_length=MAX_SEQ_LENGTH,
            shuffle_calibration_samples=False,
            pipeline="sequential",
        )

        ref_report_data = _load_report(ref_report)
        reference_debug = _modifier_diagnostics(
            ref_modifier,
            rank=None,
            sample_range=(SAMPLE_OFFSET, SAMPLE_OFFSET + NUM_SAMPLES),
        )
        del model_ref
        torch.accelerator.empty_cache()

        # DDP run
        init_dist()
        rank = dist.get_rank()

        ddp_report = f"{tmpdir}/ddp_report.json"

        ddp_split = get_rank_partition("train", NUM_SAMPLES)
        split_start, split_end = map(
            int, ddp_split.split("[", 1)[1].removesuffix("]").split(":")
        )
        ddp_sample_range = (
            SAMPLE_OFFSET + split_start,
            SAMPLE_OFFSET + split_end,
        )
        ddp_split = f"train[{SAMPLE_OFFSET + split_start}:{SAMPLE_OFFSET + split_end}]"

        torch.manual_seed(42)
        torch.get_device_module().manual_seed_all(42)

        with load_context():
            model_ddp = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL, dtype=torch.bfloat16, device_map="auto_offload"
            )

        ddp_modifier = _DiagnosticREAPPruningModifier(
            sparsity=0.25, report_path=ddp_report
        )
        oneshot(
            model=model_ddp,
            dataset="perfectblend",
            splits=ddp_split,
            recipe=ddp_modifier,
            num_calibration_samples=NUM_SAMPLES,
            max_seq_length=MAX_SEQ_LENGTH,
            shuffle_calibration_samples=False,
            pipeline="sequential",
        )

        local_debug = _modifier_diagnostics(
            ddp_modifier, rank=rank, sample_range=ddp_sample_range
        )
        ddp_debug_by_rank = [None] * dist.get_world_size()
        dist.all_gather_object(ddp_debug_by_rank, local_debug)

        diagnostic_message = [None]
        if rank == 0:
            ddp_report_data = _load_report(ddp_report)
            diagnostics = _build_diagnostics(
                ref_report_data,
                ddp_report_data,
                reference_debug,
                ddp_debug_by_rank,
                ref_modifier._layer_names,
            )
            diagnostic_json = json.dumps(diagnostics, indent=2, sort_keys=True)
            print(f"REAP_DDP_DIAGNOSTICS={diagnostic_json}", flush=True)
            if not diagnostics["passed"]:
                diagnostic_message[0] = (
                    "REAP DDP routing/saliency diagnostics failed:\n"
                    f"{diagnostic_json}"
                )

        dist.broadcast_object_list(diagnostic_message, src=0)
        if diagnostic_message[0] is not None:
            raise AssertionError(diagnostic_message[0])

        del model_ddp
        torch.accelerator.empty_cache()
        dist.barrier()
