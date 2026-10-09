# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small GLM-5.2 calibration/profile on existing server weights; no weight downloads.

Single process defaults to auto placement; torchrun uses CT auto_offload and
cpu:gloo,npu:hccl. This is a smoke/profile, not an accuracy evaluation.
"""

import argparse
import gc
import importlib.metadata
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from functools import wraps
from pathlib import Path

import psutil
import torch
import torch.distributed as dist
from compressed_tensors.offload import align_module_device
from compressed_tensors.offload.cache import CPUCache, DiskCache
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.utils import patch_attr
from datasets import Dataset, ReadInstruction, load_dataset
from loguru import logger
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    CompressedTensorsConfig,
    DataCollatorWithPadding,
    GenerationConfig,
)

from llmcompressor import oneshot
from llmcompressor.core import reset_session
from llmcompressor.datasets import get_rank_partition
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import QuaRotModifier
from llmcompressor.modifiers.transform.flex_smooth import base as flex_base
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.sequential import pipeline as sequential_pipeline
from llmcompressor.pipelines.sequential.helpers import Subgraph
from llmcompressor.recipe import Recipe
from llmcompressor.utils import load_context


class Profile:
    """Example-local timings and I/O counters; does not change algorithm behavior."""

    def __init__(self, device, mode="summary", sample_every=1000):
        if mode not in ("summary", "sampled", "full") or sample_every < 1:
            raise ValueError("Use summary/sampled/full and a positive sample interval")
        self.device = device
        self.mode, self.sample_every = mode, sample_every
        self.npu_devices = (
            (
                [torch.npu.current_device()]
                if dist.is_initialized()
                else list(range(torch.npu.device_count()))
            )
            if device == "npu"
            else []
        )
        self.process = psutil.Process()
        self.stages = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
        self.cache_io = defaultdict(lambda: {"calls": 0, "logical_bytes": 0})
        self.report = {
            "schema_version": 2,
            "stages": self.stages,
            "cache_io": self.cache_io,
            "profiling": {
                "mode": mode,
                "sample_every": sample_every,
                "hot_seconds": "synchronized_wall"
                if mode == "full"
                else "host_dispatch",
                "sample_policy": (
                    "first and every Nth call per hot label; no extrapolation"
                ),
                "nested_stages": "inclusive; do not sum parent and child seconds",
                "memory_policy": (
                    "boundary/sample observations; allocator peaks are process-wide"
                ),
            },
        }

    def synchronize(self):
        for device in self.npu_devices:
            torch.npu.synchronize(device)

    def memory(self):
        result = {"cpu_rss_bytes": self.process.memory_info().rss}
        if self.npu_devices:
            result["npu_memory"] = {
                device: dict(
                    allocated_bytes=torch.npu.memory_allocated(device),
                    reserved_bytes=torch.npu.memory_reserved(device),
                    peak_allocated_bytes=torch.npu.max_memory_allocated(device),
                    peak_reserved_bytes=torch.npu.max_memory_reserved(device),
                )
                for device in self.npu_devices
            }
        io = self.process.io_counters()
        result.update(
            process_read_bytes=io.read_bytes, process_write_bytes=io.write_bytes
        )
        return result

    @contextmanager
    def stage(self, name, *, hot=False):
        record = self.stages[name]
        sampled = (
            not hot
            or self.mode == "full"
            or (self.mode == "sampled" and record["calls"] % self.sample_every == 0)
        )
        kind = (
            "synchronized_wall" if not hot or self.mode == "full" else "host_dispatch"
        )
        record["timing_kind"] = kind
        if sampled:
            sync_start = time.perf_counter()
            self.synchronize()
            sync_before = time.perf_counter() - sync_start
            before = self.memory()
        started = time.perf_counter()
        try:
            yield
        finally:
            dispatched = time.perf_counter() - started
            record["calls"] += 1
            record["host_dispatch_seconds"] = (
                record.get("host_dispatch_seconds", 0) + dispatched
            )
            sync_after = 0.0
            if sampled:
                sync_start = time.perf_counter()
                self.synchronize()
                sync_after = time.perf_counter() - sync_start
                record["synchronized_samples"] = (
                    record.get("synchronized_samples", 0) + 1
                )
                record["sampled_synchronized_seconds"] = (
                    record.get("sampled_synchronized_seconds", 0)
                    + dispatched
                    + sync_after
                )
                record["sync_before_seconds"] = (
                    record.get("sync_before_seconds", 0) + sync_before
                )
                record["sync_after_seconds"] = (
                    record.get("sync_after_seconds", 0) + sync_after
                )
                after = self.memory()
                record["last_memory"] = after
                record["max_observed_cpu_rss_bytes"] = max(
                    record.get("max_observed_cpu_rss_bytes", 0),
                    before["cpu_rss_bytes"],
                    after["cpu_rss_bytes"],
                )
                for key in ("process_read_bytes", "process_write_bytes"):
                    record[key] = record.get(key, 0) + after[key] - before[key]
            record["seconds"] += dispatched + (
                sync_after if kind == "synchronized_wall" else 0
            )

    def progress(self, original):
        """Suppress batch bars at their source; keep source-rank 5% summaries."""

        def batches(iterable, index, total, phase):
            source = not dist.is_initialized() or dist.get_rank() == 0
            if source and index == 1 and phase == "Calibrating":
                logger.info("[GLM calibration][rank 0] starting {} subgraphs", total)
            yield from iterable
            if (
                source
                and phase == "Propagating"
                and (index == total or index % max(1, (total + 19) // 20) == 0)
            ):
                logger.info(
                    "[GLM calibration][rank 0] completed {}/{} subgraphs", index, total
                )

        def progress(iterable, *args, **kwargs):
            match = re.fullmatch(
                r"\((\d+)/(\d+)\): (Calibrating|Propagating)", kwargs.get("desc", "")
            )
            if self.mode == "full" or match is None:
                return original(iterable, *args, **kwargs)
            return batches(iterable, int(match[1]), int(match[2]), match[3])

        return progress

    @contextmanager
    def instrument(self):
        def timed(original, name):
            @wraps(original)
            def call(*args, **kwargs):
                label = name() if callable(name) else name
                with self.stage(label, hot=label != "quarot"):
                    return original(*args, **kwargs)

            return call

        def counted(original, label):
            @wraps(original)
            def call(cache, value, *args, **kwargs):
                if value is not None:
                    entry = self.cache_io[label]
                    entry["calls"] += 1
                    entry["logical_bytes"] += value.numel() * value.element_size()
                return original(cache, value, *args, **kwargs)

            return call

        with ExitStack() as stack:
            stack.enter_context(
                patch_attr(
                    sequential_pipeline, "tqdm", self.progress(sequential_pipeline.tqdm)
                )
            )
            for owner, method, label in (
                (QuaRotModifier, "on_calibration_start", "quarot"),
                (flex_base, "search_alpha_beta", "flex_search"),
                (flex_base, "search_alpha_beta_distributed", "flex_search"),
                (flex_base, "apply_scales", "flex_apply"),
                (QuantizationModifier, "on_sequential_epoch_end", "mxfp"),
                (
                    Subgraph,
                    "forward",
                    lambda: "propagation"
                    if HooksMixin._HOOKS_DISABLED
                    else "calibration_forward",
                ),
            ):
                stack.enter_context(
                    patch_attr(owner, method, timed(getattr(owner, method), label))
                )
            for cache in (CPUCache, DiskCache):
                for method in ("onload", "update_offload"):
                    label = f"{cache.__name__}.{method}"
                    stack.enter_context(
                        patch_attr(
                            cache, method, counted(getattr(cache, method), label)
                        )
                    )
            yield


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="Existing local BF16 checkpoint"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New compressed checkpoint directory"
    )
    parser.add_argument("--device", choices=("npu", "cpu"), default="npu")
    parser.add_argument("--device-map", choices=("auto", "auto_offload"))
    parser.add_argument(
        "--profile-mode", choices=("summary", "sampled", "full"), default="summary"
    )
    parser.add_argument("--profile-sample-every", type=int, default=1000)
    parser.add_argument(
        "--max-memory",
        type=json.loads,
        help='HF max_memory JSON, e.g. {"cpu":"500GiB"}',
    )
    parser.add_argument("--offload-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Explicit HF dataset or local JSON list[str]/JSONL text/messages",
    )
    parser.add_argument("--split", default="train_sft")
    parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Global sample count; at least world_size",
    )
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--targets-per-subgraph", type=int, default=1)
    return parser.parse_args()


def resolve_device_map(device_map, world_size):
    device_map = device_map or ("auto_offload" if world_size > 1 else "auto")
    if device_map not in ("auto", "auto_offload") or (
        world_size > 1 and device_map != "auto_offload"
    ):
        raise ValueError(
            "Distributed smoke uses CT auto_offload; auto is single-process only"
        )
    return device_map


def calibration_data(args, tokenizer):
    """Tokenize this rank's upstream partition and exercise every padded batch."""
    if Path(args.dataset).is_file():
        # ModelSlim's calibration JSON is a top-level list of prompt strings.
        with Path(args.dataset).open(encoding="utf-8") as source:
            contents = source.read()
        if contents.lstrip().startswith("["):
            rows = json.loads(contents)
            if not rows or not all(isinstance(row, str) for row in rows):
                raise ValueError("Calibration JSON array must be a nonempty list[str]")
            dataset = Dataset.from_dict({"text": rows})
            partition = ReadInstruction.from_spec(
                get_rank_partition("train", args.samples)
            ).to_absolute({"train": len(dataset)})[0]
            dataset = dataset.select(
                range(partition.from_ or 0, partition.to or len(dataset))
            )
        else:
            dataset = load_dataset(
                "json",
                data_files=args.dataset,
                split=get_rank_partition("train", args.samples),
            )
    else:
        dataset = load_dataset(
            args.dataset, split=get_rank_partition(args.split, args.samples)
        )

    def tokenize(row):
        text = row.get("text")
        if text is None and "messages" in row:
            text = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False
            )
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Each calibration row needs nonempty text or messages")
        return tokenizer(
            text,
            truncation=True,
            max_length=args.sequence_length,
            add_special_tokens=False,
            return_token_type_ids=False,
        )

    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
    if not len(dataset) or any(not row["input_ids"] for row in dataset):
        raise ValueError("Each rank needs nonempty calibration samples")
    lengths = [sum(row["attention_mask"]) for row in dataset]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer, return_tensors="pt"),
    )
    # Materialize batches now, before transforms, without retaining a second cache.
    samples, tokens = 0, 0
    for batch in loader:
        ids, mask = batch["input_ids"], batch["attention_mask"]
        if ids.ndim != 2 or ids.shape != mask.shape:
            raise ValueError("Calibration requires matching [batch, tokens] IDs/masks")
        samples += ids.shape[0]
        tokens += int(mask.sum())
    if samples != len(dataset) or tokens != sum(lengths):
        raise ValueError("Calibration collation changed sample/valid-token counts")
    requested = ReadInstruction.from_spec(
        get_rank_partition("train", args.samples)
    ).to_absolute({"train": args.samples})[0]
    return loader, {
        "partition": get_rank_partition("train", args.samples),
        "samples_expected": (requested.to or args.samples) - (requested.from_ or 0),
        "samples_local": samples,
        "batches_local": len(loader),
        "tokens_local_valid": tokens,
        "sequence_length_min": min(lengths),
        "sequence_length_max": max(lengths),
    }


def calibration_preflight(args, report, *, tokenizer=None, generation_config=None):
    """Validate local metadata and all batches before loading model weights."""
    loader, stats, error = None, None, None
    try:
        if generation_config is None:
            if (args.model / "generation_config.json").is_file():
                generation_config = GenerationConfig.from_pretrained(
                    args.model, local_files_only=True
                )
            else:
                # Match the legacy local config fallback without loading weights.
                generation_config = GenerationConfig.from_model_config(
                    AutoConfig.from_pretrained(args.model, local_files_only=True)
                )
        prepare_generation_config(
            generation_config, report, phase="pre_load_validation"
        )
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        loader, stats = calibration_data(args, tokenizer)
    except Exception as caught:
        error = repr(caught)
    peers = [{"error": error, "stats": stats}]
    if dist.is_initialized():
        peers = [None] * dist.get_world_size()
        dist.all_gather_object(peers, {"error": error, "stats": stats})
    if any(peer["error"] is not None for peer in peers):
        raise ValueError(f"Calibration preflight failed: {peers}")
    counts = [peer["stats"]["samples_local"] for peer in peers]
    expected = [peer["stats"]["samples_expected"] for peer in peers]
    if counts != expected or sum(counts) != args.samples:
        raise ValueError(f"Calibration partition counts {counts}, expected {expected}")
    total_tokens = sum(peer["stats"]["tokens_local_valid"] for peer in peers)
    report.update(
        **stats,
        samples_global=sum(counts),
        tokens_global_valid=total_tokens,
        sequence_length_mean=total_tokens / sum(counts),
        batch_size=args.batch_size,
        dataset_source=str(args.dataset),
        preflight="passed",
    )
    for key in ("sequence_length_min", "sequence_length_max"):
        operation = min if key.endswith("min") else max
        report[key] = operation(peer["stats"][key] for peer in peers)
    logger.info(
        "Calibration preflight: {}",
        {
            key: value
            for key, value in report.items()
            if key not in ("stages", "cache_io", "versions", "generation_config")
        },
    )
    return loader, tokenizer


def model_preflight(model, flex, report):
    """Recheck loaded metadata and the existing Flex layout before transforms."""
    error = None
    try:
        prepare_generation_config(model.generation_config, report)
        flex._preflight(model)
    except Exception as caught:
        error = repr(caught)
    errors = [error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    if any(error is not None for error in errors):
        raise ValueError(f"Model preflight failed: {errors}")
    report["flex_max_tokens"] = flex.max_tokens


@torch.no_grad()
def quantized_tiles(model, *, dequantized=False):
    """Check one 1x32 tile per MXFP format, without running a full forward."""
    tiles, seen = {}, set()
    for name, module in model.named_modules():
        scheme = getattr(module, "quantization_scheme", None)
        if scheme is None or scheme.weights is None or scheme.weights.num_bits in seen:
            continue
        seen.add(scheme.weights.num_bits)
        with align_module_device(module):
            weight = module.weight[:1, :32]
            scale = module.weight_scale[:1, :1]
            value = (
                weight
                if dequantized
                else fake_quantize(weight, scale, None, scheme.weights)
            )
            tiles[name] = {
                "weight": value.detach().cpu().clone(),
                "scale": scale.detach().cpu().clone(),
                "bits": scheme.weights.num_bits,
            }
    if seen != {4, 8}:
        raise ValueError(f"Expected mixed MXFP4/MXFP8, found {seen}")
    return tiles


def prepare_generation_config(config, report, *, phase="load_validation"):
    """Unset inactive GLM sampling metadata, recording the original value."""
    details = report.setdefault("generation_config", {"adjustments": []})
    details[phase] = "pending"
    if config.do_sample is not True and config.top_p not in (None, 1.0):
        details["adjustments"].append(
            {
                "field": "top_p",
                "original": config.top_p,
                "value": None,
                "action": "unset_non_sampling_top_p",
                "do_sample": config.do_sample,
                "phase": phase,
            }
        )
        config.top_p = None
    try:
        config.validate(strict=True)
    except ValueError:
        details[phase] = "failed"
        raise
    details[phase] = "passed"


def main():
    args = parse_args()
    world_size, rank = (
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("RANK", "0")),
    )
    if not args.model.is_dir():
        raise ValueError("--model must be an existing local checkpoint directory")
    if (
        args.samples < world_size
        or args.sequence_length <= 0
        or args.targets_per_subgraph <= 0
        or args.batch_size <= 0
        or args.profile_sample_every <= 0
    ):
        raise ValueError(
            "Use positive lengths/partition size and at least one sample per rank"
        )
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(
            "Use a new output directory; existing checkpoints are not overwritten"
        )
    device_map = resolve_device_map(args.device_map, world_size)
    if args.device == "npu":
        os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
        os.environ.setdefault("HCCL_HOST_SOCKET_PORT_RANGE", "auto")
        import torch_npu  # noqa: F401

        torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if world_size > 1:
        dist.init_process_group(
            backend="cpu:gloo,npu:hccl" if args.device == "npu" else "gloo",
            timeout=timedelta(hours=3),
        )
    profile = Profile(args.device, args.profile_mode, args.profile_sample_every)
    report_dir = args.report_dir or args.output.with_name(args.output.name + "-profile")
    report_dir.mkdir(parents=True, exist_ok=True)
    offload = args.offload_dir or args.output.with_name(args.output.name + "-offload")
    memory = args.max_memory
    if memory:
        memory = {
            int(key) if key.isdecimal() else key: value for key, value in memory.items()
        }
    loading_args = dict(
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=device_map,
        offload_folder=str(offload),
    )
    if memory:
        loading_args["max_memory"] = memory
    profile.report.update(
        rank=rank,
        world_size=world_size,
        device_map=device_map,
        model=str(args.model),
        samples_global=args.samples,
        sequence_length=args.sequence_length,
        flex_max_tokens=None,
        versions={
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "compressed-tensors", "llmcompressor")
        },
        status="running",
    )
    profile.report["commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    try:
        recipe = Recipe.create_instance(
            str(Path(__file__).with_name("mixed_mxfp.yaml"))
        ).modifiers
        recipe[0].precision = torch.float32
        recipe[1].max_tokens = None
        with profile.stage("pre_load_preflight"):
            loader, tokenizer = calibration_preflight(args, profile.report)
        with profile.stage("model_load"), load_context():
            model = AutoModelForCausalLM.from_pretrained(
                str(args.model), **loading_args
            ).eval()
        with profile.stage("model_preflight"):
            model_preflight(model, recipe[1], profile.report)
        with profile.stage("oneshot_total"), profile.instrument():
            oneshot(
                model=model,
                processor=tokenizer,
                dataset=loader,
                recipe=recipe,
                pipeline="sequential",
                sequential_targets=[
                    r"re:.*\.input_layernorm$",
                    "ExpertMLP",
                    "GlmMoeDsaMLP",
                ],
                sequential_targets_per_subgraph=args.targets_per_subgraph,
                propagate_error=True,
                moe_calibrate_all_experts=False,
            )
        profile.report["quarot"] = model.config.quarot_config
        profile.report["flex"] = model.config.flex_smooth_config
        tiles = quantized_tiles(model) if rank == 0 else None
        if dist.is_initialized():
            dist.barrier()  # Finish the source's reads before any rank compresses.
        with profile.stage("save"):
            # Match GenerationConfig.save_pretrained before expensive compression.
            model.generation_config.validate(strict=True)
            profile.report["generation_config"]["pre_save_validation"] = "passed"
            model.save_pretrained(args.output, save_compressed=True)
            if rank == 0:
                tokenizer.save_pretrained(args.output)
        # Free the transformed model before loading the output. Only rank 0 reloads;
        # every rank must leave distributed mode first to avoid collective loading.
        del model, recipe
        reset_session()
        gc.collect()
        if args.device == "npu":
            torch.npu.empty_cache()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        if rank == 0:
            with profile.stage("reload"), load_context():
                restored = AutoModelForCausalLM.from_pretrained(
                    str(args.output),
                    **loading_args,
                    quantization_config=CompressedTensorsConfig(dequantize=True),
                )
            actual = quantized_tiles(restored, dequantized=True)
            for name, expected in tiles.items():
                for key in ("weight", "scale"):
                    torch.testing.assert_close(
                        actual[name][key], expected[key], rtol=0, atol=0
                    )
            profile.report["reload_tiles"] = {
                name: value["bits"] for name, value in tiles.items()
            }
            profile.report["reload_tiles_exact"] = True
            profile.report["checkpoint_bytes"] = sum(
                path.stat().st_size for path in args.output.glob("*.safetensors")
            )
        profile.report["status"] = "passed"
    except Exception as error:
        profile.report.update(status="failed", error=repr(error))
        raise
    finally:
        (report_dir / f"rank-{rank}.json").write_text(
            json.dumps(profile.report, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
