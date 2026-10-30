#!/usr/bin/env python3
"""Create one W4A16 iMatrix checkpoint for the ablation manifest."""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from compressed_tensors.quantization import preset_name_to_scheme
from compressed_tensors.utils.impl_backend import ImplBackend
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context


def _load_model(model_cfg: dict):
    model_id = model_cfg["hf_id"]
    if model_cfg["id"] == "qwen3.8-27b":
        from transformers import Qwen3_5ForConditionalGeneration

        with load_context(Qwen3_5ForConditionalGeneration):
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        processor = AutoProcessor.from_pretrained(model_id)
        return model, None, processor, processor

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer, None, tokenizer


def _patch_triton_counter(counter: dict[str, int]) -> None:
    """Count backend dispatches so each checkpoint proves Triton was used."""
    name = "_grid_search_observer"
    entries = ImplBackend._backends[name]
    replaced = []
    for backend, requirement, priority in entries:
        if backend.__name__ == "_grid_search_observer_triton":
            original = backend

            def tracked(*args, _original=original, **kwargs):
                counter["calls"] += 1
                return _original(*args, **kwargs)

            tracked.__name__ = original.__name__
            replaced.append((tracked, requirement, priority))
        else:
            replaced.append((backend, requirement, priority))
    ImplBackend._backends[name] = replaced


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--objective-id", required=True)
    parser.add_argument("--power", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Quantization must run on a CUDA GPU")
    torch.cuda.set_device(0)

    manifest = json.loads(args.manifest.read_text())
    model_cfg = next(m for m in manifest["models"] if m["id"] == args.model_id)
    objective = next(o for o in manifest["objectives"] if o["id"] == args.objective_id)
    if args.power not in manifest["powers"]:
        raise ValueError(f"Unsupported power {args.power}")

    scheme = preset_name_to_scheme("W4A16", ["Linear"])
    scheme.weights.group_size = manifest["group_size"]
    scheme.weights.observer = manifest["observer"]
    scheme.weights.observer_kwargs = {
        "norm": args.power,
        "patience": manifest["patience"],
        "grid": manifest["grid"],
        "maxshrink": manifest["maxshrink"],
        "use_groupabsmax_weighting": objective["use_groupabsmax_weighting"],
        "strict": True,
    }

    ignore = ["lm_head"]
    if model_cfg["id"] == "qwen3.8-27b":
        ignore.extend(
            [
                "re:visual.*",
                "re:model.visual.*",
                r"re:.*lm_head",
                r"re:.*embed_tokens$",
                r"re:.*linear_attn\.in_proj_a$",
                r"re:.*linear_attn\.in_proj_b$",
            ]
        )

    print(
        json.dumps(
            {
                "event": "quantization_started",
                "model": model_cfg["hf_id"],
                "power": args.power,
                "objective": objective,
                "observer_kwargs": scheme.weights.observer_kwargs,
                "cuda_device": torch.cuda.get_device_name(0),
            }
        ),
        flush=True,
    )

    model, tokenizer, processor, saved_preprocessor = _load_model(model_cfg)
    counter = {"calls": 0}
    _patch_triton_counter(counter)
    recipe = [QuantizationModifier(config_groups={"group_0": scheme}, ignore=ignore)]
    seed = manifest["calibration_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    oneshot(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        dataset=manifest["calibration_dataset"],
        splits=manifest["calibration_split"],
        recipe=recipe,
        max_seq_length=manifest["max_seq_length"],
        num_calibration_samples=manifest["calibration_samples"],
        shuffle_calibration_samples=True,
        moe_calibrate_all_experts=True,
    )
    if counter["calls"] == 0:
        raise RuntimeError("No imatrix grid-search call dispatched to Triton")

    staging = args.output.with_name(
        f".{args.output.name}.partial-{__import__('os').getpid()}"
    )
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    model.save_pretrained(staging, save_compressed=True)
    saved_preprocessor.save_pretrained(staging)
    (staging / "ablation_metadata.json").write_text(
        json.dumps(
            {
                "model_id": model_cfg["hf_id"],
                "power": args.power,
                "group_size": scheme.weights.group_size,
                "objective": objective,
                "observer_kwargs": scheme.weights.observer_kwargs,
                "triton_grid_search_calls": counter["calls"],
                "calibration": {
                    "dataset": manifest["calibration_dataset"],
                    "split": manifest["calibration_split"],
                    "samples": manifest["calibration_samples"],
                    "seed": manifest["calibration_seed"],
                    "max_seq_length": manifest["max_seq_length"],
                },
            },
            indent=2,
        )
        + "\n"
    )
    if args.output.exists():
        shutil.rmtree(args.output)
    staging.rename(args.output)
    print(
        json.dumps(
            {
                "event": "quantization_completed",
                "output": str(args.output),
                "triton_grid_search_calls": counter["calls"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
