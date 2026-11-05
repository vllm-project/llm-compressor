#!/usr/bin/env python3
"""Quantize one NVFP4 GPTQ + imatrix-observer ablation variant."""

from __future__ import annotations

import json
import shutil
import sys
import time

import quantize_nvfp4_imatrix_variant as imatrix
import torch
import transformers

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier


def _make_recipe(spec: dict, power: float, objective: str):
    scheme = imatrix._make_nvfp4_scheme(
        spec["targets"], power, objective, preset=spec.get("scheme", "NVFP4")
    )
    return [
        GPTQModifier(config_groups={"group_0": scheme}, ignore=list(spec["ignore"]))
    ]


def main() -> int:
    args = imatrix.parse_args()
    if args.objective == "groupmaxinv":
        args.objective = "groupmax_div"
    if args.power not in (2.0, 2.2, 2.4, 2.6, 2.8, 3.0):
        raise ValueError(f"unsupported power: {args.power}")

    spec = imatrix.MODEL_SPECS[args.model]
    if args.module_profile == "attention_only":
        spec = imatrix._attention_only_spec(spec)

    variant_dir = args.variant_dir.resolve()
    variant_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_link = variant_dir / "checkpoint"
    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir.resolve()
        checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_link.is_symlink():
            if checkpoint_link.resolve() != checkpoint_dir:
                raise ValueError(
                    f"checkpoint link {checkpoint_link} points to "
                    f"{checkpoint_link.resolve()}, expected {checkpoint_dir}"
                )
        elif checkpoint_link.exists():
            raise ValueError(
                f"checkpoint destination was provided but {checkpoint_link} "
                "already exists and is not a symlink"
            )
        else:
            checkpoint_link.symlink_to(checkpoint_dir, target_is_directory=True)
    else:
        checkpoint_dir = checkpoint_link
    temp_dir = checkpoint_dir.with_name(f".{checkpoint_dir.name}.incomplete")

    if (checkpoint_dir / "config.json").is_file():
        violations = imatrix._checkpoint_violations(
            checkpoint_dir, args.model, args.module_profile
        )
        try:
            existing = json.loads(
                (checkpoint_dir / "imatrix_ablation_config.json").read_text()
            )
        except (OSError, json.JSONDecodeError):
            existing = {}
        expected = {
            "model": args.model,
            "objective": args.objective,
            "power": args.power,
            "module_profile": args.module_profile,
            "quantization_method": "GPTQ+imatrix",
        }
        for key, value in expected.items():
            if existing.get(key) != value:
                violations.append(
                    f"existing checkpoint metadata {key}={existing.get(key)!r}, "
                    f"expected {value!r}"
                )
        if not violations:
            print(f"Matching GPTQ+imatrix checkpoint already exists: {checkpoint_dir}")
            return 0
        archive = checkpoint_dir.with_name(
            f"{checkpoint_dir.name}_invalid_gptq_imatrix_"
            + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        )
        print(
            f"Archiving non-matching checkpoint at {archive}: {violations}", flush=True
        )
        checkpoint_dir.rename(archive)
        if checkpoint_link.is_symlink() and not checkpoint_link.exists():
            checkpoint_link.unlink()
            checkpoint_link.symlink_to(checkpoint_dir, target_is_directory=True)

    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    started = time.time()
    print(
        f"Loading {spec['model_id']} for GPTQ+imatrix: power={args.power:g}, "
        f"objective={args.objective}, profile={args.module_profile}",
        flush=True,
    )
    model, preprocessor = imatrix._load_model_and_processor(spec)
    imatrix._validate_router_linear_ignores(model, spec)
    attention_targets = None
    if args.module_profile == "attention_only":
        attention_targets = imatrix._expected_attention_only_targets(model, args.model)
        if args.model == "gemma4_26b_a4b_it":
            # vLLM rewrites Gemma4 module paths from
            # `model.language_model.*` to `language_model.model.*`, but leaves
            # regex targets unchanged. Store exact target paths so vLLM can
            # map them before matching the fused qkv projection.
            spec["targets"] = attention_targets
    if args.model == "nemotron35_lightning_30b_a3b" and args.module_profile == "all":
        imatrix._validate_nemotron_attention_ignores(model, spec)

    recipe = _make_recipe(spec, args.power, args.objective)
    kwargs = {
        "model": model,
        "recipe": recipe,
        "dataset": imatrix.CALIBRATION_DATASET,
        "splits": imatrix.CALIBRATION_SPLIT,
        "max_seq_length": imatrix.MAX_SEQUENCE_LENGTH,
        "num_calibration_samples": imatrix.NUM_CALIBRATION_SAMPLES,
        "moe_calibrate_all_experts": spec["moe_calibrate_all_experts"],
        "preprocessing_num_workers": 8,
        "dataloader_num_workers": 0,
    }
    kwargs["processor" if spec["processor"] else "tokenizer"] = preprocessor
    oneshot(**kwargs)

    temp_dir.mkdir(parents=True)
    model.save_pretrained(temp_dir, save_compressed=True)
    preprocessor.save_pretrained(temp_dir)
    torch.cuda.synchronize()

    violations = imatrix._checkpoint_violations(
        temp_dir,
        args.model,
        args.module_profile,
        expected_attention_targets=attention_targets,
    )
    if violations:
        raise RuntimeError(
            f"checkpoint violates model-specific NVFP4 policy: {violations}"
        )

    metadata = {
        "model": args.model,
        "model_id": spec["model_id"],
        "scheme": spec.get("scheme", "NVFP4"),
        "quantization_method": "GPTQ+imatrix",
        "modifier": "GPTQModifier",
        "weight_observer": "imatrix_mse",
        "objective": args.objective,
        "power": args.power,
        "calibration_dataset": imatrix.CALIBRATION_DATASET,
        "calibration_split": imatrix.CALIBRATION_SPLIT,
        "num_calibration_samples": imatrix.NUM_CALIBRATION_SAMPLES,
        "max_sequence_length": imatrix.MAX_SEQUENCE_LENGTH,
        "targets": spec["targets"],
        "module_profile": args.module_profile,
        "target_module_names": attention_targets,
        "ignore": spec["ignore"],
        "moe_calibrate_all_experts": spec["moe_calibrate_all_experts"],
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "llmcompressor_version": __import__("llmcompressor").__version__,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    imatrix._write_json(temp_dir / "imatrix_ablation_config.json", metadata)
    temp_dir.rename(checkpoint_dir)
    print(f"Saved compressed GPTQ+imatrix checkpoint: {checkpoint_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"GPTQ+IMATRIX QUANTIZATION FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise
