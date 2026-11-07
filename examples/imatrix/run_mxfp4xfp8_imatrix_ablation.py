#!/usr/bin/env python3
"""Run the MXFP4xFP8 imatrix power/objective ablation.

Quantization is launched through ``chg`` in the rhdev environment. Evaluation
uses the repository's eval-agent CLI and the designated vLLM environment.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEME_NAME = "MXFP4XFP8"
SCHEME_SLUG = SCHEME_NAME.lower()
DEFAULT_RUN_ROOT = Path(
    os.environ.get("MXFP4XFP8_RUN_ROOT", ROOT / "runs/mxfp4xfp8_imatrix_ablation")
)
HF_CACHE = Path(os.environ.get("HF_HOME", ROOT / "hf_cache"))
for _cache_var in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
    os.environ.setdefault(_cache_var, str(HF_CACHE))
SKILL_ROOT = Path(os.environ.get("EVAL_AGENT_SKILL_DIR", ROOT / "eval-agent"))
EVAL_AGENT = Path(os.environ.get("EVAL_AGENT", ROOT / ".venv/bin/eval-agent"))
VLLM_ENV = Path(os.environ.get("VLLM_ENV", ROOT / ".venvs/vllm"))
RHVENV_PYTHON = Path(os.environ.get("RHVENV_PYTHON", ROOT / ".venvs/rhdev/bin/python"))
VLLM_BINARY = Path(os.environ.get("VLLM_BINARY", VLLM_ENV / "bin/vllm"))
LIGHTEVAL_ENV = Path(os.environ.get("LIGHTEVAL_ENV", ROOT / ".venvs/lighteval"))
DATASET_ID = "mlabonne/open-perfectblend"
DATASET_SPLIT = "train[:512]"
MAX_SEQ_LENGTH = 2048
NUM_CALIBRATION_SAMPLES = 512
POWERS = (2.0, 2.2, 2.4, 2.6, 2.8, 3.0)
OBJECTIVES = ("importance_error", "weight_magnitude")
OBSERVER_SETTINGS = {
    "default": {
        "observer": "imatrix_mse",
        "grid": 100,
        "grid_type": "exponential",
        "maxshrink": 0.95,
        "expand": 1.0,
    },
    "expanded": {
        "observer": "nvfp4_expanded_imatrix",
        "grid": 200,
        "grid_type": "exponential",
        "maxshrink": 63 / 64,
        "expand": 1.0,
    },
}
TASKS = {
    "instruct": ("gsm8k_platinum_cot_llama", "mmlu_cot_llama", "math_500"),
    "reasoning": ("gpqa_diamond",),
}
# MMLU's 5,700 smoke requests take hours at the per-subject smoke limit. Keep
# it in every full seed round, while using the smaller instruct tasks to
# validate serving and calibrate concurrency before the grid starts.
SMOKE_TASKS = {
    "instruct": ("gsm8k_platinum_cot_llama", "math_500"),
    "reasoning": TASKS["reasoning"],
}
MAX_LENGTHS = {"instruct": 20480, "reasoning": 69632}
# Qwen3-30B-A3B's standard config tops out at 40,960, but the official model
# card documents YaRN extension. GPQA's registry allows 65,536 output tokens;
# scale its 32,768-token native rotary context by 2.125 to cover the 69,632
# reasoning limit. Keep the extension on reasoning only to avoid changing
# short-context instruct inference.
MODEL_MAX_LENGTHS = {
    "qwen3_30b_a3b": {"reasoning": 69632},
    "qwen25_7b": {"reasoning": 69632},
}
MODEL_CATEGORY_SERVER_ARGS = {
    "qwen3_30b_a3b": {
        "reasoning": [
            "--hf-overrides",
            json.dumps(
                {
                    "rope_parameters": {
                        "rope_type": "yarn",
                        "factor": 2.125,
                        "original_max_position_embeddings": 32768,
                        "rope_theta": 1000000,
                    }
                },
                separators=(",", ":"),
            ),
        ],
    },
    "qwen25_7b": {
        "reasoning": [
            "--hf-overrides",
            json.dumps(
                {
                    "rope_parameters": {
                        "rope_type": "yarn",
                        "factor": 2.125,
                        "original_max_position_embeddings": 32768,
                        "rope_theta": 1000000,
                    }
                },
                separators=(",", ":"),
            ),
        ],
    },
}
MODEL_RUN_DIR_SUFFIXES = {
    "qwen3_30b_a3b": {"reasoning": "yarn2p125"},
    "qwen25_7b": {"reasoning": "yarn2p125"},
}
INITIAL_CONCURRENCY = {"instruct": 64, "reasoning": 32}
TIMEOUTS = {"instruct": 1200, "reasoning": 3600}
PORT_BASE = 28100
PORT_BASE += int(os.environ.get("MXFP4_ABLATION_PORT_OFFSET", "0"))
PORT_MODEL_STRIDE = 640
PORT_VARIANT_STRIDE = 8
PORT_REASONING_OFFSET = 320
MODEL_NAME_PREFIX = SCHEME_SLUG
RESULTS_MD = ROOT / f"{SCHEME_SLUG}_ablation_results.md"
MONITOR_INTERVAL_SECONDS = 600
RETAIN_CHECKPOINT_MODELS_ACROSS_SEEDS = {
    "qwen25_7b",
    "qwen38_27b",
    "qwen3_30b_a3b",
}
STATE_LOCK = threading.Lock()
DISK_RESERVATION_LOCK = threading.Condition()
DISK_RESERVED_BYTES = 0
MIN_FREE_DISK_BYTES = 25 * 1024**3
IMPLEMENTATION_FILES = (
    "src/llmcompressor/observers/imatrix.py",
    "src/llmcompressor/observers/grid_search.py",
    "src/llmcompressor/utils/triton_utils.py",
    "tests/llmcompressor/observers/test_imatrix.py",
    "examples/imatrix/run_mxfp4xfp8_imatrix_ablation.py",
)

# Revisions captured by the completed MXFP4xFP8 runs. Keeping these pinned makes
# a rerun reproducible even when model cards or calibration data are updated.
PINNED_MODEL_REVISIONS = {
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "Qwen/Qwen3.8-27B": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "Qwen/Qwen3-30B-A3B": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
}
PINNED_DATASET_REVISION = "af60f3c18201652a83a93f46fcfee1b646ba3df7"


def eval_registry_path() -> Path:
    path = SKILL_ROOT / "eval_agent/benchmarks/registry.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find eval-agent registry at {path}. Set "
            "EVAL_AGENT_SKILL_DIR to the directory containing SKILL.md and "
            "eval_agent/benchmarks/registry.yaml."
        )
    return path


def max_model_length(model_key: str, category: str) -> int:
    """Return the eval-agent default capped at the model's supported context."""
    return min(
        MAX_LENGTHS[category],
        MODEL_MAX_LENGTHS.get(model_key, {}).get(category, MAX_LENGTHS[category]),
    )


def input_activation_config() -> dict[str, Any]:
    """Return the activation quantization declared by the MXFP4XFP8 preset."""
    return {
        "num_bits": 8,
        "type": "float",
        "strategy": "group",
        "symmetric": True,
        "dynamic": True,
        "group_size": 128,
    }


def evaluation_backend_metadata() -> dict[str, str]:
    """Describe and enforce the effective W4A8 path used by these runs."""
    return {
        "effective_quantization": (
            "MXFP4 weights (W4) with dynamic FP8 activations (A8)."
        ),
        "expected_linear_kernel": "HummingMxFp4LinearKernel",
        "expected_moe_kernel": "Humming experts for MoE models",
        "note": (
            "The checkpoint must retain the MXFP4XFP8 dynamic FP8 activation "
            "configuration. Evaluation rejects logs that do not show the "
            "Humming MXFP4 linear kernel or that report activation quantization "
            "being ignored."
        ),
    }


MODELS: dict[str, dict[str, Any]] = {
    "qwen38_27b": {
        "id": "Qwen/Qwen3.8-27B",
        "architecture": "qwen35",
        "ignore": [
            "re:visual.*",
            "re:model.visual.*",
            "re:.*lm_head",
            "re:.*embed_tokens$",
            r"re:.*linear_attn\.in_proj_a$",
            r"re:.*linear_attn\.in_proj_b$",
        ],
        "reasoning_parser": "qwen3",
        "server_args": ["--max-num-seqs", "900"],
    },
    "qwen3_30b_a3b": {
        "id": "Qwen/Qwen3-30B-A3B",
        "architecture": "causal",
        "ignore": [
            "lm_head",
            r"re:.*mlp\.gate$",
            r"re:.*mlp\.shared_expert_gate$",
        ],
        "reasoning_parser": "qwen3",
        "server_args": [],
    },
    "qwen25_7b": {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "architecture": "causal",
        "ignore": ["lm_head"],
        "reasoning_parser": None,
        "server_args": [],
    },
}

_EXCLUDED_MODELS = {
    key.strip()
    for key in os.environ.get("MXFP4_ABLATION_EXCLUDE_MODELS", "").split(",")
    if key.strip()
}
if _EXCLUDED_MODELS:
    unknown = _EXCLUDED_MODELS - MODELS.keys()
    if unknown:
        raise ValueError(f"Unknown excluded models: {sorted(unknown)}")
    MODELS = {
        key: value for key, value in MODELS.items() if key not in _EXCLUDED_MODELS
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def powerslug(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def objective_uses_magnitude(objective: str) -> bool:
    return objective == "weight_magnitude"


def implementation_source_hash() -> str:
    digest = hashlib.sha256()
    for relative_path in IMPLEMENTATION_FILES:
        path = ROOT / relative_path
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_mxfp4xfp8_checkpoint(checkpoint_dir: Path) -> str | None:
    """Reject checkpoints that lost their dynamic FP8 activation config."""
    config_path = checkpoint_dir / "config.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return f"cannot read checkpoint config {config_path}: {exc}"

    groups = config.get("quantization_config", {}).get("config_groups", {})
    for group in groups.values():
        activation = group.get("input_activations") or {}
        weights = group.get("weights") or {}
        if (
            activation.get("num_bits") == 8
            and activation.get("type") == "float"
            and activation.get("strategy") == "group"
            and activation.get("dynamic") is True
            and activation.get("group_size") == 128
            and weights.get("num_bits") == 4
            and weights.get("type") == "float"
            and weights.get("group_size") == 32
            and weights.get("scale_dtype") == "torch.uint8"
        ):
            return None
    return (
        "checkpoint config does not declare MXFP4 weights with dynamic "
        "group-128 FP8 input activations"
    )


def _validate_vllm_fp8_kernel_log(log_path: Path, model_key: str) -> str | None:
    """Ensure vLLM actually selected kernels that consume FP8 activations."""
    try:
        log = log_path.read_text(errors="replace")
    except OSError as exc:
        return f"cannot read vLLM server log {log_path}: {exc}"
    if "Using HummingMxFp4LinearKernel" not in log:
        return "vLLM did not select HummingMxFp4LinearKernel"
    if re.search(r"activation quantization.*ignored|weight-only \(A16\)", log, re.I):
        return "vLLM reported that activation quantization was ignored"
    if model_key == "qwen3_30b_a3b" and not re.search(r"Using Humming\w*Experts", log):
        return "vLLM did not select a Humming MoE kernel"
    return None


def observer_variants(model_key: str) -> tuple[str, ...]:
    # Qwen3.8-27B is the default-vs-expanded observer comparison model.
    return ("default", "expanded") if model_key == "qwen38_27b" else ("expanded",)


def job_key(model_key: str, power: float, objective: str, observer_variant: str) -> str:
    return f"{model_key}__{observer_variant}__p{powerslug(power)}__{objective}"


def job_specs() -> list[dict[str, Any]]:
    return [
        {
            "key": job_key(model_key, power, objective, observer_variant),
            "model": model_key,
            "observer_variant": observer_variant,
            "power": power,
            "objective": objective,
        }
        for model_key in MODELS
        for observer_variant in observer_variants(model_key)
        for power in POWERS
        for objective in OBJECTIVES
    ]


def baseline_specs() -> list[dict[str, Any]]:
    return [
        {
            "model": model_key,
            "observer_variant": observer_variant,
            "power": 2.0,
            "objective": "importance_error",
        }
        for model_key in MODELS
        for observer_variant in observer_variants(model_key)
    ]


def _revision_info() -> dict[str, Any]:
    revisions = {
        spec["id"]: PINNED_MODEL_REVISIONS[spec["id"]] for spec in MODELS.values()
    }
    return {
        "models": revisions,
        "dataset": {"id": DATASET_ID, "revision": PINNED_DATASET_REVISION},
    }


def _generation_params(model_id: str, revision: str) -> dict[str, Any]:
    from transformers import GenerationConfig

    config = GenerationConfig.from_pretrained(model_id, revision=revision)
    result: dict[str, Any] = {}
    temperature = getattr(config, "temperature", None)
    if temperature is not None and temperature > 0:
        result["temperature"] = temperature
        for name in ("top_p", "top_k"):
            value = getattr(config, name, None)
            if value is not None:
                result[name] = value
    return result


def _chat_template(model_key: str, model_revision: str) -> str:
    from huggingface_hub import hf_hub_download

    spec = MODELS[model_key]
    repo_id = spec["id"]
    revision = model_revision
    try:
        template_path = hf_hub_download(
            repo_id=repo_id,
            filename="chat_template.jinja",
            revision=revision,
        )
        return Path(template_path).read_text()
    except Exception:
        pass

    from transformers import AutoProcessor, AutoTokenizer

    try:
        processor = AutoProcessor.from_pretrained(
            repo_id, revision=revision, trust_remote_code=True
        )
        template = getattr(processor, "chat_template", None)
        if template is None and getattr(processor, "tokenizer", None) is not None:
            template = processor.tokenizer.chat_template
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id, revision=revision, trust_remote_code=True
        )
        template = tokenizer.chat_template
    if isinstance(template, dict):
        template = template.get("default") or next(iter(template.values()))
    if not isinstance(template, str) or not template.strip():
        raise RuntimeError(f"No chat template found for {repo_id}@{revision}")
    return template


def prepare(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "experiment_manifest.json"
    manifest = read_json(manifest_path)
    if manifest is not None:
        source_hash = implementation_source_hash()
        calibration = manifest.setdefault("calibration", {})
        weight = calibration.setdefault("weight", {})
        manifest["experiment_type"] = "mxfp4xfp8_imatrix_ablation"
        calibration["scheme"] = "MXFP4XFP8 compressed-tensors preset"
        calibration["input_activations"] = {
            "num_bits": 8,
            "type": "float",
            "strategy": "group",
            "symmetric": True,
            "dynamic": True,
            "group_size": 128,
        }
        weight.update(
            {
                "observers": OBSERVER_SETTINGS,
                "default_observer_variant": "default",
                "expanded_observer_variant": "expanded",
                "patience": 0,
                "powers": list(POWERS),
                "objectives": {
                    "importance_error": "importance * abs(Q(w)-w)^power",
                    "weight_magnitude": (
                        "importance * abs(Q(w)-w)^power * " "sqrt(w^2 + groupabsmax^2)"
                    ),
                },
            }
        )
        manifest["jobs"] = [
            {
                **spec,
                "key": job_key(
                    spec["model"],
                    spec["power"],
                    spec["objective"],
                    spec["observer_variant"],
                ),
            }
            for spec in job_specs()
        ]
        if manifest.get("implementation_source_sha256") != source_hash:
            manifest["implementation_source_sha256"] = source_hash
            manifest["implementation_files"] = list(IMPLEMENTATION_FILES)
        manifest["updated_at"] = utc_now()
        manifest["design_change"] = (
            "MXFP4XFP8 imatrix ablation with exponential E8M0 search."
        )
        evaluation = manifest.setdefault("evaluation", {})
        evaluation.update(evaluation_backend_metadata())
        atomic_json(manifest_path, manifest)
        return manifest

    revisions = _revision_info()
    from transformers import AutoConfig

    model_manifest: dict[str, Any] = {}
    templates_dir = root / "chat_templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    for model_key, spec in MODELS.items():
        model_id = spec["id"]
        revision = revisions["models"][model_id]
        config = AutoConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=True
        )
        template = _chat_template(model_key, revision)
        template_path = templates_dir / f"{model_key}.jinja"
        template_path.write_text(template)
        model_manifest[model_key] = {
            "id": model_id,
            "revision": revision,
            "model_type": config.model_type,
            "architectures": getattr(config, "architectures", None),
            "max_position_embeddings": getattr(config, "max_position_embeddings", None),
            "generation_params": _generation_params(model_id, revision),
            "chat_template": str(template_path),
            "ignore": spec["ignore"],
            "reasoning_parser": spec["reasoning_parser"],
        }

    from datasets import load_dataset

    raw_dataset = load_dataset(
        DATASET_ID,
        split=DATASET_SPLIT,
        revision=revisions["dataset"]["revision"],
    )
    dataset_fingerprint = getattr(raw_dataset, "_fingerprint", None)
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    registry_path = eval_registry_path()
    registry_hash = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    manifest = {
        "experiment_type": "mxfp4xfp8_imatrix_ablation",
        "created_at": utc_now(),
        "reference_pr": "https://github.com/vllm-project/compressed-tensors/pull/913",
        "repo": str(ROOT),
        "repo_commit": git_commit,
        "repo_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "implementation_source_sha256": implementation_source_hash(),
        "implementation_files": list(IMPLEMENTATION_FILES),
        "model_revisions": revisions,
        "models": model_manifest,
        "calibration": {
            "dataset_id": DATASET_ID,
            "dataset_revision": revisions["dataset"]["revision"],
            "split": DATASET_SPLIT,
            "raw_dataset_fingerprint": dataset_fingerprint,
            "num_calibration_samples": NUM_CALIBRATION_SAMPLES,
            "max_seq_length": MAX_SEQ_LENGTH,
            "scheme": "MXFP4XFP8 compressed-tensors preset",
            "weight": {
                "num_bits": 4,
                "type": "float",
                "strategy": "group",
                "symmetric": True,
                "dynamic": False,
                "group_size": 32,
                "scale_dtype": "torch.uint8",
                "zp_dtype": "torch.uint8",
                "observers": OBSERVER_SETTINGS,
                "default_observer_variant": "default",
                "expanded_observer_variant": "expanded",
                "patience": 0,
                "powers": list(POWERS),
                "objectives": {
                    "importance_error": "importance * abs(Q(w)-w)^power",
                    "weight_magnitude": (
                        "importance * abs(Q(w)-w)^power * " "sqrt(w^2 + groupabsmax^2)"
                    ),
                },
            },
            "input_activations": {
                **input_activation_config(),
            },
        },
        "evaluation": {
            "registry": str(registry_path),
            "registry_sha256": registry_hash,
            "tasks": {key: list(value) for key, value in TASKS.items()},
            "lm_eval_venv": str(VLLM_ENV),
            "lighteval_venv": str(LIGHTEVAL_ENV),
            "smoke_concurrency_initial": INITIAL_CONCURRENCY,
            **evaluation_backend_metadata(),
        },
        "jobs": [
            {
                **spec,
                "key": job_key(
                    spec["model"],
                    spec["power"],
                    spec["objective"],
                    spec["observer_variant"],
                ),
            }
            for spec in job_specs()
        ],
    }
    atomic_json(manifest_path, manifest)
    return manifest


def _load_quant_model(model_key: str, revision: str):
    import torch
    import transformers

    from llmcompressor.utils import load_context

    spec = MODELS[model_key]
    if spec["architecture"] == "qwen35":
        model_cls = getattr(transformers, "Qwen3_5ForConditionalGeneration")
    else:
        model_cls = transformers.AutoModelForCausalLM
    with load_context(model_cls):
        model = model_cls.from_pretrained(
            spec["id"],
            revision=revision,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    if model_key == "qwen38_27b":
        processor = transformers.AutoProcessor.from_pretrained(
            spec["id"], revision=revision, trust_remote_code=True
        )
        return model, {"processor": processor}
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        spec["id"], revision=revision, trust_remote_code=True
    )
    return model, {"tokenizer": tokenizer}


def quantize_one(
    root: Path,
    model_key: str,
    power: float,
    objective: str,
    observer_variant: str,
    checkpoint_dir: Path | None = None,
) -> None:
    import torch
    from compressed_tensors.quantization import preset_name_to_scheme

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    manifest = prepare(root)
    spec = MODELS[model_key]
    entry = manifest["models"][model_key]
    revision = entry["revision"]
    job = job_key(model_key, power, objective, observer_variant)
    checkpoint_dir = checkpoint_dir or root / "checkpoints" / job
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model, processor_arg = _load_quant_model(model_key, revision)
    observer_settings = OBSERVER_SETTINGS[observer_variant]
    scheme = preset_name_to_scheme(SCHEME_NAME, ["Linear"])
    weights = scheme.weights
    weights.observer = observer_settings["observer"]
    weights.observer_kwargs = {
        "norm": float(power),
        "grid": observer_settings["grid"],
        "grid_type": observer_settings["grid_type"],
        "maxshrink": observer_settings["maxshrink"],
        "expand": observer_settings["expand"],
        "patience": 0,
        "magnitude_normalization": objective_uses_magnitude(objective),
    }
    recipe = QuantizationModifier(
        config_groups={"group_0": scheme},
        ignore=spec["ignore"],
    )
    atomic_json(
        checkpoint_dir / "ablation_config.json",
        {
            "job": job,
            "model_id": spec["id"],
            "model_revision": revision,
            "power": power,
            "objective": objective,
            "observer_variant": observer_variant,
            "observer": weights.observer,
            "observer_kwargs": weights.observer_kwargs,
            "scheme": scheme.model_dump(mode="json"),
            "ignore": spec["ignore"],
            "calibration_dataset": DATASET_ID,
            "dataset_revision": manifest["calibration"]["dataset_revision"],
            "split": DATASET_SPLIT,
            "samples": NUM_CALIBRATION_SAMPLES,
            "max_seq_length": MAX_SEQ_LENGTH,
        },
    )

    torch.accelerator.reset_peak_memory_stats()
    started = time.monotonic()
    oneshot(
        model=model,
        **processor_arg,
        model_revision=revision,
        trust_remote_code_model=True,
        precision="bfloat16",
        save_compressed=True,
        dataset="perfectblend",
        splits=DATASET_SPLIT,
        raw_kwargs={"revision": manifest["calibration"]["dataset_revision"]},
        recipe=[recipe],
        max_seq_length=MAX_SEQ_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        output_dir=str(checkpoint_dir),
        moe_calibrate_all_experts=True,
    )
    elapsed_s = time.monotonic() - started
    peak_memory_gb = torch.accelerator.max_memory_allocated() / (1024**3)
    model_files = list(checkpoint_dir.glob("*.safetensors"))
    if not model_files:
        raise RuntimeError(f"No compressed safetensors found in {checkpoint_dir}")
    atomic_json(
        checkpoint_dir / "quantization_complete.json",
        {
            "job": job,
            "completed_at": utc_now(),
            "elapsed_seconds": elapsed_s,
            "peak_cuda_memory_gib": peak_memory_gb,
            "checkpoint_bytes": sum(path.stat().st_size for path in model_files),
        },
    )
    print(
        f"QUANTIZED job={job} elapsed={elapsed_s / 60:.1f}m "
        f"peak_cuda={peak_memory_gb:.1f}GiB output={checkpoint_dir}",
        flush=True,
    )


def _run_logged(
    command: list[str],
    log_path: Path,
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n[{utc_now()}] COMMAND {shlex.join(command)}\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return proc.wait()


def _run_quant_reserved(root: Path, spec: dict[str, Any], checkpoint_dir: Path) -> int:
    command = [
        "chg",
        "run",
        "--gpus",
        "1",
        "--",
        str(RHVENV_PYTHON),
        str(Path(__file__).resolve()),
        "quantize",
        "--root",
        str(root),
        "--model",
        spec["model"],
        "--power",
        str(spec["power"]),
        "--objective",
        spec["objective"],
        "--observer-variant",
        spec["observer_variant"],
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    key = job_key(
        spec["model"], spec["power"], spec["objective"], spec["observer_variant"]
    )
    log = root / "quantization_logs" / f"{key}.log"
    return _run_logged(command, log)


def _kv_cache_peak(log_path: Path) -> float | None:
    try:
        data = log_path.read_text(errors="replace")
    except FileNotFoundError:
        return None
    values = [
        float(match.group(1))
        for match in re.finditer(r"GPU KV cache usage:\s*([\d.]+)%", data)
    ]
    return max(values) if values else None


def _server_command(
    root: Path,
    model_key: str,
    checkpoint_dir: Path,
    category: str,
    port: int,
    tensor_parallel_size: int,
    log_requests: bool,
    served_name: str | None = None,
) -> str:
    served_name = served_name or f"{MODEL_NAME_PREFIX}-{model_key}"
    server_args = [
        "env",
        # vLLM's environment currently has flashinfer-python 0.7.0.post1
        # and flashinfer-cubin 0.6.16.post3. FlashInfer documents this flag
        # for bypassing its strict version guard when the cubin package is
        # unavailable at the matching Python package version.
        "FLASHINFER_DISABLE_VERSION_CHECK=1",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1",
        str(VLLM_BINARY),
        "serve",
        str(checkpoint_dir),
        "--served-model-name",
        served_name,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-model-len",
        str(max_model_length(model_key, category)),
        "--port",
        str(port),
        "--chat-template",
        str(root / "chat_templates" / f"{model_key}.jinja"),
        "--trust-remote-code",
    ]
    reasoning_parser = MODELS[model_key]["reasoning_parser"]
    if reasoning_parser is not None:
        server_args.extend(["--reasoning-parser", reasoning_parser])
    server_args.extend(MODELS[model_key]["server_args"])
    server_args.extend(MODEL_CATEGORY_SERVER_ARGS.get(model_key, {}).get(category, []))
    if log_requests:
        server_args.append("--enable-log-requests")
    return shlex.join(
        ["chg", "run", "--gpus", str(tensor_parallel_size), "--", *server_args]
    )


def _eval_base_command(
    root: Path,
    model_key: str,
    checkpoint_dir: Path,
    category: str,
    run_dir: Path,
    port: int,
    concurrency: int,
    tensor_parallel_size: int,
    timeout: int,
    smoke_only: bool,
    resume: bool,
    selected_seeds: list[int] | None = None,
    task_names: tuple[str, ...] | None = None,
    served_name: str | None = None,
) -> list[str]:
    model_entry = prepare(root)["models"][model_key]
    served_name = served_name or f"{MODEL_NAME_PREFIX}-{model_key}"
    server_cmd = _server_command(
        root,
        model_key,
        checkpoint_dir,
        category,
        port,
        tensor_parallel_size,
        log_requests=smoke_only,
        served_name=served_name,
    )
    cmd = [str(EVAL_AGENT)]
    if resume:
        cmd.extend(["resume", "--run-dir", str(run_dir)])
        cmd.extend(["--num-concurrent", str(concurrency)])
        cmd.extend(["--timeout", str(timeout)])
        # Resume may also wait in chg's FIFO queue for a GPU. Keep the same
        # long startup window as new runs instead of eval-agent's 600s default.
        cmd.extend(["--health-timeout", "86400"])
        return cmd
    cmd.extend(
        [
            "run",
            "--model",
            served_name,
            "--server-cmd",
            server_cmd,
            "--gen-params",
            json.dumps(model_entry["generation_params"], separators=(",", ":")),
            "--category",
            category,
            "--tasks",
            ",".join(task_names or TASKS[category]),
            "--port",
            str(port),
            "--max-length",
            str(MAX_LENGTHS[category]),
            "--num-concurrent",
            str(concurrency),
            "--timeout",
            str(timeout),
            "--health-timeout",
            # The command may wait in chg's FIFO queue until a GPU is released.
            "86400",
            "--lm-eval-venv",
            str(VLLM_ENV),
            "--lighteval-venv",
            str(LIGHTEVAL_ENV),
            "--run-dir",
            str(run_dir),
        ]
    )
    if selected_seeds:
        cmd.extend(["--seeds", ",".join(str(seed) for seed in selected_seeds)])
    if smoke_only:
        cmd.append("--smoke-only")
    return cmd


def _retryable_failure(log_path: Path) -> str | None:
    try:
        text = log_path.read_text(errors="replace").lower()
    except FileNotFoundError:
        return None
    if "cuda out of memory" in text or "out of memory" in text:
        return "oom"
    if "preempted" in text or "http 507" in text or "aborted request" in text:
        return "kv_cache"
    # Ignore the run command's ``--timeout`` argument; only retry when the
    # log reports an actual timeout condition.
    if re.search(r"\b(?:timed out|timeout error|timeouterror)\b", text):
        return "timeout"
    return None


def _evaluation_log_failure(run_dir: Path) -> str | None:
    """Reject nominally successful harness exits with missing API responses."""
    for log_path in run_dir.rglob("*.log"):
        try:
            text = log_path.read_text(errors="replace").lower()
        except OSError:
            continue
        if "api call failed after" in text and "returning empty response" in text:
            return "api_retries_exhausted"
        if (
            "device-side assert triggered" in text
            or "enginecore encountered a fatal error" in text
        ):
            return "cuda_device_assert"
    return None


def evaluate_category(
    root: Path,
    spec: dict[str, Any],
    checkpoint_dir: Path,
    category: str,
    *,
    smoke: bool,
    concurrency: int,
    tensor_parallel_size: int = 1,
    max_smoke_adjustments: int = 2,
    seed: int | None = None,
    task_names: tuple[str, ...] | None = None,
    run_tag: str | None = None,
) -> dict[str, Any]:
    model_key = spec["model"]
    checkpoint_error = _validate_mxfp4xfp8_checkpoint(checkpoint_dir)
    if checkpoint_error is not None:
        return {"status": "failed", "reason": checkpoint_error}
    key = job_key(model_key, spec["power"], spec["objective"], spec["observer_variant"])
    smoke_base = root / "smoke" / key / category
    full_run_dir = root / "evaluations" / key / category
    if seed is not None:
        full_run_dir = full_run_dir / f"seed_{seed}"
        run_dir_suffix = run_tag or MODEL_RUN_DIR_SUFFIXES.get(model_key, {}).get(
            category
        )
        if run_dir_suffix:
            full_run_dir = full_run_dir.with_name(
                f"{full_run_dir.name}_{run_dir_suffix}"
            )
        elif max_model_length(model_key, category) < MAX_LENGTHS[category]:
            full_run_dir = full_run_dir.with_name(
                f"{full_run_dir.name}_ctx{max_model_length(model_key, category)}"
            )
    chosen_dir = smoke_base if smoke else full_run_dir
    chosen_dir.mkdir(parents=True, exist_ok=True)
    attempt = 0
    kv_peak = None
    # Reuse an interrupted full-run manifest on the first retry. This preserves
    # completed task/seed pairs and carries the long chg queue health timeout.
    resume_run_dir: Path | None = (
        chosen_dir if not smoke and (chosen_dir / "manifest.json").is_file() else None
    )
    resume = resume_run_dir is not None
    timeout = TIMEOUTS[category]
    smoke_session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    while True:
        attempt += 1
        if smoke:
            run_dir = chosen_dir / smoke_session / f"attempt_{attempt:02d}"
        elif resume and resume_run_dir is not None:
            run_dir = resume_run_dir
        elif tensor_parallel_size > 1 and not resume:
            run_dir = chosen_dir / f"tp{tensor_parallel_size}_attempt_{attempt:02d}"
        elif attempt > 1 and not resume:
            run_dir = chosen_dir / f"attempt_{attempt:02d}"
        else:
            run_dir = chosen_dir
        observer_order = observer_variants(model_key)
        # Comparison-only observer arms are intentionally excluded from the
        # general ablation matrix, but still need unique evaluation ports.
        observer_index = (
            observer_order.index(spec["observer_variant"])
            if spec["observer_variant"] in observer_order
            else len(observer_order)
        )
        variant_index = observer_index * len(POWERS) * len(OBJECTIVES)
        variant_index += POWERS.index(spec["power"]) * len(OBJECTIVES)
        variant_index += OBJECTIVES.index(spec["objective"])
        port = PORT_BASE + list(MODELS).index(model_key) * PORT_MODEL_STRIDE
        port += variant_index * PORT_VARIANT_STRIDE
        port += (0 if category == "instruct" else PORT_REASONING_OFFSET) + attempt
        command = _eval_base_command(
            root,
            model_key,
            checkpoint_dir,
            category,
            run_dir,
            port,
            concurrency,
            tensor_parallel_size,
            timeout,
            smoke,
            resume,
            selected_seeds=[seed] if seed is not None else None,
            task_names=task_names,
            # LiteLLM's shared disk-cache key omits base_url, so use a model
            # name that is unique to this checkpoint, category, and seed.
            served_name=(
                f"{MODEL_NAME_PREFIX}-{key}-{category}-seed{seed}"
                if seed is not None
                else f"{MODEL_NAME_PREFIX}-{key}-{category}-smoke"
            ),
        )
        run_key = f"{key}_{category}"
        if seed is not None:
            run_key += f"_seed{seed}"
        log_path = root / "orchestration_logs" / f"{run_key}_attempt_{attempt:02d}.log"
        code = _run_logged(command, log_path)
        server_log = run_dir / "logs/vllm_server.log"
        kv_peak = _kv_cache_peak(server_log)
        eval_failure = _evaluation_log_failure(run_dir)
        backend_failure = (
            _validate_vllm_fp8_kernel_log(server_log, model_key) if code == 0 else None
        )
        if backend_failure is not None:
            return {
                "status": "failed",
                "reason": backend_failure,
                "run_dir": str(run_dir),
                "attempts": attempt,
                "orchestration_log": str(log_path),
            }
        # KV-cache telemetry is required to calibrate smoke concurrency, but
        # full runs can complete successfully when vLLM does not emit usage
        # samples (for example, short reasoning tasks). Treat missing telemetry
        # as a failure only when a smoke run needs it for calibration.
        if (
            code == 0
            and eval_failure is None
            and (not smoke or (kv_peak is not None and kv_peak <= 85.0))
        ):
            return {
                "status": "completed",
                "run_dir": str(run_dir),
                "attempts": attempt,
                "num_concurrent": concurrency,
                "tensor_parallel_size": tensor_parallel_size,
                "kv_cache_peak_percent": kv_peak,
            }

        failure = _retryable_failure(log_path) or eval_failure
        if smoke and code == 0 and kv_peak is not None and kv_peak > 85.0:
            concurrency = max(1, concurrency // 2)
            if attempt >= max_smoke_adjustments:
                return {
                    "status": "failed",
                    "reason": (
                        "KV-cache remained above 85% after concurrency reductions"
                    ),
                    "run_dir": str(run_dir),
                    "num_concurrent": concurrency,
                    "kv_cache_peak_percent": kv_peak,
                }
            continue

        if failure == "oom" and smoke and tensor_parallel_size == 1:
            tensor_parallel_size = 2
            concurrency = max(1, concurrency // 2)
            continue
        if failure == "oom" and not smoke and tensor_parallel_size == 1:
            tensor_parallel_size = 2
            concurrency = max(1, concurrency // 2)
            resume = False
            resume_run_dir = None
            continue
        if failure == "oom" and not smoke and tensor_parallel_size == 2:
            tensor_parallel_size = 4
            concurrency = max(1, concurrency // 2)
            resume = False
            resume_run_dir = None
            continue
        if failure == "kv_cache":
            concurrency = max(1, concurrency // 2)
        if failure == "timeout":
            timeout *= 2
        if failure in {"api_retries_exhausted", "cuda_device_assert"}:
            concurrency = max(1, concurrency // 2)
            resume = False
            resume_run_dir = None
        elif not smoke and run_dir.joinpath("manifest.json").exists():
            resume = True
            resume_run_dir = run_dir
        if attempt >= 2:
            return {
                "status": "failed",
                "reason": failure or f"eval-agent exit code {code}",
                "run_dir": str(run_dir),
                "attempts": attempt,
                "num_concurrent": concurrency,
                "tensor_parallel_size": tensor_parallel_size,
                "kv_cache_peak_percent": kv_peak,
                "orchestration_log": str(log_path),
            }


def _state_path(root: Path, key: str) -> Path:
    return root / "status" / f"{key}.json"


def _write_job_state(root: Path, key: str, update: dict[str, Any]) -> None:
    with STATE_LOCK:
        state = read_json(_state_path(root, key), {}) or {}
        state.update(update)
        state["updated_at"] = utc_now()
        atomic_json(_state_path(root, key), state)


def _ensure_quantized(root: Path, spec: dict[str, Any], checkpoint_dir: Path) -> None:
    if (checkpoint_dir / "quantization_complete.json").is_file():
        return
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    key = job_key(
        spec["model"], spec["power"], spec["objective"], spec["observer_variant"]
    )
    _write_job_state(root, key, {"status": "quantizing", "started_at": utc_now()})
    code = _run_quant_reserved(root, spec, checkpoint_dir)
    if code != 0 or not (checkpoint_dir / "quantization_complete.json").is_file():
        _write_job_state(
            root, key, {"status": "quantization_failed", "exit_code": code}
        )
        raise RuntimeError(f"Quantization failed for {key}; see quantization_logs")
    _write_job_state(
        root,
        key,
        {
            "status": "quantized",
            "quantization": read_json(checkpoint_dir / "quantization_complete.json"),
        },
    )


def _estimated_checkpoint_bytes(
    root: Path, model_key: str, observer_variant: str
) -> int:
    baseline = job_key(model_key, 2.0, "importance_error", observer_variant)
    marker = (
        read_json(root / "checkpoints" / baseline / "quantization_complete.json", {})
        or {}
    )
    checkpoint_bytes = marker.get("checkpoint_bytes")
    if checkpoint_bytes is None:
        checkpoint_gib = {
            "qwen25_7b": 8,
            "qwen38_27b": 24,
            "qwen3_30b_a3b": 24,
        }[model_key]
        checkpoint_bytes = checkpoint_gib * 1024**3
    # Leave 15% headroom for variant-specific tensors and temporary writes.
    return int(checkpoint_bytes * 1.15)


def _reserve_checkpoint_space(root: Path, model_key: str, observer_variant: str) -> int:
    global DISK_RESERVED_BYTES
    reservation = _estimated_checkpoint_bytes(root, model_key, observer_variant)
    with DISK_RESERVATION_LOCK:
        while True:
            free_bytes = shutil.disk_usage(root).free
            if free_bytes - DISK_RESERVED_BYTES - reservation >= MIN_FREE_DISK_BYTES:
                DISK_RESERVED_BYTES += reservation
                return reservation
            print(
                f"Waiting for disk space before quantizing {model_key}: "
                f"free={free_bytes / 1024**3:.1f}GiB, "
                f"reserved={DISK_RESERVED_BYTES / 1024**3:.1f}GiB, "
                f"request={reservation / 1024**3:.1f}GiB",
                flush=True,
            )
            DISK_RESERVATION_LOCK.wait(timeout=60)


def _release_checkpoint_space(reservation: int = 0) -> None:
    global DISK_RESERVED_BYTES
    with DISK_RESERVATION_LOCK:
        DISK_RESERVED_BYTES = max(0, DISK_RESERVED_BYTES - reservation)
        DISK_RESERVATION_LOCK.notify_all()


def run_smoke_job(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    key = job_key(
        spec["model"], spec["power"], spec["objective"], spec["observer_variant"]
    )
    prior_state = read_json(_state_path(root, key), {}) or {}
    if prior_state.get("smoke_status") == "completed":
        return {
            "job": key,
            "status": "completed",
            "smoke": prior_state.get("smoke", {}),
            "skipped": True,
        }
    checkpoint_dir = root / "checkpoints" / key
    try:
        _ensure_quantized(root, spec, checkpoint_dir)
        results = {}
        for category in TASKS:
            results[category] = evaluate_category(
                root,
                spec,
                checkpoint_dir,
                category,
                smoke=True,
                concurrency=INITIAL_CONCURRENCY[category],
                task_names=SMOKE_TASKS[category],
            )
            if results[category]["status"] != "completed":
                raise RuntimeError(
                    f"Smoke failed for {key}/{category}: {results[category]}"
                )
        _write_job_state(
            root,
            key,
            {"smoke_status": "completed", "smoke": results},
        )
        return {"job": key, "status": "completed", "smoke": results}
    except Exception as exc:
        _write_job_state(
            root,
            key,
            {"smoke_status": "failed", "error": repr(exc)},
        )
        return {"job": key, "status": "failed", "error": repr(exc)}


def run_prequantize_job(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Quantize one next-round config while the baseline smoke suite runs."""
    key = job_key(
        spec["model"], spec["power"], spec["objective"], spec["observer_variant"]
    )
    checkpoint_dir = root / "checkpoints" / key
    try:
        if not (checkpoint_dir / "quantization_complete.json").is_file():
            reservation = _reserve_checkpoint_space(
                root, spec["model"], spec["observer_variant"]
            )
            try:
                _ensure_quantized(root, spec, checkpoint_dir)
            finally:
                _release_checkpoint_space(reservation)
        return {"job": key, "status": "completed", "prequantized": True}
    except Exception as exc:
        return {"job": key, "status": "failed", "error": repr(exc)}


def _concurrency_for(root: Path, model_key: str, category: str) -> tuple[int, int]:
    state = read_json(root / "concurrency.json", {}) or {}
    entry = state.get(model_key, {}).get(category)
    if not entry or entry.get("status") != "completed":
        raise RuntimeError(f"No passing smoke calibration for {model_key}/{category}")
    return entry["num_concurrent"], entry["tensor_parallel_size"]


def run_full_job(
    root: Path, spec: dict[str, Any], seed: int, seed_rounds: list[int]
) -> dict[str, Any]:
    key = job_key(
        spec["model"], spec["power"], spec["objective"], spec["observer_variant"]
    )
    checkpoint_dir = root / "checkpoints" / key
    state = read_json(_state_path(root, key), {}) or {}
    seed_key = str(seed)
    seed_status = state.get("full_seed_status", {})
    if seed_status.get(seed_key) == "completed":
        return {"job": key, "seed": seed, "status": "completed", "skipped": True}
    try:
        completion_marker = checkpoint_dir / "quantization_complete.json"
        if completion_marker.is_file():
            _ensure_quantized(root, spec, checkpoint_dir)
        else:
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
            reservation = _reserve_checkpoint_space(
                root, spec["model"], spec["observer_variant"]
            )
            try:
                _ensure_quantized(root, spec, checkpoint_dir)
            finally:
                _release_checkpoint_space(reservation)
        results = state.get("full_evaluations", {})
        seed_results = results.get(seed_key, {})
        for category in TASKS:
            if seed_results.get(category, {}).get("status") == "completed":
                continue
            concurrency, tensor_parallel_size = _concurrency_for(
                root, spec["model"], category
            )
            category_result = evaluate_category(
                root,
                spec,
                checkpoint_dir,
                category,
                smoke=False,
                concurrency=concurrency,
                tensor_parallel_size=tensor_parallel_size,
                seed=seed,
            )
            seed_results[category] = category_result
            results[seed_key] = seed_results
            _write_job_state(root, key, {"full_evaluations": results})
            if category_result["status"] != "completed":
                raise RuntimeError(
                    f"Full evaluation failed for {key}/{category}: {category_result}"
                )
        current_state = read_json(_state_path(root, key), {}) or {}
        seed_status = current_state.get("full_seed_status", {})
        seed_status[seed_key] = "completed"
        accepted_seed_statuses = {"completed", "partial_mmlu_cancelled"}
        round_statuses = [
            seed_status.get(str(round_seed)) for round_seed in seed_rounds
        ]
        all_seeds_complete = all(
            status in accepted_seed_statuses for status in round_statuses
        )
        overall_status = (
            "partial"
            if all_seeds_complete and "partial_mmlu_cancelled" in round_statuses
            else "completed"
            if all_seeds_complete
            else "in_progress"
        )
        _write_job_state(
            root,
            key,
            {
                "full_status": overall_status,
                "full_seed_status": seed_status,
                "full_evaluations": results,
            },
        )
        remove_checkpoint = spec["model"] not in RETAIN_CHECKPOINT_MODELS_ACROSS_SEEDS
        if remove_checkpoint:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            _release_checkpoint_space()
        return {
            "job": key,
            "seed": seed,
            "status": "completed",
            "all_seeds_complete": all_seeds_complete,
            "evaluations": seed_results,
        }
    except Exception as exc:
        if (
            checkpoint_dir.exists()
            and not (checkpoint_dir / "quantization_complete.json").is_file()
        ):
            shutil.rmtree(checkpoint_dir)
            _release_checkpoint_space()
        _write_job_state(
            root,
            key,
            {"full_status": "failed", "error": repr(exc)},
        )
        return {"job": key, "seed": seed, "status": "failed", "error": repr(exc)}


def _full_seed_rounds(selected_seeds: list[int] | None = None) -> list[int]:
    import yaml

    registry_path = eval_registry_path()
    registry = yaml.safe_load(registry_path.read_text())
    registry_seeds = registry.get("seeds", [])
    seeds = registry_seeds[:3] if selected_seeds is None else selected_seeds
    categories = registry.get("categories", {})
    task_counts = []
    for category, task_names in SMOKE_TASKS.items():
        configured = {task["name"]: task for task in categories[category]["tasks"]}
        task_counts.extend(
            configured[task_name]["n_repetitions"] for task_name in task_names
        )
    if (
        not seeds
        or any(seed not in registry_seeds for seed in seeds)
        or not task_counts
        or min(task_counts) < len(seeds)
    ):
        raise RuntimeError(
            "The selected seeds must exist in the registry and the requested tasks "
            "must have enough repetitions; "
            f"registry seeds; found seeds={seeds}, repetitions={task_counts}"
        )
    return seeds


def _active_quantization_processes() -> list[str]:
    script_path = str(Path(__file__).resolve())
    active = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return active
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            command = (process_dir / "cmdline").read_bytes().decode(errors="replace")
            command = command.replace("\0", " ").strip()
            process_name = (process_dir / "comm").read_text().strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (
            process_name.startswith("python")
            and f"{script_path} quantize " in f" {command} "
        ):
            active.append(f"pid={process_dir.name} {command}")
    return active


def _write_concurrency(
    root: Path, _smoke_results: list[dict[str, Any]]
) -> dict[str, Any]:
    calibration: dict[str, Any] = {}
    for spec in baseline_specs():
        key = job_key(
            spec["model"], spec["power"], spec["objective"], spec["observer_variant"]
        )
        state = read_json(_state_path(root, key), {}) or {}
        if state.get("smoke_status") == "completed":
            model_calibration = calibration.setdefault(spec["model"], {})
            for category, result in state.get("smoke", {}).items():
                current = model_calibration.get(category)
                if current is None:
                    model_calibration[category] = result
                else:
                    current["num_concurrent"] = min(
                        current["num_concurrent"], result["num_concurrent"]
                    )
                    current["tensor_parallel_size"] = max(
                        current["tensor_parallel_size"],
                        result["tensor_parallel_size"],
                    )
                    current["kv_cache_peak_percent"] = max(
                        current["kv_cache_peak_percent"],
                        result["kv_cache_peak_percent"],
                    )
    atomic_json(root / "concurrency.json", calibration)
    return calibration


def _recover_smoke_state(root: Path, spec: dict[str, Any]) -> bool:
    """Combine successful smoke tasks across retry directories for one baseline."""
    key = job_key(
        spec["model"], spec["power"], spec["objective"], spec["observer_variant"]
    )
    state = read_json(_state_path(root, key), {}) or {}
    if state.get("smoke_status") == "completed":
        return True

    smoke_seed = _full_seed_rounds()[0]
    recovered: dict[str, Any] = {}
    for category, task_names in SMOKE_TASKS.items():
        completed_by_task: dict[str, list[dict[str, Any]]] = {
            task_name: [] for task_name in task_names
        }
        category_root = root / "smoke" / key / category
        for events_path in category_root.glob("**/events.jsonl"):
            run_dir = events_path.parent
            manifest = read_json(run_dir / "manifest.json", {}) or {}
            if not manifest.get("smoke_only") or manifest.get("category") != category:
                continue
            kv_peak = _kv_cache_peak(run_dir / "logs" / "vllm_server.log")
            if kv_peak is None or kv_peak > 85.0:
                continue
            tensor_parallel_match = re.search(
                r"--tensor-parallel-size\s+(\d+)", manifest.get("server_cmd", "")
            )
            if tensor_parallel_match is None:
                continue
            try:
                events = [
                    json.loads(line)
                    for line in events_path.read_text().splitlines()
                    if line.strip()
                ]
            except json.JSONDecodeError:
                continue
            for event in events:
                task_name = event.get("task")
                if (
                    event.get("event") == "eval_completed"
                    and event.get("seed") == smoke_seed
                    and task_name in completed_by_task
                ):
                    completed_by_task[task_name].append(
                        {
                            "run_dir": str(run_dir),
                            "num_concurrent": manifest["num_concurrent"],
                            "tensor_parallel_size": int(tensor_parallel_match.group(1)),
                            "kv_cache_peak_percent": kv_peak,
                        }
                    )
        if any(not candidates for candidates in completed_by_task.values()):
            return False
        task_evidence = [candidates[-1] for candidates in completed_by_task.values()]
        run_dirs = sorted({evidence["run_dir"] for evidence in task_evidence})
        recovered[category] = {
            "status": "completed",
            "run_dir": run_dirs[-1],
            "run_dirs": run_dirs,
            "attempts": len(run_dirs),
            "num_concurrent": min(
                evidence["num_concurrent"] for evidence in task_evidence
            ),
            "tensor_parallel_size": max(
                evidence["tensor_parallel_size"] for evidence in task_evidence
            ),
            "kv_cache_peak_percent": max(
                evidence["kv_cache_peak_percent"] for evidence in task_evidence
            ),
            "recovered_from_completed_tasks": True,
        }

    _write_job_state(
        root,
        key,
        {"smoke_status": "completed", "smoke": recovered, "error": None},
    )
    return True


def run_smoke_phase(root: Path, workers: int) -> list[dict[str, Any]]:
    prepare(root)
    warm_specs = [
        {
            "model": model_key,
            "observer_variant": "expanded",
            "power": 2.2,
            "objective": "importance_error",
        }
        for model_key in MODELS
    ]
    print(
        f"Smoke phase: {len(baseline_specs())} baseline checks plus "
        f"{len(warm_specs)} prequantizations; workers={workers}",
        flush=True,
    )
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_smoke_job, root, spec): f"smoke:{spec['model']}"
            for spec in baseline_specs()
        }
        futures.update(
            {
                pool.submit(run_prequantize_job, root, spec): (
                    f"prequantize:{spec['model']}"
                )
                for spec in warm_specs
            }
        )
        results.extend(_collect_with_monitor(root, "smoke", None, futures))
    calibration = _write_concurrency(root, results)
    if len(calibration) != len(MODELS):
        raise RuntimeError(
            "At least one model did not pass smoke and concurrency calibration"
        )
    print(f"Smoke calibration complete for {len(calibration)} models", flush=True)
    return results


def _record_gpu_monitoring(root: Path, phase: str, seed: int | None) -> None:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    chg = subprocess.run(["chg", "status"], capture_output=True, text=True)
    record = {
        "time": utc_now(),
        "phase": phase,
        "seed": seed,
        "nvidia_smi": gpu.stdout if gpu.returncode == 0 else gpu.stderr,
        "chg_status": chg.stdout if chg.returncode == 0 else chg.stderr,
    }
    with (root / "monitoring.jsonl").open("a") as out:
        out.write(json.dumps(record, sort_keys=True) + "\n")
    print(
        f"GPU MONITOR phase={phase} seed={seed} "
        f"nvidia-smi={record['nvidia_smi'].strip().replace(chr(10), '; ')}",
        flush=True,
    )


def _collect_with_monitor(
    root: Path,
    phase: str,
    seed: int | None,
    futures: dict[concurrent.futures.Future, str],
) -> list[dict[str, Any]]:
    pending = set(futures)
    results: list[dict[str, Any]] = []
    while pending:
        done, pending = concurrent.futures.wait(
            pending,
            timeout=MONITOR_INTERVAL_SECONDS,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        if not done:
            _record_gpu_monitoring(root, phase, seed)
            continue
        for future in done:
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "job": futures[future],
                    "status": "failed",
                    "error": repr(exc),
                }
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
        _record_gpu_monitoring(root, phase, seed)
        if phase == "full":
            write_summary(root)
    return results


def run_full_phase(
    root: Path, workers: int, selected_seeds: list[int] | None = None
) -> list[dict[str, Any]]:
    prepare(root)
    # Verify smoke coverage before any full benchmark starts.
    for spec in baseline_specs():
        _recover_smoke_state(root, spec)
        key = job_key(
            spec["model"], spec["power"], spec["objective"], spec["observer_variant"]
        )
        state = read_json(_state_path(root, key), {}) or {}
        if state.get("smoke_status") != "completed":
            raise RuntimeError(f"Smoke must pass before full evaluation: {key}")
    calibration = _write_concurrency(root, [])
    if len(calibration) != len(MODELS):
        raise RuntimeError(
            f"Expected smoke calibration for all {len(MODELS)} models; "
            f"found {len(calibration)}"
        )
    jobs = job_specs()
    active_quantizations = []
    for spec in jobs:
        key = job_key(
            spec["model"], spec["power"], spec["objective"], spec["observer_variant"]
        )
        state = read_json(_state_path(root, key), {}) or {}
        if state.get("status") == "quantizing":
            active_quantizations.append(key)
    active_quantizations.extend(_active_quantization_processes())
    if active_quantizations:
        raise RuntimeError(
            "Wait for currently running quantizations before starting full evaluation: "
            + ", ".join(active_quantizations)
        )

    seed_rounds = _full_seed_rounds(selected_seeds)
    print(
        f"Full phase: {len(jobs)} configs; {len(seed_rounds)} seed-major rounds; "
        f"workers={workers}",
        flush=True,
    )
    results = []
    for seed in seed_rounds:
        print(f"Starting seed round {seed}", flush=True)
        round_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_full_job, root, spec, seed, seed_rounds): spec["key"]
                for spec in jobs
            }
            round_results = _collect_with_monitor(root, "full", seed, futures)
        results.extend(round_results)
        write_summary(root)
        failures = [
            result for result in round_results if result["status"] != "completed"
        ]
        if failures:
            raise RuntimeError(
                f"Seed {seed} did not complete across all configs; later seed rounds "
                f"are held until this round is repaired. Failed jobs: {failures}"
            )
    write_summary(root)
    return results


def write_summary(
    root: Path, specs: list[dict[str, Any]] | None = None
) -> tuple[Path, Path]:
    accumulated: dict[tuple[str, str, float, str, str, str, str], list[float]] = {}
    for spec in job_specs() if specs is None else specs:
        key = job_key(
            spec["model"], spec["power"], spec["objective"], spec["observer_variant"]
        )
        state = read_json(_state_path(root, key), {}) or {}
        for seed_evaluations in state.get("full_evaluations", {}).values():
            for category, evaluation in seed_evaluations.items():
                if evaluation.get("status") != "completed":
                    continue
                run_dir = Path(evaluation.get("run_dir", ""))
                summary = read_json(run_dir / "summary_data.json", {}) or {}
                for task, metrics in summary.get("results", {}).items():
                    for metric, scores in metrics.items():
                        mean = scores.get("mean")
                        if mean is None:
                            continue
                        result_key = (
                            spec["model"],
                            spec["observer_variant"],
                            spec["power"],
                            spec["objective"],
                            category,
                            task,
                            metric,
                        )
                        accumulated.setdefault(result_key, []).append(float(mean))
    rows: list[dict[str, Any]] = []
    for result_key, seed_scores in accumulated.items():
        model_key, observer_variant, power, objective, category, task, metric = (
            result_key
        )
        rows.append(
            {
                "model": MODELS[model_key]["id"],
                "model_key": model_key,
                "observer_variant": observer_variant,
                "power": power,
                "objective": objective,
                "category": category,
                "task": task,
                "metric": metric,
                "mean": sum(seed_scores) / len(seed_scores),
                "n_seeds": len(seed_scores),
            }
        )
    csv_path = root / "ablation_summary.csv"
    headers = [
        "model",
        "model_key",
        "observer_variant",
        "power",
        "objective",
        "category",
        "task",
        "metric",
        "mean",
        "n_seeds",
    ]
    import csv

    with csv_path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    md_path = root / "ablation_summary.md"
    with md_path.open("w") as out:
        out.write(f"# {SCHEME_NAME} imatrix observer ablation results\n\n")
        out.write(
            "Scores are means across the completed eval-agent registry seed rounds. "
            "Raw task results and logs are in `evaluations/`.\n\n"
        )
        if not rows:
            out.write("No completed benchmark scores yet.\n")
        else:
            out.write(
                "| Model | Observer | Power | Objective | Task | Metric | "
                "Mean | Seeds |\n"
                "|---|---|---:|---|---|---|---:|---:|\n"
            )
            for row in rows:
                out.write(
                    f"| {row['model']} | {row['observer_variant']} | "
                    f"{row['power']:g} | {row['objective']} | "
                    f"{row['task']} | {row['metric']} | {row['mean']:.6f} | "
                    f"{row['n_seeds']} |\n"
                )
    return csv_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--root", type=Path, default=DEFAULT_RUN_ROOT)
    quant_parser = subparsers.add_parser("quantize")
    quant_parser.add_argument("--root", type=Path, default=DEFAULT_RUN_ROOT)
    quant_parser.add_argument("--model", choices=MODELS, required=True)
    quant_parser.add_argument("--power", type=float, choices=POWERS, required=True)
    quant_parser.add_argument("--objective", choices=OBJECTIVES, required=True)
    quant_parser.add_argument(
        "--observer-variant", choices=OBSERVER_SETTINGS, default="expanded"
    )
    quant_parser.add_argument("--checkpoint-dir", type=Path)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--root", type=Path, default=DEFAULT_RUN_ROOT)
    smoke_parser.add_argument("--workers", type=int, default=4)
    full_parser = subparsers.add_parser("full")
    full_parser.add_argument("--root", type=Path, default=DEFAULT_RUN_ROOT)
    full_parser.add_argument("--workers", type=int, default=8)
    full_parser.add_argument(
        "--seeds",
        help=(
            "Comma-separated eval-agent seeds; defaults to the first three "
            "registry seeds"
        ),
    )
    summary_parser = subparsers.add_parser("summary")
    summary_parser.add_argument("--root", type=Path, default=DEFAULT_RUN_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        manifest = prepare(args.root)
        print(json.dumps(manifest, indent=2))
        return 0
    if args.command == "quantize":
        quantize_one(
            args.root,
            args.model,
            args.power,
            args.objective,
            args.observer_variant,
            args.checkpoint_dir,
        )
        return 0
    if args.command == "smoke":
        results = run_smoke_phase(args.root, args.workers)
        return 0 if all(item["status"] == "completed" for item in results) else 1
    if args.command == "full":
        seeds = (
            [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
            if args.seeds
            else None
        )
        results = run_full_phase(args.root, args.workers, seeds)
        failed = [item for item in results if item["status"] != "completed"]
        print(
            "Full run finished: "
            f"{len(results) - len(failed)}/{len(results)} jobs completed"
        )
        return 0 if not failed else 1
    if args.command == "summary":
        csv_path, md_path = write_summary(args.root)
        print(csv_path)
        print(md_path)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
