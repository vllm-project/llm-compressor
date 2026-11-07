#!/usr/bin/env python3
"""Run a model-free MXFP4xFP8 MSE power and magnitude-normalization ablation.

Model-free quantization is performed once per model/power/normalization tuple.
Evaluation is seed-major: every configuration receives seed 1234 before any
configuration begins seed 2345, then seed 3456. Quantization and evaluation
jobs reserve GPUs through ``chg``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import getpass
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEME_NAME = "MXFP4XFP8"
SCHEME_SLUG = SCHEME_NAME.lower()
DEFAULT_RUN_ROOT = Path(
    os.environ.get(
        "MXFP4XFP8_MSE_RUN_ROOT", ROOT / "runs/mxfp4xfp8_mse_model_free_ablation"
    )
)
HF_CACHE = Path(os.environ.get("HF_HOME", ROOT / "hf_cache"))
RHVENV_PYTHON = Path(os.environ.get("RHVENV_PYTHON", ROOT / ".venvs/rhdev/bin/python"))
BASE_RUNNER = ROOT / "examples/imatrix/run_mxfp4xfp8_imatrix_ablation.py"
POWERS = (2.0, 2.2, 2.4, 2.6, 2.8, 3.0)
OBJECTIVES = ("magnitude_off", "magnitude_on")
SEEDS = (1234, 2345, 3456)
OBSERVER_VARIANT = "exponential_1_to_1_64"
OBSERVER_KWARGS = {
    "grid_type": "exponential",
    "grid": 8,
    "maxshrink": 63 / 64,
    "expand": 1.0,
    "patience": 1000,
}
TASKS = {
    "instruct": ("gsm8k_platinum_cot_llama", "mmlu_cot_llama", "math_500"),
    "reasoning": ("gpqa_diamond",),
}
SMOKE_TASKS = {
    "instruct": ("gsm8k_platinum_cot_llama", "math_500"),
    "reasoning": TASKS["reasoning"],
}
MAX_WORKERS = 6
MAX_ATTEMPTS = 2
EVALUATION_MODEL_ALIAS_VERSION = 2
MONITOR_INTERVAL_SECONDS = 600
STATE_LOCK = threading.Lock()
SUMMARY_LOCK = threading.Lock()
MANIFEST_CACHE: dict[str, Any] | None = None

for _cache_var in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
    os.environ.setdefault(_cache_var, str(HF_CACHE))


def load_base_runner():
    """Load the existing eval-agent command builder without modifying it."""
    spec = importlib.util.spec_from_file_location(
        "mxfp4_imatrix_eval_helpers", BASE_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load shared eval helpers from {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    except (OSError, json.JSONDecodeError):
        return default


def model_specs(base: Any) -> dict[str, dict[str, Any]]:
    keys = ("qwen25_7b", "qwen38_27b", "qwen3_30b_a3b")
    excluded = {
        key.strip()
        for key in os.environ.get("MXFP4_ABLATION_EXCLUDE_MODELS", "").split(",")
        if key.strip()
    }
    keys = tuple(key for key in keys if key not in excluded)
    models = {key: dict(base.MODELS[key]) for key in keys}
    # Qwen3.8 has a Mamba cache with fewer than vLLM's default 1024 decode
    # blocks at this context length. Bound the scheduler to the cache capacity.
    models["qwen38_27b"]["server_args"] = [
        *models["qwen38_27b"].get("server_args", []),
        "--max-num-seqs",
        "900",
    ]
    # Model-free matching treats every 2D ``*.weight`` tensor as a Linear
    # target. Keep modules that are not Linear layers dense, matching the
    # model-loaded MXFP4XFP8 ablation's target-class behavior.
    for model_key in ("qwen25_7b", "qwen3_30b_a3b"):
        models[model_key]["ignore"] = [
            *models[model_key]["ignore"],
            r"re:.*embed_tokens$",
        ]
    # The model-loaded Linear target excludes this Conv1d weight by class. The
    # model-free matcher only has tensor names, so make that exclusion explicit
    # to preserve the same set of quantized linear weights.
    models["qwen38_27b"]["ignore"] = [
        *models["qwen38_27b"]["ignore"],
        r"re:.*linear_attn\.conv1d$",
    ]
    return models


def powerslug(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def job_key(model_key: str, power: float, objective: str) -> str:
    return f"{model_key}__model_free_mse_exp1to1_64__p{powerslug(power)}__{objective}"


def job_specs(models: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "model": model_key,
            "power": power,
            "objective": objective,
            "observer_variant": OBSERVER_VARIANT,
            "key": job_key(model_key, power, objective),
        }
        for model_key in models
        for power in POWERS
        for objective in OBJECTIVES
    ]


def prepare(root: Path) -> dict[str, Any]:
    """Create or validate the pinned MXFP4xFP8 MSE run manifest."""
    global MANIFEST_CACHE
    root.mkdir(parents=True, exist_ok=True)
    existing = read_json(root / "experiment_manifest.json")
    if existing:
        if existing.get("experiment_type") != f"{SCHEME_SLUG}_mse_model_free":
            raise RuntimeError(f"Run directory belongs to another experiment: {root}")
        base = load_base_runner()
        calibration = existing.setdefault("calibration", {})
        calibration["input_activations"] = base.input_activation_config()
        if SCHEME_NAME == "MXFP4XFP8":
            calibration["note"] = (
                "weight_only describes the offline model-free PTQ search; the "
                "MXFP4XFP8 preset still declares dynamic FP8 input activations "
                "for inference."
            )
        existing.setdefault("evaluation", {}).update(base.evaluation_backend_metadata())
        models = model_specs(base)
        old_keys = {item["key"] for item in existing.get("jobs", [])}
        current_specs = job_specs(models)
        new_keys = {item["key"] for item in current_specs}
        excluded_models = {
            key.strip()
            for key in os.environ.get("MXFP4_ABLATION_EXCLUDE_MODELS", "").split(",")
            if key.strip()
        }
        superseded = [
            {
                "key": key,
                "reason": (
                    "model excluded from this run via MXFP4_ABLATION_EXCLUDE_MODELS"
                    if key.split("__", 1)[0] in excluded_models
                    else "model-free preflight rejected a non-2D Conv1d tensor"
                ),
            }
            for key in sorted(old_keys - new_keys)
        ]
        if superseded:
            existing.setdefault("superseded_jobs", []).extend(
                item
                for item in superseded
                if item["key"]
                not in {old["key"] for old in existing.get("superseded_jobs", [])}
            )
        for model_key, spec in models.items():
            if model_key in existing.get("models", {}):
                model_entry = existing["models"][model_key]
                ignore_record = (
                    read_json(
                        root / "model_free_non_2d_ignores" / f"{model_key}.json", {}
                    )
                    or {}
                )
                scanned_ignores = (
                    ignore_record.get("ignore", [])
                    if ignore_record.get("revision") == model_entry.get("revision")
                    else []
                )
                extra_ignores = list(
                    dict.fromkeys(
                        [
                            *model_entry.get("model_free_non_2d_ignores", []),
                            *scanned_ignores,
                        ]
                    )
                )
                model_entry["model_free_non_2d_ignores"] = extra_ignores
                model_entry["ignore"] = list(
                    dict.fromkeys([*spec["ignore"], *extra_ignores])
                )
        existing["jobs"] = current_specs
        existing["updated_at"] = utc_now()
        current_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if existing.get("runner_sha256") != current_hash:
            existing["runner_sha256"] = current_hash
            existing["updated_at"] = utc_now()
            atomic_json(root / "experiment_manifest.json", existing)
        MANIFEST_CACHE = existing
        atomic_json(root / "experiment_manifest.json", existing)
        return existing

    base = load_base_runner()
    models = model_specs(base)
    model_entries: dict[str, dict[str, Any]] = {}
    from transformers import AutoConfig

    model_revisions = {
        spec["id"]: base.PINNED_MODEL_REVISIONS[spec["id"]] for spec in models.values()
    }
    templates_dir = root / "chat_templates"
    templates_dir.mkdir(parents=True, exist_ok=True)

    for model_key, spec in models.items():
        model_id = spec["id"]
        revision = model_revisions[model_id]
        config = AutoConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=True
        )
        template_path = templates_dir / f"{model_key}.jinja"
        template_path.write_text(base._chat_template(model_key, revision))
        model_entries[model_key] = {
            "id": model_id,
            "revision": revision,
            "model_type": config.model_type,
            "architectures": getattr(config, "architectures", None),
            "max_position_embeddings": getattr(config, "max_position_embeddings", None),
            "generation_params": base._generation_params(model_id, revision),
            "chat_template": str(template_path),
            "ignore": list(spec["ignore"]),
            "reasoning_parser": spec["reasoning_parser"],
        }

    registry_path = base.eval_registry_path()
    implementation_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    repo_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "experiment_type": f"{SCHEME_SLUG}_mse_model_free",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "repo": str(ROOT),
        "repo_commit": repo_commit,
        "runner_sha256": implementation_hash,
        "model_revisions": model_revisions,
        "models": model_entries,
        "calibration": {
            "scheme": SCHEME_NAME,
            "weight_only": True,
            "input_activations": base.input_activation_config(),
            "note": (
                "weight_only describes the offline model-free PTQ search; the "
                "MXFP4XFP8 preset still declares dynamic FP8 input activations "
                "for inference."
            )
            if SCHEME_NAME == "MXFP4XFP8"
            else "The model-free PTQ search operates on the quantized weights.",
            "method": "model_free_ptq",
            "group_size": 32,
            "scale_dtype": "torch.uint8 (E8M0 MX scales)",
            "zero_point_dtype": "torch.uint8",
            "observer": "nvfp4_expanded_mse",
            "powers": list(POWERS),
            "magnitude_normalization": [False, True],
            "observer_kwargs": OBSERVER_KWARGS,
            "candidate_multipliers": [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625],
            "candidate_count": 7,
            "quantization_seed": None,
        },
        "evaluation": {
            "tasks": {key: list(value) for key, value in TASKS.items()},
            "seeds": list(SEEDS),
            "registry": str(registry_path),
            "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            "vllm_environment": str(base.VLLM_ENV),
            "lm_eval_environment": str(base.VLLM_ENV),
            "lighteval_environment": str(base.LIGHTEVAL_ENV),
            **base.evaluation_backend_metadata(),
        },
        "jobs": job_specs(models),
    }
    atomic_json(root / "experiment_manifest.json", manifest)
    MANIFEST_CACHE = manifest
    return manifest


def checkpoint_dir(root: Path, spec: dict[str, Any]) -> Path:
    return root / "checkpoints" / spec["key"]


def state_path(root: Path, key: str) -> Path:
    return root / "status" / f"{key}.json"


def write_state(root: Path, key: str, update: dict[str, Any]) -> dict[str, Any]:
    with STATE_LOCK:
        path = state_path(root, key)
        state = read_json(path, {}) or {}
        state.update(update)
        state["updated_at"] = utc_now()
        atomic_json(path, state)
        return state


def magnitude_enabled(spec: dict[str, Any]) -> bool:
    return spec["objective"] == "magnitude_on"


def evaluate_with_budget(
    root: Path,
    base: Any,
    spec: dict[str, Any],
    category: str,
    *,
    smoke: bool,
    concurrency: int,
    tensor_parallel_size: int,
    seed: int | None = None,
    task_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Use at most two evaluator attempts per task, including after restart."""
    state = read_json(state_path(root, spec["key"]), {}) or {}
    attempt_key = (
        f"smoke:model_alias_v2:{category}"
        if smoke
        else f"seed:{seed}:{category}:model_alias_v2"
    )
    attempts_by_task = state.get("evaluation_attempts", {})
    used = int(attempts_by_task.get(attempt_key, 0))
    saved = None
    if state.get("evaluation_model_alias_version") == EVALUATION_MODEL_ALIAS_VERSION:
        saved = (
            state.get("smoke", {}).get("categories", {}).get(category)
            if smoke
            else state.get("full_evaluations", {}).get(str(seed), {}).get(category)
        )
    if saved and saved.get("status") == "completed":
        return saved
    if used >= MAX_ATTEMPTS:
        return {
            "status": "failed",
            "reason": "two total evaluation attempts already used",
            "attempts": used,
        }

    model_key = spec["model"]
    checkpoint = checkpoint_dir(root, spec)
    timeout = base.TIMEOUTS[category]
    resume_dir: Path | None = None
    resume = False
    kv_peak = None
    latest_result: dict[str, Any] = {"status": "failed", "reason": "no attempt made"}
    consumed = 0
    smoke_session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    for offset in range(MAX_ATTEMPTS - used):
        attempt = used + offset + 1
        if smoke:
            run_dir = (
                root
                / "smoke"
                / spec["key"]
                / category
                / smoke_session
                / f"attempt_{attempt:02d}"
            )
        elif resume and resume_dir is not None:
            run_dir = resume_dir
        elif tensor_parallel_size > 1:
            run_dir = (
                root
                / "evaluations"
                / spec["key"]
                / category
                / f"seed_{seed}"
                / f"tp{tensor_parallel_size}_attempt_{attempt:02d}"
            )
        elif attempt > 1:
            run_dir = (
                root
                / "evaluations"
                / spec["key"]
                / category
                / f"seed_{seed}"
                / f"attempt_{attempt:02d}"
            )
        else:
            run_dir = root / "evaluations" / spec["key"] / category / f"seed_{seed}"
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        if (
            not smoke
            and run_dir.joinpath("manifest.json").is_file()
            and attempt == used + 1
        ):
            resume = True
            resume_dir = run_dir

        objective_index = base.OBJECTIVES.index(spec["objective"])
        variant_index = base.POWERS.index(spec["power"]) * len(base.OBJECTIVES)
        variant_index += objective_index
        port = (
            base.PORT_BASE + list(base.MODELS).index(model_key) * base.PORT_MODEL_STRIDE
        )
        port += variant_index * base.PORT_VARIANT_STRIDE
        port += (0 if category == "instruct" else base.PORT_REASONING_OFFSET) + attempt
        safe_job_key = spec["key"].replace("to1", "to_1")
        served_name = (
            f"{base.MODEL_NAME_PREFIX}-{safe_job_key}-{category}-"
            f"seed{seed if seed is not None else 'smoke'}"
        )
        if "o1" in served_name:
            raise RuntimeError(
                f"Served model name is misdetected as an o1 model: {served_name}"
            )
        command = base._eval_base_command(
            root,
            model_key,
            checkpoint,
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
            served_name=served_name,
        )
        log_key = f"{spec['key']}_{category}_{'smoke' if smoke else f'seed{seed}'}"
        log_path = root / "orchestration_logs" / f"{log_key}_attempt_{attempt:02d}.log"
        try:
            code = base._run_logged(command, log_path)
        except Exception as exc:
            code = 1
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as log:
                log.write(f"\n[{utc_now()}] runner exception: {exc!r}\n")
        server_log = run_dir / "logs" / "vllm_server.log"
        kv_peak = base._kv_cache_peak(server_log)
        failure = base._retryable_failure(log_path) or base._retryable_failure(
            server_log
        )
        eval_failure = base._evaluation_log_failure(run_dir)
        if (
            code == 0
            and eval_failure is None
            and (not smoke or (kv_peak is not None and kv_peak <= 85.0))
        ):
            latest_result = {
                "status": "completed",
                "run_dir": str(run_dir),
                "attempts": attempt - used,
                "num_concurrent": concurrency,
                "tensor_parallel_size": tensor_parallel_size,
                "kv_cache_peak_percent": kv_peak,
            }
            consumed = attempt - used
            break

        if smoke and code == 0 and kv_peak is not None and kv_peak > 85.0:
            concurrency = max(1, concurrency // 2)
            failure = "kv_cache"
        elif failure == "oom" and tensor_parallel_size == 1:
            tensor_parallel_size = 2
            concurrency = max(1, concurrency // 2)
            resume = False
            resume_dir = None
        elif failure == "kv_cache":
            concurrency = max(1, concurrency // 2)
        elif failure == "timeout":
            timeout *= 2
        if failure in {"api_retries_exhausted", "cuda_device_assert"}:
            concurrency = max(1, concurrency // 2)
            resume = False
            resume_dir = None
        elif not smoke and run_dir.joinpath("manifest.json").is_file() and not failure:
            resume = True
            resume_dir = run_dir
        consumed = attempt - used
        latest_result = {
            "status": "failed",
            "reason": failure or eval_failure or f"eval-agent exit status {code}",
            "run_dir": str(run_dir),
            "attempts": consumed,
            "num_concurrent": concurrency,
            "tensor_parallel_size": tensor_parallel_size,
            "kv_cache_peak_percent": kv_peak,
            "orchestration_log": str(log_path),
        }
        if failure in {"oom", "kv_cache"}:
            concurrency_state = read_json(root / "concurrency.json", {}) or {}
            previous = concurrency_state.get(model_key, {}).get(category, {})
            concurrency_state.setdefault(model_key, {}).setdefault(category, {}).update(
                {
                    "status": previous.get("status", "completed"),
                    "num_concurrent": concurrency,
                    "tensor_parallel_size": tensor_parallel_size,
                    "last_adjustment": failure,
                    "updated_at": utc_now(),
                }
            )
            atomic_json(root / "concurrency.json", concurrency_state)
    attempts_by_task[attempt_key] = used + consumed
    write_state(root, spec["key"], {"evaluation_attempts": attempts_by_task})
    return latest_result


def quantize_worker(root: Path, spec: dict[str, Any], model_snapshot: Path) -> None:
    """Quantize one checkpoint. This command runs inside the rhdev venv."""
    import torch
    from compressed_tensors.quantization import preset_name_to_scheme
    from huggingface_hub import snapshot_download

    from llmcompressor import model_free_ptq

    manifest = prepare(root)
    model_entry = manifest["models"][spec["model"]]
    snapshot_path = Path(model_snapshot)
    if not snapshot_path.is_dir():
        snapshot_path = Path(
            snapshot_download(
                repo_id=model_entry["id"],
                revision=model_entry["revision"],
                cache_dir=str(HF_CACHE),
            )
        )
    out_dir = checkpoint_dir(root, spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_record = {
        "job": spec["key"],
        "model": model_entry["id"],
        "model_revision": model_entry["revision"],
        "method": "model_free_ptq",
        "scheme": SCHEME_NAME,
        "observer": "nvfp4_expanded_mse",
        "observer_kwargs": {
            **OBSERVER_KWARGS,
            "norm": float(spec["power"]),
            "magnitude_normalization": magnitude_enabled(spec),
        },
        "candidate_multipliers": [2.0**-i for i in range(7)],
        "ignore": model_entry["ignore"],
    }
    atomic_json(out_dir / "ablation_config.json", config_record)

    scheme = preset_name_to_scheme(SCHEME_NAME, ["Linear"])
    scheme.weights.observer = "nvfp4_expanded_mse"
    scheme.weights.observer_kwargs = {
        **OBSERVER_KWARGS,
        "norm": float(spec["power"]),
        "magnitude_normalization": magnitude_enabled(spec),
    }

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model_free_ptq(
        model_stub=str(snapshot_path),
        save_directory=str(out_dir),
        scheme=scheme,
        ignore=model_entry["ignore"],
        max_workers=1,
        device="cuda:0",
    )
    elapsed = time.monotonic() - started
    safetensors = list(out_dir.glob("*.safetensors"))
    if not safetensors:
        raise RuntimeError(f"No quantized safetensors were written to {out_dir}")
    atomic_json(
        out_dir / "quantization_complete.json",
        {
            "job": spec["key"],
            "completed_at": utc_now(),
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "checkpoint_bytes": sum(path.stat().st_size for path in safetensors),
            "model_snapshot": str(snapshot_path),
        },
    )
    print(
        f"QUANTIZED job={spec['key']} elapsed={elapsed / 60:.1f}m " f"output={out_dir}",
        flush=True,
    )


def _run_logged(command: list[str], log_path: Path, cwd: Path = ROOT) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n[{utc_now()}] COMMAND {' '.join(command)}\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return proc.wait()


def ensure_quantized(root: Path, spec: dict[str, Any]) -> bool:
    out_dir = checkpoint_dir(root, spec)
    if (out_dir / "quantization_complete.json").is_file():
        return True
    prepare(root)
    snapshot_record = read_json(root / "model_snapshots" / f"{spec['model']}.json", {})
    if not snapshot_record or not Path(snapshot_record.get("path", "")).is_dir():
        raise RuntimeError(f"Model snapshot is not cached for {spec['model']}")

    state = read_json(state_path(root, spec["key"]), {}) or {}
    attempts = int(state.get("quantization_attempts", 0))
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        write_state(
            root,
            spec["key"],
            {"quantization_status": "running", "quantization_attempts": attempts},
        )
        if out_dir.exists():
            shutil.rmtree(out_dir)
        command = [
            "chg",
            "run",
            "--gpus",
            "1",
            "--",
            str(RHVENV_PYTHON),
            str(Path(__file__).resolve()),
            "quantize-worker",
            "--root",
            str(root),
            "--job-key",
            spec["key"],
            "--model-snapshot",
            snapshot_record["path"],
        ]
        log_path = (
            root / "quantization_logs" / f"{spec['key']}_attempt_{attempts:02d}.log"
        )
        code = _run_logged(command, log_path)
        if code == 0 and (out_dir / "quantization_complete.json").is_file():
            write_state(
                root,
                spec["key"],
                {
                    "quantization_status": "completed",
                    "quantization_attempts": attempts,
                    "quantization_error": None,
                },
            )
            return True
        state = {
            "quantization_status": "retrying" if attempts < MAX_ATTEMPTS else "failed",
            "quantization_attempts": attempts,
            "quantization_error": f"quantizer exit status {code}; log={log_path}",
        }
        write_state(root, spec["key"], state)
        print(
            f"QUANTIZATION FAILURE {spec['key']} attempt={attempts}: "
            f"{state['quantization_error']}",
            flush=True,
        )
    return False


def cache_model(root: Path, model_key: str) -> None:
    import re as regex

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    manifest = prepare(root)
    entry = manifest["models"][model_key]
    path = snapshot_download(
        repo_id=entry["id"],
        revision=entry["revision"],
        cache_dir=str(HF_CACHE),
    )
    non_2d_modules = set()
    for weight_file in Path(path).rglob("*.safetensors"):
        with safe_open(str(weight_file), framework="pt", device="cpu") as handle:
            for tensor_name in handle.keys():
                module_name, _, param_name = tensor_name.rpartition(".")
                if param_name != "weight":
                    continue
                if len(handle.get_slice(tensor_name).get_shape()) != 2:
                    non_2d_modules.add(r"re:^" + regex.escape(module_name) + r"$")
    ignore_patterns = sorted(non_2d_modules)
    atomic_json(
        root / "model_free_non_2d_ignores" / f"{model_key}.json",
        {
            "model": entry["id"],
            "revision": entry["revision"],
            "ignore": ignore_patterns,
        },
    )
    atomic_json(
        root / "model_snapshots" / f"{model_key}.json",
        {"model": entry["id"], "revision": entry["revision"], "path": path},
    )
    print(
        f"CACHED model={model_key} path={path} "
        f"non_2d_weight_modules={len(ignore_patterns)}",
        flush=True,
    )


def cache_models(root: Path, models: dict[str, dict[str, Any]], workers: int) -> None:
    manifest = prepare(root)
    pending = []
    for model_key in models:
        record = read_json(root / "model_snapshots" / f"{model_key}.json", {}) or {}
        ignore_record = (
            read_json(root / "model_free_non_2d_ignores" / f"{model_key}.json", {})
            or {}
        )
        if (
            record.get("revision") == manifest["models"][model_key]["revision"]
            and Path(record.get("path", "")).is_dir()
            and ignore_record.get("revision")
            == manifest["models"][model_key]["revision"]
        ):
            continue
        pending.append(model_key)
    if not pending:
        print("All pinned model revisions are already cached.", flush=True)
        return
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(workers, len(pending))
    ) as pool:
        futures = {
            pool.submit(
                _run_logged,
                [
                    str(RHVENV_PYTHON),
                    str(Path(__file__).resolve()),
                    "cache-model-worker",
                    "--root",
                    str(root),
                    "--model",
                    model_key,
                ],
                root / "cache_logs" / f"{model_key}.log",
            ): model_key
            for model_key in pending
        }
        failures = []
        for future in concurrent.futures.as_completed(futures):
            model_key = futures[future]
            code = future.result()
            if code != 0 or not read_json(
                root / "model_snapshots" / f"{model_key}.json"
            ):
                failures.append(model_key)
            print(
                f"MODEL CACHE {model_key}: {'ok' if code == 0 else 'failed'}",
                flush=True,
            )
        if failures:
            raise RuntimeError(f"Model cache preparation failed: {failures}")
    manifest = prepare(root)
    base = load_base_runner()
    models = model_specs(base)
    for model_key, model_spec in models.items():
        ignore_record = (
            read_json(root / "model_free_non_2d_ignores" / f"{model_key}.json", {})
            or {}
        )
        if ignore_record.get("revision") != manifest["models"][model_key]["revision"]:
            raise RuntimeError(f"Non-2D tensor scan missing for {model_key}")
        extra = ignore_record.get("ignore", [])
        manifest["models"][model_key]["model_free_non_2d_ignores"] = extra
        manifest["models"][model_key]["ignore"] = list(
            dict.fromkeys([*model_spec["ignore"], *extra])
        )
    manifest["updated_at"] = utc_now()
    atomic_json(root / "experiment_manifest.json", manifest)


def configure_eval_helper(
    base: Any, root: Path, manifest: dict[str, Any], models: dict[str, dict[str, Any]]
) -> None:
    base.MODELS = models
    base.TASKS = TASKS
    base.SMOKE_TASKS = SMOKE_TASKS
    base.POWERS = POWERS
    base.OBJECTIVES = OBJECTIVES
    base.MODEL_NAME_PREFIX = f"{SCHEME_SLUG}-mse"
    base.observer_variants = lambda model_key: (OBSERVER_VARIANT,)
    base.job_key = lambda model_key, power, objective, observer_variant: job_key(
        model_key, power, objective
    )
    base.prepare = lambda _root: manifest
    base.MODEL_RUN_DIR_SUFFIXES = {
        key: value
        for key, value in getattr(base, "MODEL_RUN_DIR_SUFFIXES", {}).items()
        if key in models
    }
    base.MODEL_MAX_LENGTHS = {
        key: value
        for key, value in getattr(base, "MODEL_MAX_LENGTHS", {}).items()
        if key in models
    }


def smoke_phase(
    root: Path, base: Any, models: dict[str, dict[str, Any]], workers: int
) -> None:
    concurrency_state = read_json(root / "concurrency.json", {}) or {}
    prepare(root)
    specs = [
        next(
            spec
            for spec in job_specs(models)
            if spec["model"] == model_key
            and spec["power"] == 2.0
            and spec["objective"] == "magnitude_off"
        )
        for model_key in models
    ]
    # Two additional quantizations occupy available GPU slots while model smoke
    # evaluations establish vLLM concurrency on each architecture.
    extra_quant_specs = [
        next(
            spec
            for spec in job_specs(models)
            if spec["model"] == model_key
            and spec["power"] == 2.0
            and spec["objective"] == "magnitude_on"
        )
        for model_key in ("qwen25_7b", "qwen38_27b")
    ]

    def smoke_model(spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        model_key = spec["model"]
        prior = read_json(state_path(root, spec["key"]), {}) or {}
        if (
            prior.get("evaluation_model_alias_version")
            == EVALUATION_MODEL_ALIAS_VERSION
            and prior.get("smoke", {}).get("status") == "completed"
            and all(
                concurrency_state.get(model_key, {}).get(category, {}).get("status")
                == "completed"
                for category in TASKS
            )
        ):
            print(f"SMOKE REUSE {model_key}: prior smoke passed", flush=True)
            return model_key, prior["smoke"]
        if not ensure_quantized(root, spec):
            return model_key, {
                "status": "failed",
                "reason": "quantization exhausted retries",
            }
        category_results = {}
        for category in TASKS:
            old = concurrency_state.get(model_key, {}).get(category, {})
            concurrency = int(
                old.get("num_concurrent", base.INITIAL_CONCURRENCY[category])
            )
            tp = int(old.get("tensor_parallel_size", 1))
            result = evaluate_with_budget(
                root,
                base,
                spec,
                category,
                smoke=True,
                concurrency=concurrency,
                tensor_parallel_size=tp,
                task_names=SMOKE_TASKS[category],
            )
            category_results[category] = result
            if result.get("status") == "completed":
                with STATE_LOCK:
                    concurrency_state.setdefault(model_key, {})[category] = {
                        "status": "completed",
                        "num_concurrent": result["num_concurrent"],
                        "tensor_parallel_size": result["tensor_parallel_size"],
                        "kv_cache_peak_percent": result["kv_cache_peak_percent"],
                        "smoke_run_dir": result["run_dir"],
                    }
                    atomic_json(root / "concurrency.json", concurrency_state)
        result = {
            "status": "completed"
            if all(
                item.get("status") == "completed" for item in category_results.values()
            )
            else "failed",
            "categories": category_results,
        }
        write_state(
            root,
            spec["key"],
            {
                "smoke": result,
                "evaluation_model_alias_version": EVALUATION_MODEL_ALIAS_VERSION,
            },
        )
        return model_key, result

    print(f"Smoke phase: {len(specs)} models, workers={workers}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(smoke_model, spec): f"smoke {spec['model']}" for spec in specs
        }
        futures.update(
            {
                pool.submit(ensure_quantized, root, spec): f"extra quant {spec['key']}"
                for spec in extra_quant_specs
            }
        )
        results = monitor_batch(root, "smoke", None, futures)
        failures = []
        for value in results:
            print(f"SMOKE RESULT {value}", flush=True)
            if isinstance(value, tuple) and value[1].get("status") != "completed":
                failures.append(value[0])
            elif value is False:
                failures.append("extra quantization exhausted retries")
            elif isinstance(value, dict) and value.get("status") != "completed":
                failures.append(
                    value.get("job", value.get("error", "unknown smoke failure"))
                )
    if failures:
        raise RuntimeError(
            f"Smoke checks failed; inspect smoke and orchestration logs: {failures}"
        )
    # Ensure every model/category has calibrated settings before full evals.
    for model_key in models:
        for category in TASKS:
            if (
                concurrency_state.get(model_key, {}).get(category, {}).get("status")
                != "completed"
            ):
                raise RuntimeError(
                    f"Missing smoke calibration for {model_key}/{category}"
                )
    print("Smoke phase passed for every model and evaluation category.", flush=True)


def update_progress(root: Path, phase: str, seed: int | None = None) -> None:
    manifest = read_json(root / "experiment_manifest.json", {}) or {}
    specs = manifest.get("jobs", [])
    quantized = sum(
        (checkpoint_dir(root, spec) / "quantization_complete.json").is_file()
        for spec in specs
    )
    statuses = []
    for spec in specs:
        state = read_json(state_path(root, spec["key"]), {}) or {}
        if seed is None:
            statuses.append(state.get("quantization_status", "pending"))
        else:
            statuses.append(state.get("full_seed_status", {}).get(str(seed), "pending"))
    counts: dict[str, int] = {}
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    print(
        f"PROGRESS phase={phase} seed={seed} quantized={quantized}/{len(specs)} "
        f"states={counts}",
        flush=True,
    )
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    if gpu.returncode == 0:
        print(f"GPU UTILIZATION\n{gpu.stdout.rstrip()}", flush=True)
    chg = subprocess.run(["chg", "status"], capture_output=True, text=True)
    if chg.returncode == 0:
        print(f"CHG STATUS\n{chg.stdout.rstrip()}", flush=True)
    adapt_future_concurrency(root, gpu.stdout, chg.stdout)
    report = {
        "at": utc_now(),
        "phase": phase,
        "seed": seed,
        "quantized": quantized,
        "job_count": len(specs),
        "states": counts,
        "nvidia_smi": gpu.stdout if gpu.returncode == 0 else gpu.stderr,
        "chg_status": chg.stdout if chg.returncode == 0 else chg.stderr,
    }
    with (root / "monitoring.jsonl").open("a") as out:
        out.write(json.dumps(report, sort_keys=True) + "\n")


def adapt_future_concurrency(root: Path, gpu_text: str, chg_text: str) -> None:
    """Adjust queued eval concurrency when repeated telemetry shows a clear issue."""
    gpu_util: dict[int, float] = {}
    for line in gpu_text.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 2:
            try:
                gpu_util[int(fields[0])] = float(fields[1].rstrip("%"))
            except ValueError:
                continue
    owned = []
    user = getpass.getuser()
    for line in chg_text.splitlines():
        match = re.match(r"\s*(\d+)\s+│\s+● IN_USE\s+│\s+([^│]+)", line)
        if match and match.group(2).strip() == user:
            index = int(match.group(1))
            if index in gpu_util:
                owned.append(gpu_util[index])
    if not owned:
        return

    state_path_ = root / "monitoring_state.json"
    state = read_json(state_path_, {}) or {}
    average = sum(owned) / len(owned)
    state["last_owned_gpu_utilization"] = average
    state["last_owned_gpu_count"] = len(owned)
    state["low_util_samples"] = (
        state.get("low_util_samples", 0) + 1 if average < 50 else 0
    )

    concurrency_path = root / "concurrency.json"
    concurrency_state = read_json(concurrency_path, {}) or {}
    changed = []
    if state["low_util_samples"] >= 2:
        for model_key, categories in concurrency_state.items():
            for category, entry in categories.items():
                if entry.get("status") != "completed":
                    continue
                if entry.get("kv_cache_peak_percent", 0) > 85:
                    continue
                old = int(entry.get("num_concurrent", 1))
                if old < 256:
                    entry["num_concurrent"] = min(256, old * 2)
                    entry["last_adjustment"] = "doubled_after_low_gpu_utilization"
                    entry["updated_at"] = utc_now()
                    changed.append(
                        f"{model_key}/{category}:{old}->{entry['num_concurrent']}"
                    )
        state["low_util_samples"] = 0
    if changed:
        atomic_json(concurrency_path, concurrency_state)
        print(f"ADJUSTED FUTURE EVAL CONCURRENCY {changed}", flush=True)
    state["updated_at"] = utc_now()
    atomic_json(state_path_, state)


def result_scores(root: Path, spec: dict[str, Any]) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = {
        "gsm8k": [],
        "mmlu": [],
        "math500": [],
        "gpqa": [],
    }
    state = read_json(state_path(root, spec["key"]), {}) or {}

    def lm_eval_strict(
        run_dir: Path, task: str, metric: str, seed: int
    ) -> float | None:
        candidates = list((run_dir / "results").glob(f"{task}_seed{seed}_*.json"))
        if not candidates:
            return None
        data = json.loads(
            max(candidates, key=lambda path: path.stat().st_mtime).read_text()
        )
        try:
            return float(data["results"][task][metric])
        except (KeyError, TypeError, ValueError):
            return None

    for seed in SEEDS:
        seed_eval = state.get("full_evaluations", {}).get(str(seed), {})
        instruct = seed_eval.get("instruct", {})
        reasoning = seed_eval.get("reasoning", {})
        if instruct.get("status") == "completed":
            instruct_dir = Path(instruct["run_dir"])
            value = lm_eval_strict(
                instruct_dir,
                "gsm8k_platinum_cot_llama",
                "exact_match,strict-match",
                seed,
            )
            if value is not None:
                scores["gsm8k"].append(value)
            value = lm_eval_strict(
                instruct_dir,
                "mmlu_cot_llama",
                "exact_match,strict_match",
                seed,
            )
            if value is not None:
                scores["mmlu"].append(value)
            summary = read_json(instruct_dir / "summary_data.json", {}) or {}
            try:
                scores["math500"].append(
                    float(summary["results"]["math_500"]["pass@k:k=1&n=1"]["mean"])
                )
            except (KeyError, TypeError, ValueError):
                pass
        if reasoning.get("status") == "completed":
            summary = (
                read_json(Path(reasoning["run_dir"]) / "summary_data.json", {}) or {}
            )
            try:
                scores["gpqa"].append(
                    float(summary["results"]["gpqa_diamond"]["gpqa_pass@k:k=1"]["mean"])
                )
            except (KeyError, TypeError, ValueError):
                pass
    return scores


def write_summary(root: Path) -> tuple[Path, Path]:
    with SUMMARY_LOCK:
        manifest = read_json(root / "experiment_manifest.json", {}) or {}
        rows = []
        for spec in manifest.get("jobs", []):
            scores = result_scores(root, spec)
            state = read_json(state_path(root, spec["key"]), {}) or {}
            complete_seeds = sum(
                state.get("full_seed_status", {}).get(str(seed)) == "completed"
                for seed in SEEDS
            )
            means = {
                key: (sum(values) / len(values) * 100 if values else None)
                for key, values in scores.items()
            }
            rows.append(
                {
                    "model": manifest["models"][spec["model"]]["id"],
                    "model_key": spec["model"],
                    "power": spec["power"],
                    "magnitude_normalization": magnitude_enabled(spec),
                    "grid": "exponential 1–1/64 (7 points)",
                    "seeds": complete_seeds,
                    "gsm8k_strict": means["gsm8k"],
                    "mmlu_strict": means["mmlu"],
                    "math500": means["math500"],
                    "gpqa_diamond": means["gpqa"],
                    "gsm8k_seed_count": len(scores["gsm8k"]),
                    "mmlu_seed_count": len(scores["mmlu"]),
                    "math500_seed_count": len(scores["math500"]),
                    "gpqa_seed_count": len(scores["gpqa"]),
                }
            )
        headers = [
            "model",
            "model_key",
            "power",
            "magnitude_normalization",
            "grid",
            "seeds",
            "gsm8k_strict",
            "mmlu_strict",
            "math500",
            "gpqa_diamond",
            "gsm8k_seed_count",
            "mmlu_seed_count",
            "math500_seed_count",
            "gpqa_seed_count",
        ]
        csv_path = root / "ablation_summary.csv"
        with csv_path.open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

        lines = [
            f"# {SCHEME_NAME} model-free MSE ablation results",
            "",
            "Scores are percentages averaged across completed eval-agent seed runs. "
            "Each metric uses its available completed seeds; `Seeds` counts seeds "
            "where all four evaluations completed. No overall ranking is applied.",
            "",
            "| Model | Power | Magnitude normalization | Grid | Seeds | "
            "Avg GSM8K strict | Avg MMLU strict | Avg Math500 | Avg GPQA Diamond |",
            "|---|---:|:---:|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows:

            def fmt(key: str) -> str:
                return "—" if row[key] is None else f"{row[key]:.2f}%"

            lines.append(
                f"| {row['model']} | {row['power']:g} | "
                f"{str(row['magnitude_normalization']).lower()} | "
                f"{row['grid']} | {row['seeds']} | {fmt('gsm8k_strict')} | "
                f"{fmt('mmlu_strict')} | {fmt('math500')} | {fmt('gpqa_diamond')} |"
            )
        lines.extend(
            [
                "",
                "Full task logs, per-seed outputs, checkpoint metadata, and "
                "progress records are in the run directory.",
                "",
            ]
        )
        content = "\n".join(lines)
        md_path = root / "ablation_summary.md"
        md_path.write_text(content)
    return root / "ablation_summary.csv", root / "ablation_summary.md"


def full_job(root: Path, base: Any, spec: dict[str, Any], seed: int) -> dict[str, Any]:
    state = read_json(state_path(root, spec["key"]), {}) or {}
    seed_key = str(seed)
    has_current_eval_alias = (
        state.get("evaluation_model_alias_version") == EVALUATION_MODEL_ALIAS_VERSION
    )
    if has_current_eval_alias and state.get("full_seed_status", {}).get(seed_key) in (
        "completed",
        "failed",
    ):
        return {
            "job": spec["key"],
            "seed": seed,
            "status": state["full_seed_status"][seed_key],
            "skipped": True,
        }
    if not ensure_quantized(root, spec):
        write_state(
            root,
            spec["key"],
            {
                "full_seed_status": {
                    **state.get("full_seed_status", {}),
                    seed_key: "failed",
                },
                "full_status": "failed",
            },
        )
        return {
            "job": spec["key"],
            "seed": seed,
            "status": "failed",
            "reason": "quantization failed",
        }

    state = read_json(state_path(root, spec["key"]), {}) or {}
    evaluations = state.get("full_evaluations", {}) if has_current_eval_alias else {}
    seed_evaluations = evaluations.get(seed_key, {})
    concurrency_state = read_json(root / "concurrency.json", {}) or {}
    for category in TASKS:
        if seed_evaluations.get(category, {}).get("status") == "completed":
            continue
        calibrated = concurrency_state[spec["model"]][category]
        result = evaluate_with_budget(
            root,
            base,
            spec,
            category,
            smoke=False,
            concurrency=int(calibrated["num_concurrent"]),
            tensor_parallel_size=int(calibrated["tensor_parallel_size"]),
            seed=seed,
        )
        seed_evaluations[category] = result
        evaluations[seed_key] = seed_evaluations
        write_state(
            root,
            spec["key"],
            {
                "full_evaluations": evaluations,
                "evaluation_model_alias_version": EVALUATION_MODEL_ALIAS_VERSION,
            },
        )

    complete = all(
        seed_evaluations.get(category, {}).get("status") == "completed"
        for category in TASKS
    )
    seed_statuses = state.get("full_seed_status", {}) if has_current_eval_alias else {}
    seed_statuses[seed_key] = "completed" if complete else "failed"
    complete_all = all(
        seed_statuses.get(str(round_seed)) == "completed" for round_seed in SEEDS
    )
    write_state(
        root,
        spec["key"],
        {
            "full_seed_status": seed_statuses,
            "full_status": "completed" if complete_all else "in_progress",
            "full_evaluations": evaluations,
            "evaluation_model_alias_version": EVALUATION_MODEL_ALIAS_VERSION,
        },
    )
    return {
        "job": spec["key"],
        "seed": seed,
        "status": "completed" if complete else "failed",
        "evaluations": seed_evaluations,
    }


def monitor_batch(
    root: Path,
    phase: str,
    seed: int | None,
    futures: dict[concurrent.futures.Future, str],
) -> list[Any]:
    pending = set(futures)
    results: list[Any] = []
    while pending:
        done, pending = concurrent.futures.wait(
            pending,
            timeout=MONITOR_INTERVAL_SECONDS,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        if not done:
            update_progress(root, phase, seed)
            continue
        for future in done:
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "status": "failed",
                    "job": futures[future],
                    "error": repr(exc),
                }
            results.append(result)
            print(f"JOB RESULT {json.dumps(result, sort_keys=True)}", flush=True)
        write_summary(root)
        update_progress(root, phase, seed)
    return results


def safe_full_job(
    root: Path, base: Any, spec: dict[str, Any], seed: int
) -> dict[str, Any]:
    try:
        return full_job(root, base, spec, seed)
    except Exception as exc:
        state = read_json(state_path(root, spec["key"]), {}) or {}
        seed_statuses = state.get("full_seed_status", {})
        seed_statuses[str(seed)] = "failed"
        write_state(
            root,
            spec["key"],
            {
                "full_seed_status": seed_statuses,
                "full_status": "failed",
                "last_error": repr(exc),
            },
        )
        return {
            "job": spec["key"],
            "seed": seed,
            "status": "failed",
            "error": repr(exc),
        }


def run_full(
    root: Path, base: Any, models: dict[str, dict[str, Any]], workers: int
) -> None:
    manifest = prepare(root)
    for model_key in models:
        for category in TASKS:
            state = read_json(root / "concurrency.json", {}) or {}
            if state.get(model_key, {}).get(category, {}).get("status") != "completed":
                raise RuntimeError(
                    f"Smoke calibration is missing for {model_key}/{category}"
                )
    specs = job_specs(models)
    for seed in SEEDS:
        print(f"Starting full evaluation seed round {seed}", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(safe_full_job, root, base, spec, seed): spec["key"]
                for spec in specs
            }
            monitor_batch(root, "full", seed, futures)
        write_summary(root)
        update_progress(root, "full_round_complete", seed)
    manifest["completed_at"] = utc_now()
    atomic_json(root / "experiment_manifest.json", manifest)
    write_summary(root)


def run_experiment(root: Path, workers: int) -> int:
    manifest = prepare(root)
    base = load_base_runner()
    models = model_specs(base)
    configure_eval_helper(base, root, manifest, models)
    cache_models(root, models, workers)
    smoke_phase(root, base, models, workers)
    run_full(root, base, models, workers)
    print(f"Experiment complete. Summary: {root / 'ablation_summary.md'}", flush=True)
    return 0


def check_config() -> int:
    from compressed_tensors.quantization import preset_name_to_scheme

    from llmcompressor.observers.grid_search import _make_grid_points
    from llmcompressor.observers.mse import NVFP4ExpandedMSEObserver

    scheme = preset_name_to_scheme(SCHEME_NAME, ["Linear"])
    scheme.weights.observer = "nvfp4_expanded_mse"
    scheme.weights.observer_kwargs = {
        **OBSERVER_KWARGS,
        "norm": 2.0,
        "magnitude_normalization": True,
    }
    points = _make_grid_points(
        OBSERVER_KWARGS["maxshrink"],
        OBSERVER_KWARGS["grid"],
        OBSERVER_KWARGS["grid_type"],
    )
    expected = [2.0**-index for index in range(7)]
    if points != expected:
        raise RuntimeError(f"Unexpected exponential MSE grid: {points}")
    if (
        scheme.weights.group_size != 32
        or scheme.weights.scale_dtype != __import__("torch").uint8
    ):
        raise RuntimeError(
            f"{SCHEME_NAME} preset does not have expected group/E8M0 settings"
        )
    activations = scheme.input_activations
    if (
        activations is None
        or activations.num_bits != 8
        or activations.type != "float"
        or activations.strategy != "group"
        or activations.dynamic is not True
        or activations.group_size != 128
    ):
        raise RuntimeError(
            f"{SCHEME_NAME} preset does not declare dynamic group-128 FP8 inputs"
        )
    print(
        json.dumps(
            {
                "scheme": SCHEME_NAME,
                "observer": scheme.weights.observer,
                "group_size": scheme.weights.group_size,
                "scale_dtype": str(scheme.weights.scale_dtype),
                "input_activations": {
                    "num_bits": activations.num_bits,
                    "type": activations.type,
                    "strategy": activations.strategy,
                    "dynamic": activations.dynamic,
                    "group_size": activations.group_size,
                },
                "observer_kwargs": scheme.weights.observer_kwargs,
                "candidate_multipliers": points,
                "observer_class": NVFP4ExpandedMSEObserver.__name__,
            },
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--root", type=Path, default=DEFAULT_RUN_ROOT)
    run_parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--root", type=Path, default=DEFAULT_RUN_ROOT)
    cache_parser = sub.add_parser("cache-model-worker")
    cache_parser.add_argument("--root", type=Path, required=True)
    cache_parser.add_argument("--model", required=True)
    scan_parser = sub.add_parser("scan-models-worker")
    scan_parser.add_argument("--root", type=Path, required=True)
    quant_parser = sub.add_parser("quantize-worker")
    quant_parser.add_argument("--root", type=Path, required=True)
    quant_parser.add_argument("--job-key", required=True)
    quant_parser.add_argument("--model-snapshot", type=Path, required=True)
    quant_one_parser = sub.add_parser("quantize-one")
    quant_one_parser.add_argument("--root", type=Path, default=DEFAULT_RUN_ROOT)
    quant_one_parser.add_argument("--job-key", required=True)
    summary_parser = sub.add_parser("summary")
    summary_parser.add_argument("--root", type=Path, default=DEFAULT_RUN_ROOT)
    sub.add_parser("check-config")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.root), indent=2))
        return 0
    if args.command == "cache-model-worker":
        base = load_base_runner()
        models = model_specs(base)
        if args.model not in models:
            raise KeyError(args.model)
        cache_model(args.root, args.model)
        return 0
    if args.command == "scan-models-worker":
        base = load_base_runner()
        models = model_specs(base)
        prepare(args.root)
        for model_key in models:
            cache_model(args.root, model_key)
        return 0
    if args.command == "quantize-worker":
        base = load_base_runner()
        models = model_specs(base)
        prepare(args.root)
        spec = next(item for item in job_specs(models) if item["key"] == args.job_key)
        quantize_worker(args.root, spec, args.model_snapshot)
        return 0
    if args.command == "quantize-one":
        base = load_base_runner()
        models = model_specs(base)
        prepare(args.root)
        spec = next(item for item in job_specs(models) if item["key"] == args.job_key)
        return 0 if ensure_quantized(args.root, spec) else 1
    if args.command == "summary":
        csv_path, md_path = write_summary(args.root)
        print(csv_path)
        print(md_path)
        return 0
    if args.command == "check-config":
        return check_config()
    if args.command == "run":
        if args.workers < 1 or args.workers > 6:
            raise ValueError(
                "workers must be between 1 and 6 for the current allocation"
            )
        return run_experiment(args.root, args.workers)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
