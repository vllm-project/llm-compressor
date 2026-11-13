#!/usr/bin/env python3
"""Run the selected eval-agent benchmarks on one checkpoint."""

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from eval_agent.audit import RunAudit
from eval_agent.cli import _write_summary_data
from eval_agent.runner import BenchmarkRunner, get_seeds, get_tasks_for_category
from eval_agent.server import ServerError, VLLMServer


def _selected_tasks(profile: str) -> list[dict]:
    instruct = get_tasks_for_category("instruct")
    reasoning = get_tasks_for_category("reasoning")
    task_profile = instruct if profile == "instruct" else reasoning
    gsm = next(
        task for task in task_profile if task["name"] == "gsm8k_platinum_cot_llama"
    )
    math = next(task for task in task_profile if task["name"] == "math_500")
    gpqa = next(t for t in reasoning if t["name"] == "gpqa_diamond")
    return [gsm, gpqa, math]


def _already_completed(audit: RunAudit, tasks: list[dict], seeds: list[int]) -> bool:
    events = audit.read_events()
    completed = {
        (e.get("task"), e.get("seed")) for e in events if e["event"] == "eval_completed"
    }
    expected = {(task["name"], seed) for task in tasks for seed in seeds}
    return expected <= completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument(
        "--seed-index",
        type=int,
        help="Run only this zero-based registry seed; omitted for smoke runs.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    model_cfg = next(m for m in manifest["models"] if m["id"] == args.model_id)
    tasks = _selected_tasks(model_cfg["profile"])
    mode = "smoke" if args.smoke_only else "full"
    registry_seeds = get_seeds()
    if args.seed_index is not None:
        if args.smoke_only:
            parser.error("--seed-index cannot be combined with --smoke-only")
        if args.seed_index < 0 or args.seed_index >= len(registry_seeds):
            parser.error(
                f"--seed-index must be between 0 and {len(registry_seeds) - 1}"
            )
        selected_seeds = [registry_seeds[args.seed_index]]
    else:
        selected_seeds = (
            registry_seeds[:1]
            if args.smoke_only
            else [
                seed
                for task in tasks
                for seed in registry_seeds[: task["n_repetitions"]]
            ]
        )
        selected_seeds = list(dict.fromkeys(selected_seeds))
    run_dir = args.run_root / "evaluations" / f"{args.case_id}-{mode}"
    audit = RunAudit(run_dir)
    prior_manifest = audit.load_manifest()
    if prior_manifest and _already_completed(audit, tasks, selected_seeds):
        print(
            f"All {mode} tasks for seeds {selected_seeds} already completed: {run_dir}",
            flush=True,
        )
        return 0

    max_length = 69632 if model_cfg["profile"] == "reasoning" else 20480
    timeout = 3600 if model_cfg["profile"] == "reasoning" else 1200
    reasoning_parser = model_cfg.get("reasoning_parser")
    server_args = [
        "chg",
        "run",
        "--gpus",
        "1",
        "--",
        "env",
        "VLLM_LOGGING_LEVEL=INFO",
        "CUDA_HOME=/usr/local/cuda-13.0",
        "CUDA_PATH=/usr/local/cuda-13.0",
        "PATH="
        + ":".join(
            [
                "/home/HDCharles/vllm/bin",
                "/usr/local/cuda-13.0/bin",
                os.environ.get("PATH", ""),
            ]
        ),
        "/home/HDCharles/vllm/bin/vllm",
        "serve",
        str(args.checkpoint),
        "--served-model-name",
        "ablation",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(max_length),
        "--gpu-memory-utilization",
        "0.90",
        "--trust-remote-code",
        "--port",
        str(args.port),
    ]
    if reasoning_parser:
        server_args.extend(["--reasoning-parser", reasoning_parser])
    server_cmd = shlex.join(server_args)

    audit.init_dirs()
    run_manifest = audit.write_manifest(
        model=str(args.checkpoint),
        category=f"ablation-{model_cfg['profile']}",
        server_cmd=server_cmd,
        base_gen_params=model_cfg["generation_params"],
        port=args.port,
        num_concurrent=args.concurrency,
        timeout=timeout,
        max_length=max_length,
        smoke_only=args.smoke_only,
        lm_eval_venv="/home/HDCharles/vllm",
        lighteval_venv="/home/HDCharles/vllm",
        inspect_ai_venv="/home/HDCharles/vllm",
        registry_snapshot={"seeds": get_seeds(), "tasks": tasks},
        task_names=[task["name"] for task in tasks],
    )
    run_manifest["selected_seeds"] = selected_seeds
    if args.seed_index is not None:
        run_manifest["seed_index"] = args.seed_index
    checkpoint_metadata_path = args.checkpoint / "ablation_metadata.json"
    if checkpoint_metadata_path.exists():
        checkpoint_metadata = json.loads(checkpoint_metadata_path.read_text())
        run_manifest["checkpoint_group_size"] = checkpoint_metadata.get("group_size")
    (run_dir / "manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
    audit.log_event("run_started", run_id=run_dir.name, model=str(args.checkpoint))

    server = VLLMServer(
        server_cmd=server_cmd, audit=audit, port=args.port, health_timeout=1800
    )
    try:
        server.start()
    except ServerError as exc:
        print(f"vLLM server startup failed: {exc}", file=sys.stderr)
        audit.log_event("run_completed", status="server_failed", error=str(exc))
        _write_summary_data(run_dir)
        return 1

    try:
        runner = BenchmarkRunner(
            audit=audit,
            model="ablation",
            category=f"ablation-{model_cfg['profile']}",
            base_gen_params=model_cfg["generation_params"],
            port=args.port,
            num_concurrent=args.concurrency,
            timeout=timeout,
            max_length=max_length,
            lm_eval_venv="/home/HDCharles/vllm",
            lighteval_venv="/home/HDCharles/vllm",
            inspect_ai_venv="/home/HDCharles/vllm",
            smoke_only=args.smoke_only,
            # Smoke retries should reuse successful tasks and rerun only failures.
            skip_completed=True,
            tasks=tasks,
        )
        if args.seed_index is not None:
            # Keep the immutable registry task definitions and repetition counts;
            # schedule its seeds in separate global passes from the orchestrator.
            runner.seeds = selected_seeds
        results = runner.run_all()
    finally:
        server.stop()

    status = "completed" if not results["failed"] else "completed_with_failures"
    audit.log_event(
        "run_completed",
        status=status,
        completed=len(results["completed"]),
        failed=len(results["failed"]),
        failures=results["failed"],
    )
    _write_summary_data(run_dir)
    if args.smoke_only and results["failed"]:
        return 1
    return 0 if not results["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
