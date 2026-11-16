#!/usr/bin/env python3
"""Evaluate one NVFP4 iMatrix checkpoint with the eval-agent task registry.

Run one smoke evaluation, then invoke this script once per seed. For the
research suite, MMLU is included only in the seed-1234 run; all other tasks use
seeds 1234, 2345, and 3456. Task definitions remain owned by eval-agent.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

MODEL_CATEGORIES = {
    "qwen38_27b": "reasoning",
    "qwen3_30b_a3b": "reasoning",
    "nemotron35_lightning_30b_a3b": "reasoning",
    "qwen25_7b_instruct": "instruct",
    "gemma4_26b_a4b_it": "instruct",
}
GENERATION_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_CATEGORIES, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--server-cmd", required=True)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--seed", type=int, choices=(1234, 2345, 3456))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-concurrent", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--lm-eval-venv", type=Path, default=Path(".venvs/lm-eval"))
    parser.add_argument("--lighteval-venv", type=Path, default=Path(".venvs/lighteval"))
    parser.add_argument(
        "--eval-agent-root",
        type=Path,
        default=Path(os.environ["EVAL_AGENT_ROOT"])
        if os.environ.get("EVAL_AGENT_ROOT")
        else None,
        help="Optional source checkout containing the eval_agent package",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        help="Optional isolated Hugging Face cache directory for this run",
    )
    parser.add_argument(
        "--gen-params",
        help=(
            "JSON generation parameters; defaults to supported fields in "
            "checkpoint generation_config.json"
        ),
    )
    parser.add_argument(
        "--module-profile", choices=("all", "attention_only"), default="all"
    )
    args = parser.parse_args()
    if args.phase == "full" and args.seed is None:
        parser.error("--seed is required with --phase full")
    if args.phase == "smoke" and args.seed is not None:
        parser.error("--seed is only valid with --phase full")
    if args.num_concurrent < 1 or args.max_length < 1 or args.timeout < 1:
        parser.error("--num-concurrent, --max-length, and --timeout must be positive")
    return args


def _load_eval_agent(eval_agent_root: Path | None):
    if eval_agent_root is not None:
        sys.path.insert(0, str(eval_agent_root.resolve()))
    from eval_agent.audit import RunAudit
    from eval_agent.runner import BenchmarkRunner, get_seeds, get_tasks_for_category
    from eval_agent.server import ServerError, VLLMServer
    from eval_agent.summary import generate_summary_data

    return (
        RunAudit,
        BenchmarkRunner,
        get_seeds,
        get_tasks_for_category,
        ServerError,
        VLLMServer,
        generate_summary_data,
    )


def _tasks_for_model(model: str, get_tasks_for_category, seed: int | None):
    category = MODEL_CATEGORIES[model]
    by_category = {
        name: {
            task["name"]: copy.deepcopy(task) for task in get_tasks_for_category(name)
        }
        for name in ("instruct", "reasoning")
    }
    tasks = [
        by_category[category]["gsm8k_platinum_cot_llama"],
        by_category["instruct"]["mmlu_cot_llama"],
        by_category[category]["ifeval"],
        by_category[category]["math_500"],
        by_category["reasoning"]["gpqa_diamond"],
    ]
    if seed is not None and seed != 1234:
        tasks = [task for task in tasks if task["name"] != "mmlu_cot_llama"]
    return tasks


def _generation_params(checkpoint: Path, override: str | None) -> dict:
    if override is not None:
        values = json.loads(override)
    else:
        path = checkpoint / "generation_config.json"
        values = json.loads(path.read_text()) if path.is_file() else {}
    if not isinstance(values, dict):
        raise ValueError("generation parameters must be a JSON object")
    return {key: values[key] for key in GENERATION_FIELDS if key in values}


def _repair_gemma4_attention_targets(
    checkpoint: Path, model: str, module_profile: str
) -> bool:
    """Map exact HF targets and shared V projections for Gemma4's fused QKV."""
    if model != "gemma4_26b_a4b_it" or module_profile != "attention_only":
        return False

    metadata_path = checkpoint / "imatrix_ablation_config.json"
    config_path = checkpoint / "config.json"
    try:
        metadata = json.loads(metadata_path.read_text())
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot read Gemma4 checkpoint metadata: {error}"
        ) from error

    targets = metadata.get("target_module_names")
    if not isinstance(targets, list) or not targets:
        raise RuntimeError(
            "Gemma4 attention-only checkpoint has no target-module manifest"
        )

    text_config = config.get("text_config", config)
    layer_types = text_config.get("layer_types", [])
    implicit_v = []
    if text_config.get("attention_k_eq_v"):
        implicit_v = [
            f"model.language_model.layers.{index}.self_attn.v_proj"
            for index, kind in enumerate(layer_types)
            if kind == "full_attention"
        ]
    vllm_targets = list(targets)
    vllm_targets.extend(name for name in implicit_v if name not in vllm_targets)

    groups = config.get("quantization_config", {}).get("config_groups", {})
    changed = False
    for group in groups.values():
        current = group.get("targets", [])
        has_legacy_regex = any(
            isinstance(item, str)
            and item.startswith("re:")
            and "model\\.language_model\\.layers" in item
            for item in current
        )
        overlaps_manifest = any(
            item in targets for item in current if isinstance(item, str)
        )
        if has_legacy_regex or overlaps_manifest:
            if current != vllm_targets:
                group["targets"] = vllm_targets
                changed = True
    if not changed:
        if not any(
            any(
                isinstance(item, str) and item in vllm_targets
                for item in group.get("targets", [])
            )
            for group in groups.values()
        ):
            raise RuntimeError(
                "Gemma4 attention targets did not match a quantization config group"
            )
        return False

    temporary = config_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    temporary.replace(config_path)
    print(
        "Rewrote Gemma4 vLLM targets for path mapping and shared K/V fusion", flush=True
    )
    return True


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.resolve()
    _repair_gemma4_attention_targets(checkpoint, args.model, args.module_profile)
    (
        RunAudit,
        BenchmarkRunner,
        get_seeds,
        get_tasks_for_category,
        ServerError,
        VLLMServer,
        generate_summary_data,
    ) = _load_eval_agent(args.eval_agent_root)

    tasks = _tasks_for_model(args.model, get_tasks_for_category, args.seed)
    gen_params = _generation_params(checkpoint, args.gen_params)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    audit = RunAudit(run_dir)
    audit.init_dirs()
    manifest = audit.load_manifest()
    if manifest is None:
        manifest = audit.write_manifest(
            model=str(checkpoint),
            category=MODEL_CATEGORIES[args.model],
            server_cmd=args.server_cmd,
            base_gen_params=gen_params,
            port=args.port,
            num_concurrent=args.num_concurrent,
            timeout=args.timeout,
            max_length=args.max_length,
            smoke_only=args.phase == "smoke",
            lm_eval_venv=str(args.lm_eval_venv.resolve()),
            lighteval_venv=str(args.lighteval_venv.resolve()),
            registry_snapshot={"seeds": get_seeds(), "tasks": tasks},
            task_names=[task["name"] for task in tasks],
        )

    server = VLLMServer(
        server_cmd=args.server_cmd,
        audit=audit,
        port=args.port,
        health_timeout=max(args.timeout, 3600),
    )
    try:
        server.start()
    except ServerError as error:
        audit.log_event("resume_failed", error=str(error))
        print(f"vLLM server failed to start: {error}", file=sys.stderr, flush=True)
        return 1

    try:
        runner = BenchmarkRunner(
            audit=audit,
            model=str(checkpoint),
            category=MODEL_CATEGORIES[args.model],
            base_gen_params=gen_params,
            port=args.port,
            num_concurrent=args.num_concurrent,
            timeout=args.timeout,
            max_length=args.max_length,
            lm_eval_venv=str(args.lm_eval_venv.resolve()),
            lighteval_venv=str(args.lighteval_venv.resolve()),
            smoke_only=args.phase == "smoke",
            skip_completed=False,
            tasks=tasks,
        )
        if args.phase == "full":
            runner.seeds = [args.seed]
        # Each invocation gets private client caches. This avoids collisions
        # when separate eval-agent jobs share a work directory.
        client_work = run_dir / ".eval-client-work"
        client_hf = (
            args.hf_cache_dir.resolve()
            if args.hf_cache_dir
            else run_dir / ".eval-client-hf"
        )
        client_work.mkdir(parents=True, exist_ok=True)
        client_hf.mkdir(parents=True, exist_ok=True)
        old_cwd = Path.cwd()
        old_hf_home = os.environ.get("HF_HOME")
        old_datasets = os.environ.get("HF_DATASETS_CACHE")
        try:
            os.chdir(client_work)
            os.environ["HF_HOME"] = str(client_hf)
            os.environ.setdefault("HF_DATASETS_CACHE", str(client_hf / "datasets"))
            results = runner.run_all()
        finally:
            os.chdir(old_cwd)
            if old_hf_home is None:
                os.environ.pop("HF_HOME", None)
            else:
                os.environ["HF_HOME"] = old_hf_home
            if old_datasets is None:
                os.environ.pop("HF_DATASETS_CACHE", None)
            else:
                os.environ["HF_DATASETS_CACHE"] = old_datasets
    finally:
        server.stop()

    audit.log_event(
        "run_completed",
        status="completed" if not results["failed"] else "completed_with_failures",
        completed=len(results["completed"]),
        failed=len(results["failed"]),
        failures=results["failed"],
        phase=args.phase,
        seed=args.seed,
    )
    summary = generate_summary_data(run_dir)
    (run_dir / "summary_data.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {"completed": results["completed"], "failed": results["failed"]}, indent=2
        )
    )
    return 0 if not results["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
