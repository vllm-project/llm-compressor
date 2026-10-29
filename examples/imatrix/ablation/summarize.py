#!/usr/bin/env python3
"""Collect eval-agent score summaries into one CSV and JSON matrix."""

import argparse
import csv
import fcntl
import json
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    lock_handle = (args.root / ".summarize.lock").open("w")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)

    def load_timestamped_lm_eval_metrics(results_dir: Path, task_name: str) -> dict:
        """Read timestamped lm-eval outputs whose names are not summary.py's expected name."""
        values_by_seed: dict[int, dict[str, float]] = {}
        for result_path in sorted(
            results_dir.glob(f"{task_name}_seed*.json"), key=lambda path: path.stat().st_mtime
        ):
            match = re.search(r"_seed(\d+)", result_path.name)
            if match is None:
                continue
            try:
                data = json.loads(result_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            metrics = {}
            for task_metrics in data.get("results", {}).values():
                if not isinstance(task_metrics, dict):
                    continue
                for metric, value in task_metrics.items():
                    if (
                        isinstance(value, (int, float))
                        and "_stderr" not in metric
                        and metric not in {"name", "alias", "sample_len"}
                    ):
                        metrics[metric] = float(value)
            if metrics:
                values_by_seed[int(match.group(1))] = metrics

        aggregated = {}
        metric_names = {metric for metrics in values_by_seed.values() for metric in metrics}
        for metric in metric_names:
            values = [
                metrics[metric]
                for _, metrics in sorted(values_by_seed.items())
                if metric in metrics
            ]
            aggregated[metric] = {
                "mean": sum(values) / len(values),
                "count": len(values),
            }
        return aggregated

    def load_final_task_status(case_dir: Path) -> tuple[int, int, int] | None:
        """Return final task outcomes, separating retries from final failures.

        A task can fail on an attempt and then succeed on retry. The summary
        status records both events cumulatively, so use the last terminal event
        for each seed and task when reporting completed/failed counts. Retain
        the number of failed attempts separately for auditability.
        """
        events_path = case_dir / "events.jsonl"
        if not events_path.exists():
            return None

        final_outcomes = {}
        failed_attempts = 0
        for line in events_path.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") not in {"eval_completed", "eval_failed"}:
                continue
            seed = event.get("seed")
            task = event.get("task")
            if not isinstance(seed, int) or not isinstance(task, str):
                continue
            outcome = event["event"]
            final_outcomes[(seed, task)] = outcome
            failed_attempts += outcome == "eval_failed"

        if not final_outcomes:
            return None
        completed = sum(outcome == "eval_completed" for outcome in final_outcomes.values())
        failed = sum(outcome == "eval_failed" for outcome in final_outcomes.values())
        return completed, failed, failed_attempts

    manifest = json.loads((args.root / "manifest.json").read_text())
    corrections_path = args.root / "result_corrections.json"
    corrections = {}
    if corrections_path.exists():
        try:
            corrections = json.loads(corrections_path.read_text()).get("cases", {})
        except (OSError, json.JSONDecodeError):
            corrections = {}
    rows = []
    for model in manifest["models"]:
        for power in manifest["powers"]:
            for objective in manifest["objectives"]:
                power_tag = f"p{power:g}".replace(".", "p")
                case_id = f"{model['id']}__{objective['id']}__{power_tag}"
                summary_path = (
                    args.root / "evaluations" / f"{case_id}-full" / "summary_data.json"
                )
                row = {
                    "case_id": case_id,
                    "model_id": model["id"],
                    "hf_id": model["hf_id"],
                    "profile": model["profile"],
                    "objective_id": objective["id"],
                    "objective": objective["description"],
                    "power": power,
                    "evaluation_status": "missing",
                    "tasks_completed": 0,
                    "tasks_failed": 0,
                    "failed_attempts": 0,
                }
                if summary_path.exists():
                    try:
                        summary = json.loads(summary_path.read_text())
                    except (OSError, json.JSONDecodeError):
                        # A concurrent evaluator may be replacing this file.
                        summary = {}
                    status = summary.get("status", {})
                    final_status = load_final_task_status(
                        args.root / "evaluations" / f"{case_id}-full"
                    )
                    if final_status is None:
                        row["tasks_completed"] = status.get("completed", 0)
                        row["tasks_failed"] = status.get("failed", 0)
                        row["failed_attempts"] = status.get("failed", 0)
                    else:
                        (
                            row["tasks_completed"],
                            row["tasks_failed"],
                            row["failed_attempts"],
                        ) = final_status
                    row["evaluation_status"] = (
                        "completed"
                        if row["tasks_completed"] == 9 and row["tasks_failed"] == 0
                        else "partial"
                    )
                    for seed_index, metric_corrections in corrections.get(
                        case_id, {}
                    ).items():
                        try:
                            seed_index = int(seed_index)
                        except (TypeError, ValueError):
                            continue
                        for metric_path, value in metric_corrections.items():
                            if "/" not in metric_path:
                                continue
                            task_name, metric_name = metric_path.split("/", 1)
                            result = summary.get("results", {}).get(task_name, {}).get(
                                metric_name
                            )
                            if not isinstance(result, dict):
                                continue
                            values = result.get("values")
                            if not isinstance(values, list) or not (
                                0 <= seed_index < len(values)
                            ):
                                continue
                            values[seed_index] = value
                            result["count"] = len(values)
                            result["mean"] = sum(values) / len(values)
                    for task_name, metrics in summary.get("results", {}).items():
                        for metric, result in metrics.items():
                            row[f"{task_name}/{metric}"] = result.get("mean")
                            row[f"{task_name}/{metric}/seed_count"] = result.get("count")
                    raw_lm_metrics = load_timestamped_lm_eval_metrics(
                        args.root / "evaluations" / f"{case_id}-full" / "results",
                        "gsm8k_platinum_cot_llama",
                    )
                    for metric, result in raw_lm_metrics.items():
                        row[f"gsm8k_platinum_cot_llama/{metric}"] = result["mean"]
                        row[f"gsm8k_platinum_cot_llama/{metric}/seed_count"] = result["count"]
                rows.append(row)

    out_json = args.root / "results_matrix.json"
    out_json.write_text(json.dumps(rows, indent=2) + "\n")
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    out_csv = args.root / "results_matrix.csv"
    with out_csv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
