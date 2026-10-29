#!/usr/bin/env python3
"""Finish Qwen3.8 seed 1234 evaluations using three available GPU slots."""

import concurrent.futures
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = Path("/mnt/data/engine/HDCharles/experiments/w4a16-imatrix-ablation-2026-10-02")
PYTHON = Path("/home/HDCharles/vllm/bin/python")
MODEL_ID = "qwen3.8-27b"
SEED = 1234
SEED_INDEX = 0
SLOTS = (5, 6, 8)
CONCURRENCY = 64
CURRENT_CASE = f"{MODEL_ID}__importance-error-power__p2"
CURRENT_EVALUATE_PID = 473284
TASKS = {"gsm8k_platinum_cot_llama", "gpqa_diamond", "math_500"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pid_is_current_evaluation(pid: int, case_id: str) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return b"evaluate.py" in command and case_id.encode() in command


def run_is_complete(case_id: str) -> bool:
    events_path = ROOT / "evaluations" / f"{case_id}-full" / "events.jsonl"
    try:
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
    except (OSError, json.JSONDecodeError):
        return False
    completed = {
        event.get("task")
        for event in events
        if event.get("event") == "eval_completed"
        and event.get("seed") == SEED
        and event.get("returncode") == 0
    }
    return TASKS <= completed


def main() -> int:
    manifest_path = HERE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    model = next(item for item in manifest["models"] if item["id"] == MODEL_ID)
    jobs = []
    for power in manifest["powers"]:
        power_tag = f"p{power:g}".replace(".", "p")
        for objective in manifest["objectives"]:
            case_id = f"{MODEL_ID}__{objective['id']}__{power_tag}"
            jobs.append(
                {
                    "case_id": case_id,
                    "objective_id": objective["id"],
                    "power": power,
                    "checkpoint": str(
                        ROOT / "checkpoints" / MODEL_ID / objective["id"] / power_tag
                    ),
                    "status": "pending",
                    "attempts": 0,
                }
            )

    status_path = ROOT / "seed1_qwen38_queue_status.json"
    queue_log = ROOT / "logs" / "evaluation" / "seed1-qwen38-queue.log"
    queue_log.parent.mkdir(parents=True, exist_ok=True)
    log_handle = queue_log.open("a", buffering=1)
    state_lock = threading.Lock()
    log_lock = threading.Lock()
    summary_lock = threading.Lock()
    stop_monitor = threading.Event()
    active: dict[str, str] = {}

    def report(message: str) -> None:
        line = f"[{utc_now()}] {message}"
        with log_lock:
            print(line, flush=True)
            log_handle.write(line + "\n")

    def save_status() -> None:
        with state_lock:
            payload = {
                "updated_at": utc_now(),
                "seed_index": SEED_INDEX,
                "seed": SEED,
                "slots": list(SLOTS),
                "concurrency": CONCURRENCY,
                "active": dict(active),
                "completed": sum(job["status"] == "completed" for job in jobs),
                "failed": sum(job["status"] == "failed" for job in jobs),
                "total": len(jobs),
                "jobs": jobs,
            }
            temp = status_path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(payload, indent=2) + "\n")
            temp.replace(status_path)

    def summarize() -> None:
        with summary_lock:
            result = subprocess.run(
                [str(PYTHON), str(HERE / "summarize.py"), "--root", str(ROOT)],
                cwd=REPO,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            report(f"summary refresh exited {result.returncode}")

    def monitor() -> None:
        while not stop_monitor.wait(300):
            with state_lock:
                completed = sum(job["status"] == "completed" for job in jobs)
                failed = sum(job["status"] == "failed" for job in jobs)
                active_now = dict(active)
            report(
                f"5-minute monitor: Qwen3.8 seed1234 {completed}/{len(jobs)} "
                f"completed, failed={failed}, active={active_now or '-'}"
            )
            for command in (
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
                    "--format=csv,noheader",
                ],
                ["chg", "status"],
            ):
                try:
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    report(result.stdout.strip() or result.stderr.strip())
                except Exception as exc:
                    report(f"monitor command failed ({command[0]}): {exc}")

    # Reconcile completed work from the audit events, since the prior serial
    # controller is being retired while its first evaluator remains in flight.
    for job in jobs:
        if run_is_complete(job["case_id"]):
            job.update(status="completed", attempts=1, finished_at=utc_now())
        elif job["case_id"] == CURRENT_CASE and pid_is_current_evaluation(
            CURRENT_EVALUATE_PID, CURRENT_CASE
        ):
            job.update(
                status="running",
                attempts=1,
                started_at="2026-10-02T19:33:10.558195+00:00",
                log=str(ROOT / "logs" / "evaluation" / f"{CURRENT_CASE}-seed1.log"),
            )
            active["existing-slot-7"] = CURRENT_CASE

    save_status()
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    def watch_existing() -> None:
        job = next(item for item in jobs if item["case_id"] == CURRENT_CASE)
        if job["status"] != "running":
            return
        while pid_is_current_evaluation(CURRENT_EVALUATE_PID, CURRENT_CASE):
            time.sleep(20)
        success = run_is_complete(CURRENT_CASE)
        if not success:
            report(f"existing case {CURRENT_CASE} ended without all task results; retrying")
            for attempt in range(1, 4):
                log_path = (
                    ROOT
                    / "logs"
                    / "evaluation"
                    / f"{CURRENT_CASE}-seed1-recovery-attempt{attempt}.log"
                )
                command = [
                    str(PYTHON),
                    "examples/imatrix/ablation/evaluate.py",
                    "--manifest",
                    "examples/imatrix/ablation/manifest.json",
                    "--model-id",
                    MODEL_ID,
                    "--checkpoint",
                    job["checkpoint"],
                    "--case-id",
                    CURRENT_CASE,
                    "--run-root",
                    str(ROOT),
                    "--port",
                    "8104",
                    "--concurrency",
                    str(CONCURRENCY),
                    "--seed-index",
                    str(SEED_INDEX),
                ]
                job["attempts"] = attempt + 1
                with log_path.open("w") as output:
                    output.write("COMMAND: " + " ".join(command) + "\n")
                    output.flush()
                    result = subprocess.run(
                        command,
                        cwd=REPO,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                success = result.returncode == 0 and run_is_complete(CURRENT_CASE)
                if success:
                    job["log"] = str(log_path)
                    break
                report(
                    f"recovery attempt {attempt}/3 failed for {CURRENT_CASE} "
                    f"(rc={result.returncode}); log={log_path}"
                )
        with state_lock:
            job["status"] = "completed" if success else "failed"
            job["finished_at"] = utc_now()
            active.pop("existing-slot-7", None)
        save_status()
        summarize()
        report(f"reconciled existing case {CURRENT_CASE}: {job['status']}")

    watcher = threading.Thread(target=watch_existing, daemon=True)
    watcher.start()

    work_lock = threading.Lock()
    pending = [job for job in jobs if job["status"] == "pending"]
    next_job = 0
    failures = 0
    failure_lock = threading.Lock()

    def worker(slot: int) -> None:
        nonlocal next_job, failures
        while True:
            with work_lock:
                if next_job >= len(pending):
                    return
                job = pending[next_job]
                next_job += 1
            with state_lock:
                job.update(
                    status="running",
                    slot=slot,
                    concurrency=CONCURRENCY,
                    started_at=utc_now(),
                )
                active[f"slot-{slot}"] = job["case_id"]
            save_status()

            success = False
            last_log = ""
            for attempt in range(1, 4):
                job["attempts"] = attempt
                last_log = str(
                    ROOT
                    / "logs"
                    / "evaluation"
                    / f"{job['case_id']}-seed1-parallel-attempt{attempt}.log"
                )
                command = [
                    str(PYTHON),
                    "examples/imatrix/ablation/evaluate.py",
                    "--manifest",
                    "examples/imatrix/ablation/manifest.json",
                    "--model-id",
                    MODEL_ID,
                    "--checkpoint",
                    job["checkpoint"],
                    "--case-id",
                    job["case_id"],
                    "--run-root",
                    str(ROOT),
                    "--port",
                    str(8100 + slot),
                    "--concurrency",
                    str(CONCURRENCY),
                    "--seed-index",
                    str(SEED_INDEX),
                ]
                with Path(last_log).open("w") as output:
                    output.write("COMMAND: " + " ".join(command) + "\n")
                    output.flush()
                    result = subprocess.run(
                        command,
                        cwd=REPO,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                success = result.returncode == 0 and run_is_complete(job["case_id"])
                if success:
                    report(f"[slot {slot}] seed1234 completed {job['case_id']}")
                    break
                report(
                    f"[slot {slot}] attempt {attempt}/3 failed for {job['case_id']} "
                    f"(rc={result.returncode}); log={last_log}"
                )

            with state_lock:
                job.update(
                    status="completed" if success else "failed",
                    log=last_log,
                    finished_at=utc_now(),
                )
                active.pop(f"slot-{slot}", None)
            save_status()
            summarize()
            if not success:
                with failure_lock:
                    failures += 1

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(SLOTS)) as pool:
            list(pool.map(worker, SLOTS))
        watcher.join()
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=2)
        save_status()
        summarize()
        log_handle.close()

    return 1 if failures or any(job["status"] == "failed" for job in jobs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
