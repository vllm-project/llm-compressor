#!/usr/bin/env python3
"""Queue Qwen3.8 seed 1234 cases on the remaining GPU slot."""

import json
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = Path("/mnt/data/engine/HDCharles/experiments/w4a16-imatrix-ablation-2026-10-02")
PYTHON = Path("/home/HDCharles/vllm/bin/python")
MODEL_ID = "qwen3.8-27b"
SLOT = 7
CONCURRENCY = 64


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    manifest = json.loads((HERE / "manifest.json").read_text())
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
    status_lock = threading.Lock()
    log_lock = threading.Lock()
    stop_monitor = threading.Event()
    active = ""
    queue_log = ROOT / "logs" / "evaluation" / "seed1-qwen38-queue.log"
    queue_log.parent.mkdir(parents=True, exist_ok=True)
    log_handle = queue_log.open("a", buffering=1)

    def report(message: str) -> None:
        line = f"[{utc_now()}] {message}"
        with log_lock:
            print(line, flush=True)
            log_handle.write(line + "\n")

    def save_status() -> None:
        with status_lock:
            payload = {
                "updated_at": utc_now(),
                "seed_index": 0,
                "seed": 1234,
                "slot": SLOT,
                "concurrency": CONCURRENCY,
                "active": active,
                "completed": sum(job["status"] == "completed" for job in jobs),
                "failed": sum(job["status"] == "failed" for job in jobs),
                "total": len(jobs),
                "jobs": jobs,
            }
            temp = status_path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(payload, indent=2) + "\n")
            temp.replace(status_path)

    def summarize() -> None:
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
            with status_lock:
                completed = sum(job["status"] == "completed" for job in jobs)
                failed = sum(job["status"] == "failed" for job in jobs)
                active_case = active
            report(
                f"5-minute monitor: Qwen3.8 seed1234 {completed}/{len(jobs)} "
                f"completed, failed={failed}, active={active_case or '-'}"
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
                    output = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    report(output.stdout.strip() or output.stderr.strip())
                except Exception as exc:
                    report(f"monitor command failed ({command[0]}): {exc}")

    save_status()
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    failures = 0
    try:
        for job in jobs:
            with status_lock:
                job["status"] = "running"
                job["started_at"] = utc_now()
                active = job["case_id"]
            save_status()
            success = False
            log_path = ROOT / "logs" / "evaluation" / f"{job['case_id']}-seed1.log"
            for attempt in range(1, 4):
                job["attempts"] = attempt
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
                    str(8100 + SLOT),
                    "--concurrency",
                    str(CONCURRENCY),
                    "--seed-index",
                    "0",
                ]
                with log_path.open("a") as output:
                    output.write(
                        f"\nATTEMPT {attempt} COMMAND: "
                        + " ".join(command)
                        + "\n"
                    )
                    output.flush()
                    result = subprocess.run(
                        command,
                        cwd=REPO,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if result.returncode == 0:
                    success = True
                    report(f"seed1234 completed {job['case_id']}")
                    break
                report(
                    f"attempt {attempt}/3 failed for {job['case_id']} "
                    f"(rc={result.returncode}); log={log_path}"
                )
            with status_lock:
                job["status"] = "completed" if success else "failed"
                job["log"] = str(log_path)
                job["finished_at"] = utc_now()
                active = ""
            failures += not success
            save_status()
            summarize()
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=2)
        save_status()
        summarize()
        log_handle.close()

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
