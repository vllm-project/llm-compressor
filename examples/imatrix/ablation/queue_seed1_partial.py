#!/usr/bin/env python3
"""Run seed 1234 evaluations for smoke-cleared models while another profile smokes."""

import concurrent.futures
import json
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = Path("/mnt/data/engine/HDCharles/experiments/w4a16-imatrix-ablation-2026-10-02")
MANIFEST = HERE / "manifest.json"
PYTHON = Path("/home/HDCharles/vllm/bin/python")
WORKERS = 7
SMOKE_CLEARED_MODELS = {
    "llama3.1-8b-instruct",
    "qwen3-30b-a3b-instruct-2507",
    "nemotron3.5-lightning-30b-a3b-bf16",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    models = {
        model["id"]: model
        for model in manifest["models"]
        if model["id"] in SMOKE_CLEARED_MODELS
    }
    jobs = []
    for power in manifest["powers"]:
        power_tag = f"p{power:g}".replace(".", "p")
        for objective in manifest["objectives"]:
            for model_id, model in models.items():
                case_id = f"{model_id}__{objective['id']}__{power_tag}"
                jobs.append(
                    {
                        "case_id": case_id,
                        "model_id": model_id,
                        "model": model,
                        "objective": objective["id"],
                        "power": power,
                        "checkpoint": str(
                            ROOT
                            / "checkpoints"
                            / model_id
                            / objective["id"]
                            / power_tag
                        ),
                        "status": "pending",
                        "attempts": 0,
                    }
                )

    status_path = ROOT / "seed1_partial_queue_status.json"
    status_lock = threading.Lock()
    summary_lock = threading.Lock()
    console_lock = threading.Lock()
    monitor_stop = threading.Event()
    active: dict[int, str] = {}
    queue_log = ROOT / "logs" / "evaluation" / "seed1-partial-queue.log"
    queue_log.parent.mkdir(parents=True, exist_ok=True)
    log_handle = queue_log.open("a", buffering=1)

    def save_status() -> None:
        with status_lock:
            payload = {
                "updated_at": utc_now(),
                "seed_index": 0,
                "seed": 1234,
                "workers": WORKERS,
                "queued_models": sorted(models),
                "completed": sum(job["status"] == "completed" for job in jobs),
                "failed": sum(job["status"] == "failed" for job in jobs),
                "total": len(jobs),
                "active": dict(active),
                "jobs": jobs,
            }
            temp = status_path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(payload, indent=2) + "\n")
            temp.replace(status_path)

    def report(message: str) -> None:
        line = f"[{utc_now()}] {message}"
        with console_lock:
            print(line, flush=True)
            log_handle.write(line + "\n")

    def refresh_summary() -> None:
        with summary_lock:
            subprocess.run(
                [str(PYTHON), str(HERE / "summarize.py"), "--root", str(ROOT)],
                cwd=REPO,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )

    def monitor() -> None:
        while not monitor_stop.wait(300):
            completed = sum(job["status"] == "completed" for job in jobs)
            failed = sum(job["status"] == "failed" for job in jobs)
            report(
                f"5-minute monitor: seed1234 {completed}/{len(jobs)} completed, "
                f"failed={failed}, active={dict(active)}"
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

    def run_job(job_index: int, slot: int) -> bool:
        job = jobs[job_index]
        concurrency = 64
        with status_lock:
            job["status"] = "running"
            job["slot"] = slot
            job["concurrency"] = concurrency
            active[slot] = job["case_id"]
        save_status()

        success = False
        for attempt in range(1, 4):
            job["attempts"] = attempt
            log_path = (
                ROOT
                / "logs"
                / "evaluation"
                / f"{job['case_id']}-seed1-attempt{attempt}.log"
            )
            command = [
                str(PYTHON),
                "examples/imatrix/ablation/evaluate.py",
                "--manifest",
                "examples/imatrix/ablation/manifest.json",
                "--model-id",
                job["model_id"],
                "--checkpoint",
                str(job["checkpoint"]),
                "--case-id",
                job["case_id"],
                "--run-root",
                str(ROOT),
                "--port",
                str(8100 + slot),
                "--concurrency",
                str(concurrency),
                "--seed-index",
                "0",
            ]
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
            if result.returncode == 0:
                success = True
                report(f"[slot {slot}] seed1234 completed {job['case_id']}")
                break
            report(
                f"[slot {slot}] attempt {attempt}/3 failed for {job['case_id']} "
                f"(rc={result.returncode}); log={log_path}"
            )

        with status_lock:
            job["status"] = "completed" if success else "failed"
            job["log"] = str(log_path)
            job["finished_at"] = utc_now()
            active.pop(slot, None)
        save_status()
        refresh_summary()
        return success

    save_status()
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    failures = 0
    next_job = 0
    queue_lock = threading.Lock()
    failure_lock = threading.Lock()

    def worker(slot: int) -> None:
        nonlocal failures, next_job
        while True:
            with queue_lock:
                if next_job >= len(jobs):
                    return
                job_index = next_job
                next_job += 1
            if not run_job(job_index, slot):
                with failure_lock:
                    failures += 1

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(worker, range(WORKERS)))
    finally:
        monitor_stop.set()
        monitor_thread.join(timeout=2)
        save_status()
        refresh_summary()
        log_handle.close()

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
