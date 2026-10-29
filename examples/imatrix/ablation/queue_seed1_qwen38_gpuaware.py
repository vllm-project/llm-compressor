#!/usr/bin/env python3
"""Continue Qwen3.8 seed 1234 evaluations across currently free GPUs."""

import concurrent.futures
import json
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
SLOTS = tuple(range(5, 13))
CONCURRENCY = 64
TASKS = {"gsm8k_platinum_cot_llama", "gpqa_diamond", "math_500"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_args(pid: int) -> list[str]:
    try:
        return [
            value.decode(errors="replace")
            for value in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if value
        ]
    except OSError:
        return []


def find_active_evaluations() -> dict[str, tuple[int, int]]:
    active = {}
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            pid = int(cmdline.parent.name)
        except ValueError:
            continue
        args = process_args(pid)
        if (
            not args
            or not any(arg.endswith("examples/imatrix/ablation/evaluate.py") for arg in args)
            or "--model-id" not in args
            or args[args.index("--model-id") + 1] != MODEL_ID
            or "--seed-index" not in args
            or args[args.index("--seed-index") + 1] != str(SEED_INDEX)
            or "--case-id" not in args
            or "--port" not in args
        ):
            continue
        case_id = args[args.index("--case-id") + 1]
        slot = int(args[args.index("--port") + 1]) - 8100
        active[case_id] = (pid, slot)
    return active


def pid_is_evaluating(pid: int, case_id: str) -> bool:
    args = process_args(pid)
    return "examples/imatrix/ablation/evaluate.py" in " ".join(args) and case_id in args


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


def free_gpu_count() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode:
        return 0
    try:
        return sum(int(line.strip()) < 1024 for line in result.stdout.splitlines())
    except ValueError:
        return 0


def compute_pids() -> set[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    pids = set()
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            try:
                pids.add(int(line.strip()))
            except ValueError:
                pass
    return pids


def descendants(pid: int) -> set[int]:
    found = set()
    pending = [pid]
    while pending:
        parent = pending.pop()
        try:
            children = Path(f"/proc/{parent}/task/{parent}/children").read_text().split()
        except OSError:
            continue
        for child in children:
            try:
                child_pid = int(child)
            except ValueError:
                continue
            if child_pid not in found:
                found.add(child_pid)
                pending.append(child_pid)
    return found


def main() -> int:
    manifest_path = HERE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    status_path = ROOT / "seed1_qwen38_queue_status.json"
    queue_log = ROOT / "logs" / "evaluation" / "seed1-qwen38-queue.log"
    queue_log.parent.mkdir(parents=True, exist_ok=True)
    log_handle = queue_log.open("a", buffering=1)
    log_lock = threading.Lock()
    state_lock = threading.Lock()
    work_lock = threading.Lock()
    launch_lock = threading.Lock()
    summary_lock = threading.Lock()
    monitor_stop = threading.Event()
    active: dict[str, str] = {}
    launch_reservations = 0

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

    try:
        previous = json.loads(status_path.read_text())
        previous_jobs = {job["case_id"]: job for job in previous.get("jobs", [])}
    except (OSError, json.JSONDecodeError):
        previous_jobs = {}
    active_processes = find_active_evaluations()
    existing_by_slot = {}
    for job in jobs:
        case_id = job["case_id"]
        old = previous_jobs.get(case_id, {})
        for key in ("attempts", "started_at", "finished_at", "log", "concurrency"):
            if key in old:
                job[key] = old[key]
        if run_is_complete(case_id):
            job.update(status="completed", finished_at=job.get("finished_at", utc_now()))
            continue
        if case_id in active_processes:
            pid, slot = active_processes[case_id]
            job.update(
                status="running",
                attempts=max(int(job.get("attempts", 0)), 1),
                slot=slot,
                concurrency=CONCURRENCY,
            )
            active[str(slot)] = case_id
            existing_by_slot[slot] = (job, pid)
        elif old.get("status") == "failed":
            job.update(status="pending", attempts=0)

    save_status()

    def monitor() -> None:
        while not monitor_stop.wait(300):
            with state_lock:
                completed = sum(job["status"] == "completed" for job in jobs)
                failed = sum(job["status"] == "failed" for job in jobs)
                active_now = dict(active)
            report(
                f"5-minute monitor: Qwen3.8 seed1234 {completed}/{len(jobs)} "
                f"completed, failed={failed}, active={active_now or '-'}, "
                f"free_gpus={free_gpu_count()}"
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

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    def claim_gpu() -> None:
        nonlocal launch_reservations
        while True:
            free = free_gpu_count()
            with launch_lock:
                if free > launch_reservations:
                    launch_reservations += 1
                    return
            time.sleep(10)

    def wait_for_assignment(process: subprocess.Popen, case_id: str) -> None:
        nonlocal launch_reservations
        while process.poll() is None:
            if descendants(process.pid) & compute_pids():
                with launch_lock:
                    launch_reservations -= 1
                report(f"GPU assigned to {case_id}; active Qwen3.8 model load complete")
                return
            time.sleep(5)
        with launch_lock:
            launch_reservations -= 1

    def run_job(job: dict, slot: int) -> bool:
        start_attempt = min(max(int(job.get("attempts", 0)) + 1, 1), 3)
        for attempt in range(start_attempt, 4):
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
            with state_lock:
                job.update(
                    status="running",
                    attempts=attempt,
                    slot=slot,
                    concurrency=CONCURRENCY,
                    started_at=job.get("started_at", utc_now()),
                )
                active[str(slot)] = job["case_id"]
            save_status()
            report(f"[slot {slot}] starting {job['case_id']} attempt {attempt}")
            claim_gpu()
            claim_held = True
            try:
                with log_path.open("w") as output:
                    output.write("COMMAND: " + " ".join(command) + "\n")
                    output.flush()
                    process = subprocess.Popen(
                        command,
                        cwd=REPO,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                    )
                    wait_for_assignment(process, job["case_id"])
                    claim_held = False
                    returncode = process.wait()
            except Exception as exc:
                if claim_held:
                    with launch_lock:
                        launch_reservations -= 1
                returncode = 1
                report(f"launch exception for {job['case_id']}: {exc}")

            if returncode == 0 and run_is_complete(job["case_id"]):
                with state_lock:
                    job.update(status="completed", log=str(log_path), finished_at=utc_now())
                    active.pop(str(slot), None)
                save_status()
                summarize()
                report(f"[slot {slot}] seed1234 completed {job['case_id']}")
                return True
            report(
                f"[slot {slot}] attempt {attempt}/3 incomplete for {job['case_id']} "
                f"(rc={returncode}); log={log_path}"
            )

        with state_lock:
            job.update(status="failed", log=str(log_path), finished_at=utc_now())
            active.pop(str(slot), None)
        save_status()
        summarize()
        return False

    pending = [job for job in jobs if job["status"] == "pending"]
    next_job = 0
    failed_lock = threading.Lock()
    failures = 0

    def worker(slot: int) -> None:
        nonlocal next_job, failures
        external = existing_by_slot.get(slot)
        if external:
            job, pid = external
            while pid_is_evaluating(pid, job["case_id"]):
                time.sleep(20)
            if run_is_complete(job["case_id"]):
                with state_lock:
                    job.update(status="completed", finished_at=utc_now())
                    active.pop(str(slot), None)
                save_status()
                summarize()
                report(f"[slot {slot}] reconciled completed {job['case_id']}")
            else:
                report(f"[slot {slot}] existing process ended incomplete; recovering {job['case_id']}")
                if not run_job(job, slot):
                    with failed_lock:
                        failures += 1

        while True:
            with work_lock:
                if next_job >= len(pending):
                    return
                job = pending[next_job]
                next_job += 1
            if not run_job(job, slot):
                with failed_lock:
                    failures += 1

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(SLOTS)) as pool:
            list(pool.map(worker, SLOTS))
    finally:
        monitor_stop.set()
        monitor_thread.join(timeout=2)
        save_status()
        summarize()
        log_handle.close()

    return 1 if failures or any(job["status"] == "failed" for job in jobs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
