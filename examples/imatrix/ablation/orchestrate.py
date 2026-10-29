#!/usr/bin/env python3
"""Run the full iMatrix × norm × model matrix with periodic resource checks."""

import argparse
import concurrent.futures
import json
import re
import shlex
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RHDEV_PYTHON = Path("/home/HDCharles/rhdev/bin/python")
VLLM_PYTHON = Path("/home/HDCharles/vllm/bin/python")
DEFAULT_SLOT_COUNT = 8
MONITOR_INTERVAL = 300


def _power_tag(power: float) -> str:
    return f"p{power:g}".replace(".", "p")


def make_cases(manifest: dict) -> list[dict]:
    all_cases = []
    for model in manifest["models"]:
        for power in manifest["powers"]:
            for objective in manifest["objectives"]:
                case_id = f"{model['id']}__{objective['id']}__{_power_tag(power)}"
                all_cases.append(
                    {
                        "case_id": case_id,
                        "model_id": model["id"],
                        "objective_id": objective["id"],
                        "power": power,
                        "is_smoke_baseline": (
                            power == 2.0 and objective["id"] == "importance-error-power"
                        ),
                    }
                )
    baselines = [case for case in all_cases if case["is_smoke_baseline"]]
    remaining = [case for case in all_cases if not case["is_smoke_baseline"]]
    # Start every model's baseline before filling in the rest of the matrix.
    return baselines + remaining


class Experiment:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = args.root
        self.slot_count = args.workers
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in (
            "checkpoints",
            "logs/quantization",
            "logs/evaluation",
            "evaluations",
        ):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        snapshot_dir = self.root / "scripts"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for script in HERE.glob("*.py"):
            shutil.copyfile(script, snapshot_dir / script.name)
        shutil.copyfile(args.manifest, snapshot_dir / "manifest.json")
        shutil.copyfile(HERE / "README.md", snapshot_dir / "README.md")
        source_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        source_diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=REPO,
            capture_output=True,
            check=True,
        ).stdout
        (self.root / "llmcompressor-working-tree.patch").write_bytes(source_diff)
        self.manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(args.manifest.read_text())
        self.manifest["source_manifest"] = str(args.manifest.resolve())
        self.manifest["started_at"] = datetime.now(timezone.utc).isoformat()
        self.manifest["source_revision"] = source_revision
        self.manifest["scripts_snapshot"] = str(snapshot_dir)
        self.manifest["evaluation"] = {
            "vllm_venv": str(VLLM_PYTHON.parent.parent),
            "quantization_venv": str(RHDEV_PYTHON.parent.parent),
            "eval_agent": (
                "eval-agent registry; tasks: gsm8k_platinum_cot_llama, "
                "gpqa_diamond, math_500"
            ),
            "smoke_samples_per_task": 100,
            "seed_repetitions": 3,
            "seed_schedule": "global sequential passes; finish each seed across all checkpoints before the next",
        }
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2) + "\n")
        self.cases = make_cases(self.manifest)
        self.models = {model["id"]: model for model in self.manifest["models"]}
        self.status_path = self.root / "status.json"
        self.state_lock = threading.Lock()
        self.summary_lock = threading.Lock()
        self.state = {
            case["case_id"]: {
                **case,
                "quantization": "pending",
                "evaluation": "pending",
                "active_stage": None,
                "active_slot": None,
            }
            for case in self.cases
        }
        self.model_concurrency = {
            model["id"]: (
                args.reasoning_concurrency
                if model["profile"] == "reasoning"
                else args.instruct_concurrency
            )
            for model in self.manifest["models"]
        }
        self._save_state()
        self._monitor_stop = threading.Event()
        self._monitor_thread = threading.Thread(target=self._monitor, daemon=True)

    def _save_state(self):
        with self.state_lock:
            payload = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "model_concurrency": self.model_concurrency,
                "cases": self.state,
            }
            tmp = self.status_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2) + "\n")
            tmp.replace(self.status_path)

    def _set(self, case_id: str, **values):
        with self.state_lock:
            self.state[case_id].update(values)
        self._save_state()

    def _refresh_summary(self):
        with self.summary_lock:
            subprocess.run(
                [
                    str(VLLM_PYTHON),
                    str(HERE / "summarize.py"),
                    "--root",
                    str(self.root),
                ],
                check=False,
            )

    def checkpoint_path(self, case: dict) -> Path:
        return (
            self.root
            / "checkpoints"
            / case["model_id"]
            / case["objective_id"]
            / _power_tag(case["power"])
        )

    def quantized(self, case: dict) -> bool:
        metadata = self.checkpoint_path(case) / "ablation_metadata.json"
        if not metadata.exists():
            return False
        try:
            saved = json.loads(metadata.read_text())
            return (
                saved.get("model_id") == self.models[case["model_id"]]["hf_id"]
                and saved.get("power") == case["power"]
                and saved.get("group_size") == self.manifest["group_size"]
                and saved.get("objective", {}).get("id") == case["objective_id"]
                and saved.get("triton_grid_search_calls", 0) > 0
            )
        except (OSError, ValueError):
            return False

    def _base_command(
        self, executable: Path, script: Path, reserve_gpu: bool = True
    ) -> list[str]:
        command = ["chg", "run", "--gpus", "1", "--", "env"] if reserve_gpu else ["env"]
        if executable == RHDEV_PYTHON:
            command.append(f"PYTHONPATH={REPO / 'src'}")
        command.extend(["TOKENIZERS_PARALLELISM=false", str(executable), str(script)])
        return command

    def _run_attempt(self, command: list[str], log_path: Path) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log_file:
            log_file.write("$ " + shlex.join(command) + "\n")
            log_file.flush()
            proc = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)
        return proc.returncode

    def _quantize_case(self, case: dict, slot: int) -> bool:
        if self.quantized(case):
            self._set(case["case_id"], quantization="completed", active_stage=None)
            return True
        model = self.models[case["model_id"]]
        objective = next(
            o for o in self.manifest["objectives"] if o["id"] == case["objective_id"]
        )
        command = self._base_command(RHDEV_PYTHON, HERE / "quantize.py") + [
            "--manifest",
            str(self.manifest_path),
            "--model-id",
            case["model_id"],
            "--objective-id",
            case["objective_id"],
            "--power",
            str(case["power"]),
            "--output",
            str(self.checkpoint_path(case)),
        ]
        self._set(
            case["case_id"],
            quantization="running",
            active_stage="quantization",
            active_slot=slot,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        print(
            f"[slot {slot}] quantizing {case['case_id']} "
            f"({model['hf_id']}; {objective['description']})",
            flush=True,
        )
        for attempt in range(1, 4):
            log_path = (
                self.root
                / "logs"
                / "quantization"
                / f"{case['case_id']}.attempt{attempt}.log"
            )
            rc = self._run_attempt(command, log_path)
            if rc == 0 and self.quantized(case):
                self._set(
                    case["case_id"],
                    quantization="completed",
                    active_stage=None,
                    quantization_log=str(log_path),
                )
                print(f"[slot {slot}] quantized {case['case_id']}", flush=True)
                return True
            print(
                f"[slot {slot}] quantization attempt {attempt}/3 failed "
                f"({case['case_id']}, rc={rc}); log={log_path}",
                flush=True,
            )
        self._set(case["case_id"], quantization="failed", active_stage=None)
        return False

    def _run_eval(
        self,
        case: dict,
        slot: int,
        smoke_only: bool,
        concurrency: int,
        suffix: str = "",
        log_suffix: str = "",
        seed_index: int | None = None,
    ) -> tuple[int, Path]:
        eval_case_id = case["case_id"] + suffix
        command = self._base_command(
            VLLM_PYTHON, HERE / "evaluate.py", reserve_gpu=False
        ) + [
            "--manifest",
            str(self.manifest_path),
            "--model-id",
            case["model_id"],
            "--checkpoint",
            str(self.checkpoint_path(case)),
            "--case-id",
            eval_case_id,
            "--run-root",
            str(self.root),
            "--port",
            str(8100 + slot),
            "--concurrency",
            str(concurrency),
        ]
        if smoke_only:
            command.append("--smoke-only")
        elif seed_index is not None:
            command.extend(["--seed-index", str(seed_index)])
        log_path = self.root / "logs" / "evaluation" / f"{eval_case_id}{log_suffix}.log"
        rc = self._run_attempt(command, log_path)
        return rc, log_path

    @staticmethod
    def _peak_kv_usage(log_path: Path) -> float | None:
        text = log_path.read_text(errors="replace")
        values = [
            float(value)
            for value in re.findall(r"GPU KV cache usage:\s*([0-9.]+)%", text)
        ]
        return max(values) if values else None

    def _reuse_preflight_smoke(self, case: dict) -> bool:
        """Reuse the manually started Llama preflight when it fully passes."""
        if case["model_id"] != "llama3.1-8b-instruct":
            return False
        run_dir = (
            self.root
            / "evaluations"
            / f"preflight-{case['model_id']}-retry-smoke"
        )
        manifest_path = run_dir / "manifest.json"
        events_path = run_dir / "events.jsonl"
        server_log = run_dir / "logs" / "vllm_server.log"
        if not manifest_path.exists() or not events_path.exists() or not server_log.exists():
            return False

        try:
            manifest = json.loads(manifest_path.read_text())
            events = [
                json.loads(line)
                for line in events_path.read_text().splitlines()
                if line.strip()
            ]
        except (OSError, ValueError):
            return False

        expected_tasks = set(manifest.get("task_names") or [])
        passed_tasks = {
            task
            for event in events
            if event.get("event") == "smoke_passed"
            for task in event.get("tasks", [])
        }
        run_completed = any(
            event.get("event") == "run_completed"
            and event.get("status") == "completed"
            for event in events
        )
        if (
            manifest.get("model") != str(self.checkpoint_path(case))
            or not manifest.get("smoke_only")
            or manifest.get("checkpoint_group_size") != self.manifest["group_size"]
            or not expected_tasks
            or passed_tasks != expected_tasks
            or not run_completed
        ):
            return False

        peak = self._peak_kv_usage(server_log)
        if peak is None or peak > 85:
            return False

        concurrency = int(manifest.get("num_concurrent", 0))
        if not concurrency:
            return False
        self.model_concurrency[case["model_id"]] = concurrency
        self._set(
            case["case_id"],
            evaluation="smoke_passed",
            smoke_peak_kv_percent=peak,
            concurrency=concurrency,
            active_stage=None,
            smoke_log=str(run_dir),
        )
        print(
            f"Reusing completed Llama preflight smoke, peak KV={peak:.1f}%; "
            f"concurrency={concurrency}",
            flush=True,
        )
        return True

    def _smoke_case(self, case: dict, slot: int) -> bool:
        profile = self.models[case["model_id"]]["profile"]
        if self._reuse_preflight_smoke(case):
            return True
        concurrency = self.model_concurrency[case["model_id"]]
        initial_concurrency = concurrency
        adjusted_for_cache_pressure = False
        self._set(
            case["case_id"],
            evaluation="smoke_running",
            active_stage="smoke",
            active_slot=slot,
        )
        for attempt in range(1, 4):
            suffix = f"-smoke-attempt{attempt}"
            rc, log_path = self._run_eval(case, slot, True, concurrency, suffix)
            eval_dir = self.root / "evaluations" / f"{case['case_id']}{suffix}-smoke"
            server_log = eval_dir / "logs" / "vllm_server.log"
            peak = self._peak_kv_usage(server_log)
            if rc == 0:
                if peak is not None and peak > 85 and concurrency > 1:
                    adjusted_for_cache_pressure = True
                    concurrency = max(1, concurrency // 2)
                    self.model_concurrency[case["model_id"]] = concurrency
                    self._save_state()
                    print(
                        f"[slot {slot}] {profile} smoke peak KV {peak:.1f}%; "
                        f"retrying at concurrency {concurrency}",
                        flush=True,
                    )
                    continue
                if (
                    peak is not None
                    and peak < 50
                    and not adjusted_for_cache_pressure
                    and concurrency < min(initial_concurrency * 2, 256)
                ):
                    adjusted_for_cache_pressure = True
                    concurrency = min(initial_concurrency * 2, 256)
                    self.model_concurrency[case["model_id"]] = concurrency
                    self._save_state()
                    print(
                        f"[slot {slot}] {profile} smoke peak KV {peak:.1f}%; "
                        f"increasing concurrency to {concurrency} and checking again",
                        flush=True,
                    )
                    continue
                self.model_concurrency[case["model_id"]] = concurrency
                self._set(
                    case["case_id"],
                    evaluation="smoke_passed",
                    smoke_peak_kv_percent=peak,
                    concurrency=concurrency,
                    active_stage=None,
                )
                print(
                    f"[slot {slot}] smoke passed for {profile}, peak KV={peak}; "
                    f"concurrency={concurrency}",
                    flush=True,
                )
                return True
            concurrency = max(1, concurrency // 2)
            self.model_concurrency[case["model_id"]] = concurrency
            self._save_state()
            print(
                f"[slot {slot}] smoke attempt {attempt}/3 failed for "
                f"{case['model_id']}; retry concurrency={concurrency}; "
                f"log={log_path}",
                flush=True,
            )
        self._set(case["case_id"], evaluation="smoke_failed", active_stage=None)
        return False

    def _quantization_phase(self) -> set[str]:
        failures: set[str] = set()
        pending = [case for case in self.cases if not self.quantized(case)]
        print(
            f"Starting quantization: {len(pending)} pending checkpoints "
            f"on {self.slot_count} GPUs",
            flush=True,
        )
        for case in self.cases:
            if self.quantized(case):
                self._set(case["case_id"], quantization="completed", active_stage=None)

        pending_iter = iter(pending)
        pending_lock = threading.Lock()

        def worker(slot):
            worker_failures = set()
            while True:
                with pending_lock:
                    case = next(pending_iter, None)
                if case is None:
                    return worker_failures
                if not self._quantize_case(case, slot):
                    worker_failures.add(case["case_id"])

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.slot_count) as pool:
            futures = [pool.submit(worker, slot) for slot in range(self.slot_count)]
            for future in concurrent.futures.as_completed(futures):
                failures.update(future.result())
        return failures

    def _smoke_phase(self, quant_failures: set[str]) -> set[str]:
        smoke_failures: set[str] = set()
        baselines = [
            case
            for case in self.cases
            if case["is_smoke_baseline"] and case["case_id"] not in quant_failures
        ]
        print(
            f"Starting eval-agent smoke checks: {len(baselines)} model profiles",
            flush=True,
        )
        if not baselines:
            return smoke_failures

        def worker(slot, case):
            return case["case_id"], self._smoke_case(case, slot)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self.slot_count, len(baselines))
        ) as pool:
            futures = [
                pool.submit(worker, index, case) for index, case in enumerate(baselines)
            ]
            for future in concurrent.futures.as_completed(futures):
                case_id, success = future.result()
                if not success:
                    smoke_failures.add(case_id)
        return smoke_failures

    def _full_eval_case(self, case: dict, slot: int, seed_index: int) -> bool:
        concurrency = self.model_concurrency[case["model_id"]]
        self._set(
            case["case_id"],
            evaluation="running",
            active_stage="evaluation",
            active_slot=slot,
            concurrency=concurrency,
            evaluation_seed_index=seed_index,
        )
        print(
            f"[slot {slot}] evaluating seed {seed_index + 1}/3 for "
            f"{case['case_id']} at concurrency={concurrency}",
            flush=True,
        )
        for attempt in range(1, 4):
            rc, log_path = self._run_eval(
                case,
                slot,
                False,
                concurrency,
                log_suffix=f"-seed{seed_index + 1}-attempt{attempt}",
                seed_index=seed_index,
            )
            if rc == 0:
                with self.state_lock:
                    completed = self.state[case["case_id"]].setdefault(
                        "evaluation_seeds_completed", []
                    )
                    if seed_index not in completed:
                        completed.append(seed_index)
                        completed.sort()
                self._set(
                    case["case_id"],
                    evaluation=(
                        "completed" if seed_index == 2 else "running"
                    ),
                    active_stage=None,
                    evaluation_log=str(log_path),
                    last_completed_seed_index=seed_index,
                )
                self._refresh_summary()
                print(
                    f"[slot {slot}] seed {seed_index + 1}/3 completed for "
                    f"{case['case_id']}",
                    flush=True,
                )
                return True
            print(
                f"[slot {slot}] eval-agent attempt {attempt}/3 failed for "
                f"{case['case_id']}; resuming completed tasks; log={log_path}",
                flush=True,
            )
            concurrency = max(1, concurrency // 2)
            with self.state_lock:
                self.model_concurrency[case["model_id"]] = concurrency
            self._save_state()
        self._set(case["case_id"], evaluation="failed", active_stage=None)
        return False

    def _evaluation_phase(
        self, quant_failures: set[str], smoke_failures: set[str]
    ) -> set[str]:
        failures: set[str] = set()
        skip_models = {
            self.models[case["model_id"]]["id"]
            for case in self.cases
            if case["case_id"] in smoke_failures
        }
        skip_models.update(
            self.models[case["model_id"]]["id"]
            for case in self.cases
            if case["is_smoke_baseline"] and case["case_id"] in quant_failures
        )
        # Evaluate the baseline checkpoint too; its smoke run does not replace
        # the full-size benchmark results.
        pending = [
            case
            for case in self.cases
            if case["case_id"] not in quant_failures
            and case["model_id"] not in skip_models
        ]
        print(
            f"Starting full evaluation: {len(pending)} checkpoints "
            f"on {self.slot_count} GPUs, staged by seed",
            flush=True,
        )
        seed_count = int(self.manifest["evaluation"]["seed_repetitions"])
        for seed_index in range(seed_count):
            print(
                f"Starting global evaluation pass for seed {seed_index + 1}/{seed_count}",
                flush=True,
            )
            pending_iter = iter(pending)
            pending_lock = threading.Lock()

            def worker(slot):
                worker_failures = set()
                while True:
                    with pending_lock:
                        case = next(pending_iter, None)
                    if case is None:
                        return worker_failures
                    if not self._full_eval_case(case, slot, seed_index):
                        worker_failures.add(
                            f"{case['case_id']}@seed{seed_index + 1}"
                        )

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.slot_count
            ) as pool:
                futures = [pool.submit(worker, slot) for slot in range(self.slot_count)]
                for future in concurrent.futures.as_completed(futures):
                    failures.update(future.result())

            # Publish a partial matrix after each complete seed pass, so the
            # first seed is reviewable while later seeds are still running.
            self._refresh_summary()
        return failures

    def _monitor(self):
        while not self._monitor_stop.wait(MONITOR_INTERVAL):
            with self.state_lock:
                states = list(self.state.values())
            active = [
                f"slot{item['active_slot']}:{item['active_stage']}:{item['case_id']}"
                for item in states
                if item.get("active_stage")
            ]
            completed_quant = sum(
                item["quantization"] == "completed" for item in states
            )
            completed_eval = sum(item["evaluation"] == "completed" for item in states)
            completed_by_seed = [
                sum(
                    seed_index in item.get("evaluation_seeds_completed", [])
                    for item in states
                )
                for seed_index in range(
                    int(self.manifest["evaluation"]["seed_repetitions"])
                )
            ]
            print(
                f"\n[5-minute monitor {datetime.now(timezone.utc).isoformat()}] "
                f"quantized={completed_quant}/{len(states)} "
                f"eval_seeds="
                f"{','.join(f'{done}/{len(states)}' for done in completed_by_seed)} "
                f"full_evals={completed_eval}/{len(states)} "
                f"active={active}",
                flush=True,
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
                        command, capture_output=True, text=True, timeout=30
                    ).stdout.strip()
                    print(output, flush=True)
                except Exception as exc:
                    print(f"monitor command failed {command[0]}: {exc}", flush=True)
            for item in states:
                if not item.get("active_stage"):
                    continue
                case_id = item["case_id"]
                stage = item["active_stage"]
                if stage == "quantization":
                    logs = sorted(
                        (self.root / "logs" / "quantization").glob(
                            f"{case_id}.attempt*.log"
                        )
                    )
                elif stage == "smoke":
                    logs = sorted(
                        (self.root / "logs" / "evaluation").glob(
                            f"{case_id}-smoke-attempt*.log"
                        )
                    )
                else:
                    evaluation_logs = self.root / "logs" / "evaluation"
                    logs = list(evaluation_logs.glob(f"{case_id}-seed*-attempt*.log"))
                    if not logs:
                        logs = list(evaluation_logs.glob(f"{case_id}-attempt*.log"))
                logs = sorted(logs, key=lambda path: path.stat().st_mtime)
                if logs:
                    lines = logs[-1].read_text(errors="replace").splitlines()
                    tail = "\n".join(lines[-5:])
                    print(f"--- {case_id} [{stage}] log tail ---\n{tail}", flush=True)
            self._save_state()

    def run(self):
        self._monitor_thread.start()
        quant_failures = self._quantization_phase()
        if self.args.quant_only:
            self._monitor_stop.set()
            self._monitor_thread.join(timeout=2)
            with self.state_lock:
                self.state["_experiment"] = {
                    "quantization_failures": sorted(quant_failures),
                    "smoke_failures": [],
                    "evaluation_failures": [],
                    "quantization_only": True,
                }
            self._save_state()
            print(
                json.dumps(
                    {
                        "experiment_root": str(self.root),
                        "quantization_failures": sorted(quant_failures),
                        "status": str(self.status_path),
                    },
                    indent=2,
                ),
                flush=True,
            )
            return 1 if quant_failures else 0
        smoke_failures = self._smoke_phase(quant_failures)
        eval_failures = self._evaluation_phase(quant_failures, smoke_failures)
        subprocess.run(
            [
                str(VLLM_PYTHON),
                str(HERE / "summarize.py"),
                "--root",
                str(self.root),
            ],
            check=False,
        )
        self._monitor_stop.set()
        self._monitor_thread.join(timeout=2)
        with self.state_lock:
            self.state["_experiment"] = {
                "quantization_failures": sorted(quant_failures),
                "smoke_failures": sorted(smoke_failures),
                "evaluation_failures": sorted(eval_failures),
            }
        self._save_state()
        print(
            json.dumps(
                {
                    "experiment_root": str(self.root),
                    "quantization_failures": sorted(quant_failures),
                    "smoke_failures": sorted(smoke_failures),
                    "evaluation_failures": sorted(eval_failures),
                    "status": str(self.status_path),
                },
                indent=2,
            ),
            flush=True,
        )
        return 1 if quant_failures or smoke_failures or eval_failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/mnt/data/engine/HDCharles/experiments/w4a16-imatrix-ablation-2026-10-02"
        ),
    )
    parser.add_argument("--instruct-concurrency", type=int, default=64)
    parser.add_argument("--reasoning-concurrency", type=int, default=32)
    parser.add_argument("--workers", type=int, default=DEFAULT_SLOT_COUNT)
    parser.add_argument("--quant-only", action="store_true")
    args = parser.parse_args()
    experiment = Experiment(args)
    return experiment.run()


if __name__ == "__main__":
    sys.exit(main())
