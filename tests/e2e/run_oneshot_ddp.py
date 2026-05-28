"""
Thin DDP compression runner, invoked via torchrun for any DDP test.

Usage:
    torchrun --nproc_per_node N tests/e2e/run_oneshot_ddp.py <config_json>

Arguments:
    config_json       JSON-serialized BaseTestConfig / TestConfig
"""

import json
import torch.distributed as dist
import os
import subprocess
import sys
from pathlib import Path


def launch_ddp(num_gpus: int, config: dict) -> None:
    """Launch run_oneshot_ddp via torchrun, streaming output live."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(num_gpus),
        "--log-dir",
        "/tmp/torchrun-logs",
        "--tee",
        "3",
        str(Path(__file__)),
        json.dumps(config),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"DDP oneshot failed (exit {proc.returncode}):\n{''.join(lines)}"
        )


if __name__ == "__main__":
    # Ensure the repo root is on sys.path so `tests` is importable when torchrun
    # spawns this script directly (it does not inherit the pytest sys.path).
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    from compressed_tensors.offload import init_dist

    from tests.e2e.e2e_utils import run_oneshot_ddp

    config = json.loads(sys.argv[1])
    init_dist()
    run_oneshot_ddp(config, save_compressed=config.get("save_compressed", False))
    dist.destroy_process_group()
