"""
Thin DDP compression runner, invoked via torchrun for any DDP test.

Usage:
    torchrun --nproc_per_node N tests/e2e/run_oneshot_ddp.py <config_json>

Arguments:
    config_json       JSON-serialized BaseTestConfig / TestConfig
    --save-compressed Save with save_compressed=True and write recipe.yaml
                      (required for e2e vLLM tests; omit for lm-eval tests)
"""

import json
import os
import sys

# Ensure the repo root is on sys.path so `tests` is importable when torchrun
# spawns this script directly (it does not inherit the pytest sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from compressed_tensors.offload import init_dist

from tests.e2e.e2e_utils import run_oneshot_ddp

if __name__ == "__main__":
    config = json.loads(sys.argv[1])

    init_dist()
    run_oneshot_ddp(
        config, save_compressed=config.get("save_compressed", False)
    )
