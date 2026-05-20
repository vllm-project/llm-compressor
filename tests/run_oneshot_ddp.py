"""
Thin DDP compression runner, invoked via torchrun for any DDP test.

Usage:
    torchrun --nproc_per_node N tests/run_oneshot_ddp.py <config_json> [--save-compressed]

Arguments:
    config_json       JSON-serialized BaseTestConfig / TestConfig
    --save-compressed Save with save_compressed=True and write recipe.yaml
                      (required for e2e vLLM tests; omit for lm-eval tests)
"""

import json
import sys

from compressed_tensors.offload import init_dist

from tests.e2e.e2e_utils import run_oneshot_for_e2e_testing_ddp

if __name__ == "__main__":
    config = json.loads(sys.argv[1])

    init_dist()
    run_oneshot_for_e2e_testing_ddp(config, save_compressed=config.get("save_compressed", False))
