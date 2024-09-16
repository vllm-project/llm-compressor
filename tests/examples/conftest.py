import os

import pytest

from tests.examples.utils import get_gpu_batches


@pytest.fixture(autouse=True)
def set_visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    gpu_count = int(os.getenv("GPU_COUNT", "0"))
    worker_count = int(os.getenv("PYTEST_XDIST_WORKER_COUNT", "0"))
    worker_name = os.getenv("PYTEST_XDIST_WORKER", "")
    if not gpu_count or not worker_count or worker_name == "master":
        # no-op if: (a) GPU_COUNT isn't set, (b) not using pytest-xdist, or
        # (c) not running on pytest-xdist worker
        return

    worker_id = int(worker_name[2:])
    gpu_groups = get_gpu_batches(gpu_count, worker_count)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu_groups[worker_id])
