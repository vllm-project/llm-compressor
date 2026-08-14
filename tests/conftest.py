# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
from pathlib import Path


def pytest_configure(config):
    # Give each torchrun rank its own basetemp to avoid cleanup race conditions
    rank = os.environ.get("LOCAL_RANK")
    if rank is not None and config.option.basetemp is None:
        run_id = os.environ.get("TORCHELASTIC_RUN_ID", str(os.getpid()))
        config.option.basetemp = tempfile.mkdtemp(
            prefix=f"pytest-torchrun-{run_id}-rank{rank}-",
            dir=Path(tempfile.gettempdir()),
        )
