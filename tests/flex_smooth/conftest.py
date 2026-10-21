# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def oracle():
    from .oracle import FlexOracle

    root = Path(
        os.environ.get(
            "MODELSLIM_SOURCE",
            Path(__file__).resolve().parents[4] / "reference/msmodelslim",
        )
    )
    if not root.joinpath(
        "msmodelslim/processor/anti_outlier/flex_smooth/api.py"
    ).is_file():
        if os.environ.get("FLEXSMOOTH_REQUIRE_REFERENCE") == "1":
            pytest.fail(f"Missing ModelSlim source: {root}")
        pytest.skip("Set MODELSLIM_SOURCE to the separately licensed source checkout")
    return FlexOracle(root)
