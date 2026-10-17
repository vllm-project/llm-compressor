# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Checkpoint-free L0–L3 tests using the installed LLMC package."""

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--require-quarot-reference",
        action="store_true",
        help="Fail instead of skipping L2 if the ModelSlim source checkout is missing",
    )


@pytest.fixture(scope="module", autouse=True)
def small_thread_pool():
    # Tiny BLAS work is faster and more reproducible without a large thread pool.
    # A fixture also works when this directory is discovered after session start.
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(original)
