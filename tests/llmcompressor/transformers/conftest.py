import os

import pytest


@pytest.fixture(autouse=True)
def run_before_and_after_tests(tmp_path):
    os.environ["TRANSFORMERS_CACHE"] = str(tmp_path / "transformers")
    os.environ["HF_DATASETS_CACHE"] = str(tmp_path / "datasets")
    yield
