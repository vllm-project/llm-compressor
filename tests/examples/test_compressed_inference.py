import shutil
import sys
from pathlib import Path

import pytest

from tests.examples.utils import gen_cmd_fail_message, requires_gpu, requires_torch
from tests.testing_utils import run_cli_command


@pytest.fixture
def example_dir() -> str:
    return "examples/compressed_inference"


@pytest.mark.example
@requires_gpu
@requires_torch
class TestCompressedInference:
    """
    Tests for examples in the "compressed_inference" example folder.
    """

    def test_fp8_example_script(self, example_dir: str, tmp_path: Path):
        """
        Test for the "fp8_compressed_inference.py" script in the folder.
        """
        shutil.copytree(Path.cwd() / example_dir, tmp_path / example_dir)

        command = [sys.executable, "fp8_compressed_inference.py"]
        result = run_cli_command(command, cwd=tmp_path / example_dir)

        assert result.returncode == 0, gen_cmd_fail_message(command, result)
