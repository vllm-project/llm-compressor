from pathlib import Path

import pytest

from tests.examples.utils import (
    copy_and_run_script,
    gen_cmd_fail_message,
    requires_gpu_count,
)


@pytest.fixture
def example_dir() -> str:
    return "examples/compressed_inference"


@pytest.mark.example
@requires_gpu_count(1)
class TestCompressedInference:
    """
    Tests for examples in the "compressed_inference" example folder.
    """

    def test_fp8_example_script(self, example_dir: str, tmp_path: Path):
        """
        Test for the "fp8_compressed_inference.py" script in the folder.
        """
        script_filename = "fp8_compressed_inference.py"
        command, result = copy_and_run_script(tmp_path, example_dir, script_filename)

        assert result.returncode == 0, gen_cmd_fail_message(command, result)
