from pathlib import Path

import pytest

from tests.examples.utils import (
    copy_and_run_script,
    gen_cmd_fail_message,
    requires_gpu_count,
)


@pytest.fixture
def example_dir() -> str:
    return "examples/big_models_with_sequential_onloading"


@pytest.mark.example
@requires_gpu_count(1)
class TestCompressedInference:
    """
    Tests for examples in the "big_models_with_sequential_onloading" example folder.
    """

    def test_llama33_70b_example_script(self, example_dir: str, tmp_path: Path):
        """
        Test for the "llama3.3_70b.py" script in the folder.
        """
        script_filename = "llama3.3_70b.py"
        command, result = copy_and_run_script(tmp_path, example_dir, script_filename)

        assert result.returncode == 0, gen_cmd_fail_message(command, result)
