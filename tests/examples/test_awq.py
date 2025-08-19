from pathlib import Path

import pytest

from tests.examples.utils import (
    copy_and_run_script,
    gen_cmd_fail_message,
    requires_gpu_count,
)


@pytest.fixture
def example_dir() -> str:
    return "examples/awq"


@pytest.mark.example
@requires_gpu_count(1)
class TestAWQ:
    """
    Tests for examples in the "awq" example folder.
    """

    @pytest.mark.parametrize(
        "script_filename",
        [
            "llama_example.py",
            "qwen3_moe_example.py",
        ],
    )
    def test_awq_example_script(
        self, script_filename: str, example_dir: str, tmp_path: Path
    ):
        """
        Tests the example scripts in the folder.
        """
        command, result = copy_and_run_script(tmp_path, example_dir, script_filename)

        assert result.returncode == 0, gen_cmd_fail_message(command, result)
