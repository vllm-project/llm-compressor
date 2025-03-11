from pathlib import Path

import pytest

from tests.examples.utils import (
    copy_and_run_script,
    gen_cmd_fail_message,
    requires_gpu_count,
)


@pytest.fixture
def example_dir() -> str:
    return "examples/sparse_2of4_quantization_fp8"


@requires_gpu_count(1)
class TestSparse2of4QuantizationFP8:
    """
    Tests for examples in the "sparse_2of4_quantization_fp8" example folder.
    """

    @pytest.mark.parametrize(("flags"), [[], ["--fp8"]])
    def test_2of4_example_script(
        self, example_dir: str, tmp_path: Path, flags: list[str]
    ):
        """
        Tests for the "llama3_8b_2of4.py" example script.
        """
        script_filename = "llama3_8b_2of4.py"
        command, result = copy_and_run_script(
            tmp_path, example_dir, script_filename, flags=flags
        )

        assert result.returncode == 0, gen_cmd_fail_message(command, result)
