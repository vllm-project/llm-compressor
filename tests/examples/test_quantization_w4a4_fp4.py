
import shlex
from pathlib import Path
from typing import List

import pytest

from tests.examples.utils import (
    copy_and_run_script,
    gen_cmd_fail_message,
    requires_gpu_count,
)


@pytest.fixture
def example_dir() -> str:
    return "examples/quantization_w4a4_fp4"


@requires_gpu_count(1)
@pytest.mark.example
class TestQuantizationW4A4_FP4:
    """
    Tests for examples in the "quantization_w4a4_fp4" example folder.
    """

    @pytest.mark.parametrize(
        "script_filename",
        [
            pytest.param("llama3_example.py"),
            pytest.param("llama4_example.py"),
            pytest.param("qwen_30b_a3b.py"),
        ],
    )
    def test_quantization_w4a4_fp4_example_script(
        self, script_filename: str, example_dir: str, tmp_path: Path
    ):
        """
        Tests the example scripts in the folder.
        """
        command, result = copy_and_run_script(tmp_path, example_dir, script_filename)

        assert result.returncode == 0, gen_cmd_fail_message(command, result)

        # verify that the expected directory got generated
        nvfp4_dirs: List[Path] = list(Path.cwd().glob("*-NVFP4"))
        assert(len(nvfp4_dirs)) == 1, (
            f"unexpectedly found more than one generated folder: {nvfp4_dirs}"
        )
        print("generated model directory:/n")
        print(list(nvfp4_dirs[0].iterdir()))

        # is the generated content in the expected format?
        config_json = Path(nvfp4_dirs[0] / "config.json").read_text()
        config_format = config_json["quantization_config"]["format"]
        assert(config_format == "nvfp4-pack-quantized")
