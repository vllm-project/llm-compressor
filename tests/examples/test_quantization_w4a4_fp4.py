import json
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
            "llama3_example.py",
            "llama4_example.py",
            "qwen_30b_a3b.py",
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

        # verify the expected directory was generated
        nvfp4_dirs: List[Path] = [p for p in tmp_path.rglob("*-NVFP4") if p.is_dir()]
        assert (
            len(nvfp4_dirs)
        ) == 1, f"did not find exactly one generated folder: {nvfp4_dirs}"

        # verify the format in the generated config
        config_json = json.loads((nvfp4_dirs[0] / "config.json").read_text())
        config_format = config_json["quantization_config"]["format"]
        assert config_format == "nvfp4-pack-quantized"
