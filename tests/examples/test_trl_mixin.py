from pathlib import Path

import pytest

from tests.examples.utils import (
    copy_and_run_script,
    gen_cmd_fail_message,
    requires_gpu_count,
)


@pytest.fixture
def example_dir() -> str:
    return "examples/trl_mixin"


@pytest.mark.example
@requires_gpu_count(1)
class TestTRLMixin:
    """
    Tests for examples in the "trl_mixin" example folder.
    """

    @pytest.mark.parametrize(
        "script_filename",
        [
            "ex_trl_constant.py",
            # ex_trl_distillation.py hits CUDA OOM on 1x H100 (80 GiB VRAM)
            pytest.param("ex_trl_distillation.py", marks=pytest.mark.multi_gpu),
        ],
    )
    def test_example_scripts(
        self, example_dir: str, script_filename: str, tmp_path: Path
    ):
        """
        Test for the example scripts in the folder.
        """
        command, result = copy_and_run_script(tmp_path, example_dir, script_filename)

        assert result.returncode == 0, gen_cmd_fail_message(command, result)
