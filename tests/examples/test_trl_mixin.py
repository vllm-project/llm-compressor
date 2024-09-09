import shutil
import sys
from pathlib import Path

import pytest

from tests.examples.utils import gen_cmd_fail_message, requires_gpu, requires_torch
from tests.testing_utils import run_cli_command


@pytest.fixture
def example_dir() -> str:
    return "examples/trl_mixin"


@pytest.mark.example
@requires_gpu
@requires_torch
class TestTRLMixin:
    """
    Tests for examples in the "trl_mixin" example folder.
    """

    @pytest.mark.parametrize(
        "script_filename",
        ["ex_trl_constant.py", "ex_trl_distillation.py"],
    )
    def test_example_scripts(
        self, example_dir: str, script_filename: str, tmp_path: Path
    ):
        """
        Test for the example scripts in the folder.
        """
        shutil.copytree(Path.cwd() / example_dir, tmp_path / example_dir)

        script_path = tmp_path / example_dir / script_filename
        command = [sys.executable, str(script_path)]
        result = run_cli_command(command, cwd=tmp_path / example_dir)

        assert result.returncode == 0, gen_cmd_fail_message(command, result)
