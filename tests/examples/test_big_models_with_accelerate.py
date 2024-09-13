import shutil
import sys
from pathlib import Path

import pytest

from tests.examples.utils import (
    ReadMe,
    gen_cmd_fail_message,
    requires_gpu,
    requires_gpu_mem,
    requires_torch,
)
from tests.testing_utils import run_cli_command


@pytest.fixture
def example_dir() -> str:
    return "examples/big_models_with_accelerate"


@pytest.mark.example
class TestBigModelsWithAccelerate:
    """
    Tests for examples in the "big_models_with_accelerate" example folder.
    """

    def test_readme_has_install_command(self, example_dir: str):
        """
        Test that the README has a valid install command.
        """
        readme_path = Path.cwd() / example_dir / "README.md"
        readme = ReadMe(readme_path)

        code = readme.get_code_block_content(position=1, lang="shell")
        assert "pip install" in code

        assert code.startswith("pip install llmcompressor")

    @pytest.mark.parametrize(
        "script_filename",
        [
            "cpu_offloading_fp8.py",
            pytest.param("multi_gpu_int8.py", marks=requires_gpu_mem(630)),
            "multi_gpu_int8_sequential_update.py",
        ],
    )
    @requires_gpu
    @requires_torch
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
