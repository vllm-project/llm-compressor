import shlex
import shutil
import sys
from pathlib import Path

import pytest

from tests.examples.utils import (
    ReadMe,
    gen_cmd_fail_message,
    requires_gpu,
    requires_torch,
)
from tests.testing_utils import run_cli_command


@pytest.fixture
def example_dir() -> str:
    return "examples/quantization_w8a8_int8"


@pytest.mark.example
@requires_gpu
@requires_torch
class TestQuantizationW8A8_Int8:
    """
    Tests for examples in the "quantization_w8a8_int8" example folder.
    """

    def test_doc_example_command(self, example_dir: str, tmp_path: Path):
        """
        Test for the example command in the README.
        """
        readme_path = Path.cwd() / example_dir / "README.md"
        readme = ReadMe(readme_path)

        command = readme.get_code_block_content(position=2, lang="shell")
        assert command.startswith("python")

        shutil.copytree(Path.cwd() / example_dir, tmp_path / example_dir)

        command = shlex.split(command)
        result = run_cli_command(command, cwd=tmp_path / example_dir)

        assert result.returncode == 0, gen_cmd_fail_message(command, result)

    def test_gemma2_example_script(self, example_dir: str, tmp_path: Path):
        """
        Test for the "gemma2_example.py" script in the folder.
        """
        shutil.copytree(Path.cwd() / example_dir, tmp_path / example_dir)

        command = [sys.executable, "gemma2_example.py"]
        result = run_cli_command(command, cwd=tmp_path / example_dir)

        assert result.returncode == 0, gen_cmd_fail_message(command, result)
