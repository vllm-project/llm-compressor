import shlex
from pathlib import Path

import pytest

from tests.examples.utils import (
    ReadMe,
    copy_and_run_command,
    copy_and_run_script,
    gen_cmd_fail_message,
    requires_gpu,
    requires_torch,
)


@pytest.fixture
def example_dir() -> str:
    return "examples/quantization_w8a8_fp8"


@pytest.mark.example
@requires_gpu
@requires_torch
class TestQuantizationW8A8_FP8:
    """
    Tests for examples in the "quantization_w8a8_fp8" example folder.
    """

    def test_doc_example_command(self, example_dir: str, tmp_path: Path):
        """
        Test for the example command in the README.
        """
        readme_path = Path.cwd() / example_dir / "README.md"
        readme = ReadMe(readme_path)

        command = readme.get_code_block_content(position=2, lang="shell")
        assert command.startswith("python")

        command = shlex.split(command)
        result = copy_and_run_command(tmp_path, example_dir, command)

        assert result.returncode == 0, gen_cmd_fail_message(command, result)

    def test_gemma2_example_script(self, example_dir: str, tmp_path: Path):
        """
        Test for the "gemma2_example.py" script in the folder.
        """
        script_filename = "gemma2_example.py"
        command, result = copy_and_run_script(tmp_path, example_dir, script_filename)

        assert result.returncode == 0, gen_cmd_fail_message(command, result)
