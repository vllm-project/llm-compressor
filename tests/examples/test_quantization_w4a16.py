import shlex
import shutil
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
    return "examples/quantization_w4a16"


@pytest.mark.example
@requires_gpu
@requires_torch
class TestQuantizationW4A16:
    """
    Tests for examples in the "quantization_w4a16" example folder.
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
