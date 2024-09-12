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

# flash_attn module is required. It cannot safely be specified as a dependency because
# it rqeuires a number of non-standard packages to be installed in order to be built
# such as pytorch, and thus cannot be installed in a clean environment (those
# dependencies must be installed prior to attempting to install flash_attn)
pytest.importorskip("flash_attn", reason="flash_attn is required")


@pytest.fixture
def example_dir() -> str:
    return "examples/quantizing_moe"


@pytest.mark.example
@requires_gpu
@requires_torch
class TestQuantizingMOE:
    """
    Tests for examples in the "quantizing_moe" example folder.
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

    def test_deepseek_example_script(self, example_dir: str, tmp_path: Path):
        """
        Test for the "deepseek_moe_w8a8.py" script in the folder.
        """
        shutil.copytree(Path.cwd() / example_dir, tmp_path / example_dir)

        command = [sys.executable, "deepseek_moe_w8a8.py"]
        result = run_cli_command(command, cwd=tmp_path / example_dir)

        assert result.returncode == 0, gen_cmd_fail_message(command, result)
