import shlex
import shutil
import sys
from pathlib import Path

import pytest

from tests.examples.utils import (
    ReadMe,
    gen_cmd_fail_message,
    requires_gpu_count,
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
@pytest.mark.multi_gpu
@requires_gpu_count(2)
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

    @pytest.mark.parametrize(
        "script_filename",
        [
            pytest.param(
                "deepseek_moe_w4a16.py",
                pytest.mark.skip(reason="exceptionally long run time"),
            ),
            pytest.param("deepseek_moe_w8a8.py"),
        ],
    )
    def test_deepseek_example_script(
        self, script_filename: str, example_dir: str, tmp_path: Path
    ):
        """
        Test for the other example scripts in the folder.
        """
        shutil.copytree(Path.cwd() / example_dir, tmp_path / example_dir)

        command = [sys.executable, script_filename]
        result = run_cli_command(command, cwd=tmp_path / example_dir)

        assert result.returncode == 0, gen_cmd_fail_message(command, result)
