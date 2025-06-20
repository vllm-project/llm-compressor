import shlex
from pathlib import Path

import pytest

from tests.examples.utils import (
    ReadMe,
    copy_and_run_command,
    copy_and_run_script,
    gen_cmd_fail_message,
    requires_gpu_count,
)


@pytest.fixture
def example_dir() -> str:
    return "examples/quantizing_moe"


@pytest.mark.example
class TestQuantizingMOE:
    """
    Tests for examples in the "quantizing_moe" example folder.
    """

    @pytest.mark.multi_gpu
    @requires_gpu_count(2)
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

    @pytest.mark.parametrize(
        "script_filename",
        [
            pytest.param(
                "deepseek_moe_w4a16.py",
                marks=[
                    pytest.mark.multi_gpu,
                    pytest.mark.skip(reason="exceptionally long run time"),
                ],
            ),
            pytest.param("deepseek_moe_w8a8_fp8.py"),
            pytest.param("deepseek_moe_w8a8_int8.py", marks=pytest.mark.multi_gpu),
        ],
    )
    def test_deepseek_example_script(
        self, script_filename: str, example_dir: str, tmp_path: Path
    ):
        """
        Test for the other example scripts in the folder.
        """
        command, result = copy_and_run_script(tmp_path, example_dir, script_filename)

        assert result.returncode == 0, gen_cmd_fail_message(command, result)
