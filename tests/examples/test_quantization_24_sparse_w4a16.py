import shlex
import shutil
import sys
from pathlib import Path

import pytest

from tests.examples.utils import (
    ReadMe,
    copy_and_run_command,
    gen_cmd_fail_message,
    requires_gpu_count,
)
from tests.testing_utils import run_cli_command


@pytest.fixture
def example_dir() -> str:
    return "examples/quantization_24_sparse_w4a16"


@pytest.mark.example
@requires_gpu_count(1)
class TestQuantization24SparseW4A16:
    """
    Tests for examples in the "quantization_24_sparse_w4a16" example folder.
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

    def test_alternative_recipe(self, example_dir: str, tmp_path: Path):
        """
        Test for the example command in the README with the alternative recipe file.
        """
        shutil.copytree(str(Path.cwd() / example_dir), tmp_path / example_dir)

        # replace recipe file with alternative
        script_filename = "llama7b_sparse_w4a16.py"
        script_path = tmp_path / example_dir / script_filename
        content = script_path.read_text(encoding="utf-8")
        content = content.replace(
            "2:4_w4a16_recipe.yaml", "2:4_w4a16_group-128_recipe.yaml"
        )
        script_path.write_text(content, encoding="utf-8")

        command = [sys.executable, script_filename]
        result = run_cli_command(command, cwd=tmp_path / example_dir)

        assert result.returncode == 0, gen_cmd_fail_message(command, result)
