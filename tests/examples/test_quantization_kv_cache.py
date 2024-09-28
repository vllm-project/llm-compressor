import shlex
from pathlib import Path

import pytest

from tests.examples.utils import (
    ReadMe,
    copy_and_run_command,
    gen_cmd_fail_message,
    requires_gpu_count,
)


@pytest.fixture
def example_dir() -> str:
    return "examples/quantization_kv_cache"


@pytest.mark.example
@requires_gpu_count(1)
class TestQuantizationKVCache:
    """
    Tests for examples in the "quantization_kv_cache" example folder.
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
