from pathlib import Path

import pytest

from tests.examples.utils import (
    ReadMe,
    copy_and_run_script,
    gen_cmd_fail_message,
    requires_gpu_count,
    requires_gpu_mem,
)


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
        ("script_filename", "visible_gpus"),
        [
            pytest.param("cpu_offloading_fp8.py", "0", id="cpu_offloading"),
            pytest.param(
                "multi_gpu_int8.py",
                "",
                id="multi_gpu_int8",
                marks=[
                    requires_gpu_count(2),
                    pytest.mark.multi_gpu,
                ],
            ),
            pytest.param(
                "mult_gpus_int8_device_map.py",
                "0",
                id="mult_gpus_int8_device_map",
            ),
        ],
    )
    @requires_gpu_count(1)
    def test_example_scripts(
        self,
        example_dir: str,
        visible_gpus: str,
        script_filename: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """
        Test for the example scripts in the folder.
        """

        if visible_gpus:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_gpus)

        command, result = copy_and_run_script(tmp_path, example_dir, script_filename)

        assert result.returncode == 0, gen_cmd_fail_message(command, result)
