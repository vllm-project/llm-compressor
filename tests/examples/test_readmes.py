from tests.examples.utils import ReadMe

from pathlib import Path

import pytest


@pytest.mark.example
@pytest.mark.parametrize(
    "subdir",
    [
        "quantization_2of4_sparse_w4a16",
        "quantization_kv_cache",
        "quantization_w4a16",
        "quantization_w8a8_fp8",
        "quantization_w8a8_int8",
        "quantizing_moe",
    ],
)
def test_readmes(subdir):
    path = Path("examples") / subdir / "README.md"

    readme = ReadMe(path)

    cmd = readme.get_code_block_content(position=1, lang="bash").split()

    assert cmd[0] in ["python", "python3"]

    script_path = Path("examples") / subdir / cmd[1]

    assert script_path.is_file(), f"Could not find script at {script_path}"
