import re
from pathlib import Path

import pytest
from bs4 import BeautifulSoup
from cmarkgfm import github_flavored_markdown_to_html as gfm_to_html


class ReadMe:
    """Class for reading and parsing a README file for code blocks."""

    def __init__(self, path: Path) -> None:
        self.path = path
        content = self.path.expanduser().read_text(encoding="utf-8")

        # Normalize code fence language
        content = re.sub(r"```(shell|bash|sh)\b", "```shell", content)
        html = gfm_to_html(content)
        self.soup = BeautifulSoup(html, "html.parser")

    def get_code_block_content(self, *, position: int, lang: str) -> str:
        """
        Get contents of code block of specified language and position (starting with 0).
        :param position: position of code block to get (starting at 0)
        :param lang: language of code block to get
        :return: content of the code block
        """
        selector = f'pre[lang="{lang}"] > code'
        code_blocks = self.soup.select(selector)
        code = code_blocks[position].text.strip()
        return code


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
    cmd = readme.get_code_block_content(position=1, lang="shell").split()

    assert cmd[0] in ["python", "python3"]
    script_path = Path("examples") / subdir / cmd[1]
    assert script_path.is_file(), f"Could not find script at {script_path}"
