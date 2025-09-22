import re
from pathlib import Path
from typing import Optional

import pytest
from bs4 import BeautifulSoup, ResultSet, Tag
from cmarkgfm import github_flavored_markdown_to_html as gfm_to_html


class ReadMe:
    """
    Class representing a README (Markdown) file with methods to expedite common usage.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.content = self.path.expanduser().read_text(encoding="utf-8")
        self.__normalize_code_fence_lang()
        self.html = gfm_to_html(self.content)
        self.soup = BeautifulSoup(self.html, "html.parser")

    def __normalize_code_fence_lang(self):
        """
        Perform limited normalization on the code language of code blocks to maintain
        consistency and simplicity with locating them.
        """
        self.content = re.sub(r"```(shell|bash|sh)\b", "```shell", self.content)

    def get_code_blocks(self, *, lang: Optional[str] = None) -> ResultSet[Tag]:
        """
        Get all code blocks with language `lang`, or all code blocks if `lang` is None
        (default).
        :param lang: language of code block to filter by
        :return: code block `Tag`s found in README
        """
        lang = "shell" if lang == "bash" else lang
        selector = f'pre[lang="{lang}"] > code' if lang else "pre > code"
        tags = self.soup.select(selector)
        return tags

    def get_code_block_content(
        self, *, position: int, lang: Optional[str] = None
    ) -> str:
        """
        Get contents of code block at specified position (starting with 0). Optionally
        pass a language specifier, `lang`, to only look at code blocks highlighted for
        that language (happens prior to indexing).
        :param position: position of code block to get (starting at 0)
        :param lang: language of code block to filter by
        :return: content of the code block
        """
        code_blocks = self.get_code_blocks(lang=lang)
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

    cmd = readme.get_code_block_content(position=1, lang="bash").split()

    assert cmd[0] in ["python", "python3"]

    script_path = Path("examples") / subdir / cmd[1]

    assert script_path.is_file(), f"Could not find script at {script_path}"
