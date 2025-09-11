import re
from pathlib import Path
from typing import Optional, Union

import pytest
from bs4 import BeautifulSoup, ResultSet, Tag
from cmarkgfm import github_flavored_markdown_to_html as gfm_to_html


def requires_gpu_count(num_required_gpus: int) -> pytest.MarkDecorator:
    """
    Pytest decorator to skip based on number of available GPUs. This plays nicely with
    the CUDA_VISIBLE_DEVICES environment variable.
    """
    import torch

    num_gpus = torch.cuda.device_count()
    reason = f"{num_required_gpus} GPUs required, {num_gpus} GPUs detected"
    return pytest.mark.skipif(num_required_gpus > num_gpus, reason=reason)


def requires_gpu_mem(required_amount: Union[int, float]) -> pytest.MarkDecorator:
    """
    Pytest decorator to skip based on total available GPU memory (across all GPUs). This
    plays nicely with the CUDA_VISIBLE_DEVICES environment variable.

    Note: make sure to account for measured memory vs. simple specs. For example, H100
    has '80 GiB' VRAM, however, the actual number, at least per PyTorch, is ~79.2 GiB.

    :param amount: amount of required GPU memory in GiB
    """
    import torch

    vram_bytes = sum(
        torch.cuda.mem_get_info(device_id)[1]
        for device_id in range(torch.cuda.device_count())
    )
    actual_vram = vram_bytes / 1024**3
    reason = (
        f"{required_amount} GiB GPU memory required, "
        f"{actual_vram:.1f} GiB GPU memory found"
    )
    return pytest.mark.skipif(required_amount > actual_vram, reason=reason)


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
