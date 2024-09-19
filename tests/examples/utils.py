import itertools
import re
import shlex
from pathlib import Path
from subprocess import CompletedProcess
from typing import Generator, Iterable, List, Optional, TypeVar, Union

import pytest
from bs4 import BeautifulSoup, ResultSet, Tag
from cmarkgfm import github_flavored_markdown_to_html as gfm_to_html

from tests.testing_utils import is_gpu_available, is_torch_available

_T = TypeVar("_T")

requires_gpu = pytest.mark.skipif(not is_gpu_available(), reason="GPU is required")
requires_torch = pytest.mark.skipif(
    not is_torch_available(), reason="torch is required"
)


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


def batched(iterable: Iterable[_T], n: int) -> Generator[tuple[_T, ...], None, None]:
    # implementation from Python docs as this function is added to itertools in 3.12
    # https://docs.python.org/3/library/itertools.html#itertools.batched
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch


def get_gpu_batches(gpu_count: int, worker_count: int) -> List[str]:
    if gpu_count % worker_count != 0:
        raise ValueError("GPU count must be evenly divisible by worker count")

    group_size = gpu_count / worker_count
    groups = batched(range(gpu_count), int(group_size))
    return [",".join(map(str, group)) for group in groups]


def gen_cmd_fail_message(command: List[str], result: CompletedProcess[str]) -> str:
    """
    Generate an failure message including the command and its output.

    :param result: a `CompletedProcess` object
    :return: a formatted failure message
    """
    return (
        f"command failed with exit code {result.returncode}:\n"
        f"Command:\n{shlex.join(command)}\nOutput:\n{result.stdout}"
    )


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
        Get contents of code block at specified position (starting with 1). Optionally
        pass a language specifier, `lang`, to only look at code blocks highlighted for
        that language (happens prior to indexing).

        :param position: position of code block to get (starting at 1)
        :param lang: language of code block to filter by
        :return: content of the code block
        """
        code_blocks = self.get_code_blocks(lang=lang)
        code = code_blocks[position - 1].text.strip()
        return code
