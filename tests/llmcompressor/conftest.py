import os
import shutil
import tempfile
from typing import List

import pytest

try:
    import wandb
except Exception:
    wandb = None


os.environ["NM_TEST_MODE"] = "True"
os.environ["NM_TEST_LOG_DIR"] = "nm_temp_test_logs"


def _get_files(directory: str) -> List[str]:
    list_filepaths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            list_filepaths.append(os.path.join(os.path.abspath(root), file))
    return list_filepaths


def _files_size_mb(path_list: List[str]) -> int:
    files_size = 0
    for file_path in path_list:
        try:
            files_size += os.path.getsize(file_path)
        # if file is deleted between the time we get the list of files
        # and the time we get the size of the file, ignore it
        except FileNotFoundError:
            pass

    return files_size / 1024 / 1024


@pytest.fixture(scope="session", autouse=True)
def check_for_created_files():
    start_files_root = _get_files(directory=r".")
    start_files_temp = _get_files(directory=tempfile.gettempdir())
    yield
    if wandb:
        wandb.finish()
    log_dir = os.environ.get("NM_TEST_LOG_DIR")
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)

    # allow creation of __pycache__ directories
    end_files_root = [
        f_path for f_path in _get_files(directory=r".") if "__pycache__" not in f_path
    ]
    end_files_temp = _get_files(directory=tempfile.gettempdir())
    # assert no files created in root directory while running
    # the pytest suite
    assert len(start_files_root) >= len(end_files_root), (
        f"{len(end_files_root) - len(start_files_root)} "
        f"files created in current working "
        f"directory during pytest run. "
        f"Created files: {set(end_files_root) - set(start_files_root)}"
    )

    max_allowed_sized_temp_files_megabytes = 1
    end_files_temp = _get_files(directory=tempfile.gettempdir())
    created_temp_files = set(end_files_temp) - set(start_files_temp)
    # pytest temp files are automatically deleted, exclude from size calculation
    created_temp_files = [
        f_path for f_path in created_temp_files if "pytest-of" not in f_path
    ]

    # assert no more than 1 megabyte of temp files created in temp directory
    # while running the pytest suite (excluding files created by pytest)
    size_of_temp_files_megabytes = _files_size_mb(created_temp_files)
    assert max_allowed_sized_temp_files_megabytes >= size_of_temp_files_megabytes, (
        f"{size_of_temp_files_megabytes} "
        f"megabytes of temp files created in temp directory during pytest run. "
        f"Created files: {set(end_files_temp) - set(start_files_temp)}"
    )


@pytest.fixture(autouse=True, scope="function")
def setup_fresh_session():
    """
    setup any state tied to the execution of the given method in a
    class.  setup_method is invoked for every test method of a class.
    """
    from llmcompressor.core import reset_session

    # start with a clean session for each test
    reset_session()
    yield
    # reset the session after each test
    reset_session()
