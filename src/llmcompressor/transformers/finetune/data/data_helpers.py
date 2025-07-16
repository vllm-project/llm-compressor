import logging
import os
from typing import Any, Dict, Optional

from datasets import Dataset, load_dataset

LOGGER = logging.getLogger(__name__)
LABELS_MASK_VALUE = -100

__all__ = [
    "get_raw_dataset",
    "get_custom_datasets_from_path",
]


def get_raw_dataset(
    dataset_args,
    cache_dir: Optional[str] = None,
    streaming: Optional[bool] = False,
    **kwargs,
) -> Dataset:
    """
    Load the raw dataset from Hugging Face, using cached copy if available

    :param cache_dir: disk location to search for cached dataset
    :param streaming: True to stream data from Hugging Face, otherwise download
    :return: the requested dataset

    """
    raw_datasets = load_dataset(
        dataset_args.dataset,
        dataset_args.dataset_config_name,
        cache_dir=cache_dir,
        streaming=streaming,
        **kwargs,
    )
    return raw_datasets


def get_custom_datasets_from_path(path: str, ext: str = "json") -> Dict[str, str]:
    """
    Get a dictionary of custom datasets from a directory path. Support HF's load_dataset
     for local folder datasets https://huggingface.co/docs/datasets/loading

    This function scans the specified directory path for files with a
     specific extension (default is '.json').
    It constructs a dictionary where the keys are either subdirectory names or
     direct dataset names (depending on the directory structure)
    and the values are either file paths (if only one file exists with that name) or
     lists of file paths (if multiple files exist).

    :param path: The path to the directory containing the dataset files.
    :param ext: The file extension to filter files by. Default is 'json'.

    :return: A dictionary mapping dataset names to their file paths or lists of
     file paths.

    Example:
        dataset = get_custom_datasets_from_path("/path/to/dataset/directory", "json")

    Note:
        If datasets are organized in subdirectories, the function constructs the
         dictionary with lists of file paths.
        If datasets are found directly in the main directory, they are included with
         their respective names.

    Accepts:
        - path\
            train.json
            test.json
            val.json

        - path\
            train\
                data1.json
                data2.json
                ...
            test\
                ...
            val\
                ...

    """
    data_files = {}

    if any(filename.endswith(ext) for filename in os.listdir(path)):
        # If there are files with the given extension in the path
        for filename in os.listdir(path):
            if filename.endswith(ext):
                name, _ = os.path.splitext(filename)
                data_files[name] = os.path.join(path, filename)
    else:
        # If datasets are organized in subdirectories
        for root, dirs, files in os.walk(path):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                dir_dataset = []
                for filename in os.listdir(dir_path):
                    if filename.endswith(ext):
                        file_path = os.path.join(dir_path, filename)
                        dir_dataset.append(file_path)
                if dir_dataset:
                    data_files[dir_name] = dir_dataset

    return transform_dataset_keys(data_files)


def transform_dataset_keys(data_files: Dict[str, Any]):
    """
    Transform dict keys to `train`, `val` or `test` for the given input dict
    if matches exist with the existing keys. Note that there can only be one
    matching file name.
    Ex. Folder(train_foo.json)           -> Folder(train.json)
        Folder(train1.json, train2.json) -> Same

    :param data_files: The dict where keys will be transformed
    """
    keys = set(data_files.keys())

    def transform_dataset_key(candidate: str) -> None:
        for key in keys:
            if candidate in key:
                if key == candidate:
                    return
                val = data_files.pop(key)
                data_files[candidate] = val

    def do_transform(candidate: str) -> bool:
        return sum(candidate in key for key in keys) == 1

    dataset_keys = ("train", "val", "test")
    for dataset_key in dataset_keys:
        if do_transform(dataset_key):
            transform_dataset_key(dataset_key)

    return data_files
