import os

from huggingface_hub import list_repo_files
from transformers.utils.hub import cached_file

__all__ = ["get_checkpoint_files", "is_weights_file"]

weights_files = [
    ".bin",
    ".safetensors",
    ".pth",
    ".msgpack",
    ".pt",
]


def is_weights_file(file_name: str) -> bool:
    return any(file_name.endswith(suffix) for suffix in weights_files)


def get_checkpoint_files(model_stub: str | os.PathLike) -> list[str]:
    # In the future, this function can accept and pass download kwargs to cached_file

    if os.path.exists(model_stub):
        file_paths = walk_file_paths(model_stub)
    else:
        file_paths = list_repo_files(model_stub)

    return [(file_path, cached_file(model_stub, file_path)) for file_path in file_paths]


def walk_file_paths(root_dir: str) -> list[str]:
    all_files = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            all_files.append(full_path)
    return all_files


# distinguish relative file paths from absolute/resolved file paths
# relative file paths are used to find the save path
# resolved file paths are what are used to load data
