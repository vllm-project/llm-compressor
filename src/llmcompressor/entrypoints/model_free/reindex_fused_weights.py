import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import tqdm
from loguru import logger
from safetensors.torch import load_file, save_file

from llmcompressor.entrypoints.model_free.helpers import (
    find_safetensors_index_file,
    invert_mapping,
)
from llmcompressor.entrypoints.model_free.microscale import get_fused_names
from llmcompressor.entrypoints.model_free.model_utils import (
    get_checkpoint_files,
    is_weights_file,
)
from llmcompressor.entrypoints.model_free.save_utils import update_safetensors_index


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("model_stub", type=str, help="huggingface model hub or path to local weights files")  # noqa: E501
    parser.add_argument("save_directory", type=str, help="output directory for reindexed weights files")  # noqa: E501
    parser.add_argument("--num_workers", type=int, default=5, help="number of worker threads to save files with")  # noqa: E501
    # fmt: on
    return parser.parse_args()


def reindex_fused_weights(
    model_stub: str,
    save_directory: str,
    num_workers: int = 5,
):
    """
    Script used to reindex the safetensors files of a model such that all fused modules
    (gate_up, qkv) are in the same safetensors file. This is required by model_free_ptq
    for microscale schemes (NVFP4A16, MXFP4A16)

    This script assumes weight locality; if a set of fused weights are not in a file,
    1. the incomplete set is the last set of weights (sorted alphabetically)
    2. the remainder of the incomplete set is the next file (sorted alphabetically)

    This assumption holds true for most model checkpoints, even in the common case where
    weights are sorted alphabetically and not numerically.

    :param model_stub: huggingface model hub or path to local weights files
    :param save_directory: output directory for reindexed weights files
    :param num_workers: number of worker threads to save files with
    """

    # read files
    model_files = get_checkpoint_files(model_stub)
    index_file = find_safetensors_index_file(model_files)
    if index_file is None:
        raise ValueError(
            "This script is used to modify safetensor file shards, but was "
            "unable to find safetenors index file. No reindexing is required."
        )

    # copy non-weight files
    for file_path, resolved_path in model_files.items():
        save_path = Path(save_directory) / file_path

        if file_path.endswith("safetensors"):
            continue
        else:
            if is_weights_file(file_path):
                logger.warning(f"Skip processing for weights file {file_path}")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Copying {file_path} {save_path}")
            shutil.copyfile(resolved_path, save_path)

    # read index file
    with open(index_file, "r") as file:
        index_file_data = json.load(file)

    weight_map: dict[str, str] = index_file_data["weight_map"]
    final_weight_map: dict[str, str] = {}

    # set up copy executor and carry over
    writers = ThreadPoolExecutor(max_workers=num_workers)
    carry_over_tensors: dict[str, torch.Tensor] = {}

    # iterate in alphabetical order on assumption of weight-file locality
    file_map = invert_mapping(weight_map)
    file_map = sorted(file_map)
    progress = tqdm.tqdm(total=len(file_map))
    for file_name in file_map:
        file_path = model_files[file_name]
        save_path = os.path.join(save_directory, file_name)
        tensors = load_file(file_path)

        if len(carry_over_tensors) > 0:
            # add carryover
            tensors.update(carry_over_tensors)
            logger.info(f"Moved {list(carry_over_tensors.keys())} into {file_name}")
            carry_over_tensors = {}

        tensor_names = sorted(list(tensors.keys()))
        _matches, unmatched_sets = get_fused_names(tensor_names)
        for unmatched in unmatched_sets:
            # move to carry over
            unmatched_tensors = {
                key: tensors[key] for key in unmatched.values() if key is not None
            }
            carry_over_tensors.update(unmatched_tensors)

            # delete from current file
            for key in unmatched_tensors:
                tensor_names.remove(key)
                del tensors[key]

        # save tensors after modification
        writers.submit(_with_progress, save_file, tensors, save_path, progress=progress)
        final_weight_map.update({name: file_name for name in tensor_names})

    total_size = index_file_data["metadata"]["total_size"]
    update_safetensors_index(save_directory, total_size, final_weight_map)
    writers.shutdown(wait=True)


def _with_progress(fn: callable, *args, progress: tqdm.tqdm):
    ret = fn(*args)
    progress.update(1)
    return ret


def main():
    args = parse_args()
    reindex_fused_weights(args.model_stub, args.save_directory, args.num_workers)


if __name__ == "__main__":
    main()
