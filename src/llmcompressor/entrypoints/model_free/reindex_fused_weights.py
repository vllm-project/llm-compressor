import argparse

from compressed_tensors.entrypoints.convert import (
    reindex_checkpoint,
)
from .microscale import get_unmatched_microscale_names


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

    reindex_checkpoint(
        model_stub,
        save_directory,
        num_workers,
        get_unmatched_names=get_unmatched_microscale_names,
    )


def main():
    args = parse_args()
    reindex_fused_weights(args.model_stub, args.save_directory, args.num_workers)


if __name__ == "__main__":
    main()
