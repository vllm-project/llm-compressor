import argparse

from compressed_tensors.utils.helpers import deprecated
from loguru import logger


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("model_stub", type=str, help="huggingface model hub or path to local weights files")  # noqa: E501
    parser.add_argument("save_directory", type=str, help="output directory for reindexed weights files")  # noqa: E501
    parser.add_argument("--num_workers", type=int, default=5, help="number of worker threads to save files with")  # noqa: E501
    # fmt: on
    return parser.parse_args()


@deprecated(
    "The reindex_fused_weights preprocessing step is no longer necessary. "
    "model_free_ptq now handles fused weights directly without requiring reindexing."
)
def reindex_fused_weights(
    model_stub: str,
    save_directory: str,
    num_workers: int = 5,
):
    """
    This function is deprecated. The reindexing step is no longer necessary for
    model_free_ptq with microscale schemes (NVFP4A16, MXFP4A16).

    Previously, this script was used to reindex safetensors files so that all fused
    modules (gate_up, qkv) were in the same safetensors file. This is now handled
    automatically by model_free_ptq.

    :param model_stub: huggingface model hub or path to local weights files
    :param save_directory: output directory for reindexed weights files
    :param num_workers: number of worker threads to save files with
    """
    logger.warning(
        "reindex_fused_weights is deprecated and does nothing. "
        "The reindexing step is no longer necessary - model_free_ptq now handles "
        "fused weights directly."
    )


def main():
    args = parse_args()
    reindex_fused_weights(args.model_stub, args.save_directory, args.num_workers)


if __name__ == "__main__":
    main()
