from typing import Union

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def get_raw_dataset(
    path: str,
    **kwargs,
) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    """
    Load HF alias or a dataset in local path.

    :param path: Path or name of the dataset. Accepts HF dataset stub or
        local file directory in csv, json, parquet, etc.
        If local path is provided, it must be
            1. Download path where HF dataset was downloaded to
            2. Path containing (train, test, validation) with the same extention.
            Supported extentions are json, jsonl, csv, arrow, parquet, text,
            and xlsx,

    :return: the requested dataset

    """
    return load_dataset(
        path,
        **kwargs,
    )
