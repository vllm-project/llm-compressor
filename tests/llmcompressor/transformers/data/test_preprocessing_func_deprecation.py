import warnings

import pytest
from transformers import HfArgumentParser

from llmcompressor.args.dataset_arguments import (
    CustomDatasetArguments,
    DatasetArguments,
)


@pytest.mark.unit
@pytest.mark.parametrize("arguments", [CustomDatasetArguments, DatasetArguments])
@pytest.mark.parametrize("preprocessing_func", [lambda row: row, "prep.py:process"])
def test_preprocessing_func_warns_without_changing_value(arguments, preprocessing_func):
    with pytest.warns(FutureWarning, match="`preprocessing_func` is deprecated"):
        dataset_args = arguments(preprocessing_func=preprocessing_func)

    assert dataset_args.preprocessing_func is preprocessing_func


@pytest.mark.unit
def test_default_dataset_arguments_do_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        DatasetArguments()

    assert not caught


@pytest.mark.unit
def test_preprocessing_func_warns_when_parsed_as_oneshot_argument():
    parser = HfArgumentParser(DatasetArguments)

    with pytest.warns(FutureWarning, match="`preprocessing_func` is deprecated"):
        dataset_args = parser.parse_dict({"preprocessing_func": "prep.py:process"})[0]

    assert dataset_args.preprocessing_func == "prep.py:process"
