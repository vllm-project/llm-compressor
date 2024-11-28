# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from copy import deepcopy
from typing import Dict, List, Union

from datasets.dataset_dict import Dataset, DatasetDict

from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.transformers.utils.preprocessing_functions import (
    PreprocessingFunctionRegistry,
)
from llmcompressor.utils import import_from_path


@TextGenerationDataset.register(name="custom", alias=["json", "csv"])
class CustomDataset(TextGenerationDataset):
    """
    Child text generation class for custom local dataset supporting load
    for csv and json

    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
        Can also be set to None to load all the splits
    :param tokenizer: tokenizer to use on dataset

    """

    def __init__(self, data_args, split, tokenizer):
        data_args = deepcopy(data_args)
        super().__init__(
            text_column=data_args.text_column,
            data_args=data_args,
            split=split,
            tokenizer=tokenizer,
        )

    def get_raw_dataset(self, *_ignore, **__ignore) -> Union[DatasetDict, Dataset]:
        """Get the raw dataset and apply preprocessing func if provided"""

        # load dataset
        dataset = (
            self.data_args.dataset
            if isinstance(self.data_args.dataset, (DatasetDict, Dataset))
            else super().get_raw_dataset()  # load dataset from file or HF Hub
        )

        # preprocess dataset
        dataset = self._preprocess_dataset(dataset)
        dataset = self._remove_columns_from_dataset(dataset)

        return dataset

    def _preprocess_dataset(
        self, dataset: Union[DatasetDict, Dataset]
    ) -> Union[DatasetDict, Dataset]:
        preprocessing_func = self.data_args.preprocessing_func

        if preprocessing_func is not None:
            if callable(preprocessing_func):
                pass

            elif ":" in preprocessing_func:
                # load func_name from "/path/to/file.py:func_name"
                preprocessing_func = import_from_path(preprocessing_func)
            else:
                # load from the registry
                preprocessing_func = (
                    PreprocessingFunctionRegistry.get_value_from_registry(
                        name=preprocessing_func
                    )
                )

            dataset = self.map(
                dataset,
                function=preprocessing_func,
                batched=False,
                num_proc=self.data_args.preprocessing_num_workers,
                desc="Applying custom func to the custom dataset",
            )

        return dataset

    def _remove_columns_from_dataset(
        self, dataset: Union[DatasetDict, Dataset]
    ) -> Union[DatasetDict, Dataset]:
        remove_columns = self.data_args.remove_columns

        if not remove_columns:
            remove_columns = self._get_remove_columns_from_dataset(dataset)

        if remove_columns is not None:
            dataset = self.map(
                dataset,
                batched=True,
                remove_columns=remove_columns,
                num_proc=self.data_args.preprocessing_num_workers,
                desc="Removing unneeded columns",
            )

        return dataset

    def _get_remove_columns_from_dataset(
        self, raw_dataset: Union[DatasetDict, Dataset]
    ) -> List[str]:
        """Remove redandant columns from the dataset for processing"""

        remove_columns = raw_dataset.column_names
        if isinstance(remove_columns, Dict):
            remove_columns = raw_dataset[list(raw_dataset.keys())[0]].column_names

        remove_columns = set(remove_columns)
        if self.text_column in remove_columns:
            remove_columns.remove(self.text_column)
        if self.PROMPT_KEY in remove_columns:
            remove_columns.remove(self.PROMPT_KEY)

        return list(remove_columns)
