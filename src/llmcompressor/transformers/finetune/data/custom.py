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
        self.preprocessing_func = data_args.preprocessing_func
        self.remove_columns = data_args.remove_columns

    def get_raw_dataset(self, *_ignore, **__ignore) -> Union[DatasetDict, Dataset]:
        """Get the raw dataset and apply preprocessing func if provided"""

        dataset = self.data_args.dataset
        if isinstance(dataset, DatasetDict) or isinstance(dataset, Dataset):
            # user passed in an already instantiated dataset, just use it directly
            raw_dataset = dataset
        else:
            # dataset must be loaded from file or HF Hub
            raw_dataset = super().get_raw_dataset()

        if self.preprocessing_func is not None:
            if callable(self.preprocessing_func):
                func = self.preprocessing_func
            elif ":" in self.preprocessing_func:
                # load func_name from "/path/to/file.py:func_name"
                func = import_from_path(self.preprocessing_func)
            else:
                # load from the registry
                func = PreprocessingFunctionRegistry.get_value_from_registry(
                    name=self.preprocessing_func
                )

            raw_dataset = self.map(
                raw_dataset,
                function=func,
                batched=False,
                num_proc=self.data_args.preprocessing_num_workers,
                desc="Applying custom func to the custom dataset",
            )

        self.remove_columns = (
            self.remove_columns or self.get_remove_columns_from_dataset(raw_dataset)
        )

        if self.remove_columns is not None:
            raw_dataset = self.map(
                raw_dataset,
                batched=True,
                remove_columns=self.remove_columns,
                num_proc=self.data_args.preprocessing_num_workers,
                desc="Removing unneeded columns",
            )

        return raw_dataset

    def get_remove_columns_from_dataset(
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
