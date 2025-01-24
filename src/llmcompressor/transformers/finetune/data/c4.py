from copy import deepcopy
from typing import TYPE_CHECKING

from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.typing import Processor

if TYPE_CHECKING:
    from llmcompressor.transformers.utils.arg_parser import DatasetArguments


@TextGenerationDataset.register(name="c4")
class C4Dataset(TextGenerationDataset):
    """
    Child text generation class for the C4 dataset

    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param processor: processor or tokenizer to use on dataset
    """

    def __init__(self, data_args: "DatasetArguments", split: str, processor: Processor):
        data_args = deepcopy(data_args)
        data_args.dataset = "allenai/c4"
        data_args.text_column = "text"

        super().__init__(data_args=data_args, split=split, processor=processor)
