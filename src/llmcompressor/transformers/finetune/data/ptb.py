from copy import deepcopy

from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.transformers.finetune.data.data_args import DataTrainingArguments
from llmcompressor.utils import Processor


@TextGenerationDataset.register(name="ptb")
class PtbDataset(TextGenerationDataset):
    """
    Child text generation class for the PTB dataset

    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param processor: processor or tokenizer to use on dataset
    """

    def __init__(
        self,
        data_args: DataTrainingArguments,
        split: str,
        processor: Processor,
    ):
        data_args = deepcopy(data_args)
        data_args.dataset = "ptb_text_only"
        data_args.text_column = "sentence"

        super().__init__(
            data_args=data_args,
            split=split,
            processor=processor,
        )
