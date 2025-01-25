from copy import deepcopy
from typing import TYPE_CHECKING

from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.typing import Processor

if TYPE_CHECKING:
    from llmcompressor.transformers import DataTrainingArguments as DataArgs


@TextGenerationDataset.register(name="peoples_speech")
class PeoplesSpeech(TextGenerationDataset):
    """
    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param processor: processor or tokenizer to use on dataset
    """

    def __init__(self, data_args: "DataArgs", split: str, processor: Processor):
        data_args = deepcopy(data_args)
        data_args.dataset = "MLCommons/peoples_speech"
        data_args.dataset_config_name = "test"

        super().__init__(data_args=data_args, split=split, processor=processor)

    def dataset_template(self, example):
        return {
            "array": example["audio"]["array"],
            "sampling_rate": example["audio"]["sampling_rate"],
            "text": " " + example["text"].capitalize(),
        }
