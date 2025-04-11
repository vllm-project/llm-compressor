from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict

from datasets.formatting.formatting import LazyRow
from loguru import logger

from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.transformers.finetune.data.base import get_columns
from llmcompressor.typing import DatasetType, Processor

if TYPE_CHECKING:
    from llmcompressor.transformers import DataTrainingArguments as DataArgs


@TextGenerationDataset.register(name="peoples_speech")
class PeoplesSpeech(TextGenerationDataset):
    """
    ML Commons People's Speech audio dataset

    Unfortunately, due to the specialized nature of audio model preprocessing, some
    model specific code must be defined here. This dataset has been tested with the
    WhisperForConditionalGeneration and Qwen2AudioForConditionalGeneration model classes

    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param processor: processor or tokenizer to use on dataset
    """

    def __init__(self, dataset_args: "DataArgs", split: str, processor: Processor):
        dataset_args = deepcopy(dataset_args)
        dataset_args.dataset = "MLCommons/peoples_speech"
        dataset_args.dataset_config_name = "test"
        if not dataset_args.overwrite_cache:
            logger.warning(
                "Because audio processors are more complex, dataset mapping functions "
                "vary with model architecture and their results cannot be cached. "
                "Setting overwrite_cache=True"
            )
            dataset_args.overwrite_cache = True
        self.processor_type = processor.__class__.__name__

        super().__init__(dataset_args=dataset_args, split=split, processor=processor)

    def dataset_template(self, example):
        audio = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]

        if self.processor_type == "Qwen2AudioProcessor":
            messages = [
                {"role": "user", "content": [{"audio": None}]},
                {"role": "user", "content": [{"text": "What did the person say?"}]},
            ]
            text = self.processor.apply_chat_template(messages)
            return {"audios": [audio], "sampling_rate": sampling_rate, "text": text}

        else:
            # chat template decoder ids are appended later by self.processor.__call__
            text = " " + example["text"].capitalize()
            return {"audio": audio, "sampling_rate": sampling_rate, "text": text}

    def filter_tokenizer_args(self, dataset: DatasetType) -> DatasetType:
        if self.processor_type == "WhisperProcessor":
            tokenizer_args = ["audio", "sampling_rate", "text"]
            column_names = get_columns(dataset)

            return dataset.remove_columns(list(set(column_names) - set(tokenizer_args)))

        else:
            return super().filter_tokenizer_args(dataset)

    def tokenize(self, data: LazyRow) -> Dict[str, Any]:
        if self.processor_type == "WhisperProcessor":
            inputs = self.processor(
                audio=data["audio"],
                sampling_rate=data["sampling_rate"],
                text=data["text"],
                add_special_tokens=True,
                return_tensors="pt",
            )

            # TODO: inputs["input_features"] is a float dtype, which may conflict with
            # the dtype of the model. Add logic to in data pipeline to move inputs to
            # the matching model device and dtype
            inputs["decoder_input_ids"] = inputs["labels"]
            del inputs["labels"]

            return inputs

        else:
            return super().tokenize(data)
