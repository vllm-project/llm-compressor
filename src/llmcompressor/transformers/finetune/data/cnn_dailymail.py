from copy import deepcopy

from llmcompressor.transformers.finetune.data import DataTrainingArguments as DataArgs
from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.utils import Processor


@TextGenerationDataset.register(name="cnn_dailymail")
class CNNDailyMailDataset(TextGenerationDataset):
    """
    Text generation class for the CNN/DailyMail dataset

    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param processor: processor or tokenizer to use on dataset
    """

    SAMPLE_TEMPLATE = "Article:\n{article}\n\n### Summarization:\n{highlights}\n"

    def __init__(self, data_args: DataArgs, split: str, processor: Processor):
        data_args = deepcopy(data_args)
        data_args.dataset = "cnn_dailymail"
        data_args.dataset_config_name = "3.0.0"

        super().__init__(data_args=data_args, split=split, processor=processor)

    def dataset_template(self, sample):
        return {
            "text": self.SAMPLE_TEMPLATE.format(
                article=sample["article"], highlights=sample["highlights"]
            )
        }
