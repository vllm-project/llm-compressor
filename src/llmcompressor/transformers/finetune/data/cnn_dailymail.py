from copy import deepcopy
from typing import TYPE_CHECKING

from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.typing import Processor

if TYPE_CHECKING:
    from llmcompressor.args import DatasetArguments


@TextGenerationDataset.register(name="cnn_dailymail")
class CNNDailyMailDataset(TextGenerationDataset):
    """
    Text generation class for the CNN/DailyMail dataset

    :param dataset_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param processor: processor or tokenizer to use on dataset
    """

    SAMPLE_TEMPLATE = "Article:\n{article}\n\n### Summarization:\n{highlights}\n"

    def __init__(
        self, dataset_args: "DatasetArguments", split: str, processor: Processor
    ):
        dataset_args = deepcopy(dataset_args)
        dataset_args.dataset = "cnn_dailymail"
        dataset_args.dataset_config_name = "3.0.0"

        super().__init__(dataset_args=dataset_args, split=split, processor=processor)

    def dataset_template(self, sample):
        return {
            "text": self.SAMPLE_TEMPLATE.format(
                article=sample["article"], highlights=sample["highlights"]
            )
        }
