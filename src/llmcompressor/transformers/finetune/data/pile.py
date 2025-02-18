from copy import deepcopy
from typing import TYPE_CHECKING

from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.typing import Processor

if TYPE_CHECKING:
    from llmcompressor.args import DatasetArguments


@TextGenerationDataset.register(name="pile_eval")
class PileEvalDataset(TextGenerationDataset):
    """
    Child text generation class for the PileEval dataset
    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param tokenizer: tokenizer to use on dataset
    """

    def __init__(self, data_args: "DatasetArguments", split: str, processor: Processor):
        data_args = deepcopy(data_args)
        data_args.text_column = "text"
        data_args.dataset = "mit-han-lab/pile-val-backup"
        super().__init__(data_args=data_args, split=split, processor=processor)

    def dataset_template(self, sample):
        return {
            "text": self.processor.apply_chat_template(
                sample["text"].strip(),
            ),
        }
