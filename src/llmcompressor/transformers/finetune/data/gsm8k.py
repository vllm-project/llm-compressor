from copy import deepcopy
from typing import TYPE_CHECKING

from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.typing import Processor

if TYPE_CHECKING:
    from llmcompressor.args import DatasetArguments


@TextGenerationDataset.register(name="gsm8k")
class GSM8KDataset(TextGenerationDataset):
    """
    Child text generation class for the Grade School Math 8k dataset

    :param dataset_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param processor: processor or tokenizer to use on dataset
    """

    GSM_TEMPLATE = "Question: {question}\nAnswer:"

    def __init__(
        self, dataset_args: "DatasetArguments", split: str, processor: Processor
    ):
        dataset_args = deepcopy(dataset_args)
        dataset_args.dataset = "gsm8k"
        dataset_args.text_column = "text"

        super().__init__(dataset_args=dataset_args, split=split, processor=processor)

    def dataset_template(self, sample):
        prompt = self.GSM_TEMPLATE.format(question=sample["question"])
        text = prompt
        if "answer" in sample:
            text += " " + sample["answer"]

        return {
            "text": text,
            self.PROMPT_KEY: prompt,
        }
