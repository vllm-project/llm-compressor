from copy import deepcopy
from typing import TYPE_CHECKING

from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.typing import Processor

if TYPE_CHECKING:
    from llmcompressor.args import DatasetArguments


@TextGenerationDataset.register(name="evolcodealpaca")
class EvolCodeAlpacaDataset(TextGenerationDataset):
    """
    Child text generation class for the Evol Code Alpaca dataset

    :param dataset_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param processor: processor or tokenizer to use on dataset
    """

    EVOL_ALPACA_TEMPLATE = (
        "Below is an instruction that describes a "
        "programming task. Write a program that appropriately "
        "completes the request.\n\n### Instruction:\n{instruction}"
        "\n\n### Response:\n"
    )

    def __init__(
        self, dataset_args: "DatasetArguments", split: str, processor: Processor
    ):
        dataset_args = deepcopy(dataset_args)
        dataset_args.dataset = "theblackcat102/evol-codealpaca-v1"
        dataset_args.text_column = "text"

        super().__init__(dataset_args, split=split, processor=processor)

    def dataset_template(self, sample):
        prompt = self.EVOL_ALPACA_TEMPLATE.format(instruction=sample["instruction"])
        text = prompt
        if "output" in text:
            text += sample["output"]

        return {
            "text": text,
            self.PROMPT_KEY: prompt,
        }
