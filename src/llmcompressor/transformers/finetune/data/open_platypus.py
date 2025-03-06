from copy import deepcopy
from typing import TYPE_CHECKING

from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.typing import Processor

if TYPE_CHECKING:
    from llmcompressor.args import DatasetArguments


@TextGenerationDataset.register(name="open_platypus")
class OpenPlatypusDataset(TextGenerationDataset):
    """
    Child text generation class for the Open Platypus dataset

    :param dataset_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param processor: processor or tokenizer to use on dataset
    """

    ALPACA_TEMPLATE = {
        "prompt_input": "Below is an instruction that describes a task, paired with an "
        "input that provides further context. Write a response that appropriately "
        "completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n"
        "{input}\n\n### Response:\n",
        "prompt_no_input": "Below is an instruction that describes a task. Write a "
        "response that appropriately completes the request.\n\n### Instruction:\n{"
        "instruction}\n\n### Response:\n",
    }

    def __init__(
        self, dataset_args: "DatasetArguments", split: str, processor: Processor
    ):
        dataset_args = deepcopy(dataset_args)
        dataset_args.dataset = "garage-bAInd/Open-Platypus"
        dataset_args.text_column = "text"
        super().__init__(dataset_args=dataset_args, split=split, processor=processor)

    def dataset_template(self, sample):
        if "input" in sample and sample["input"] != "":
            prompt = self.ALPACA_TEMPLATE["prompt_input"].format(
                instruction=sample["instruction"], input=sample["input"]
            )
        else:
            prompt = self.ALPACA_TEMPLATE["prompt_no_input"].format(
                instruction=sample["instruction"]
            )

        text = prompt
        if "output" in sample:
            text += sample["output"]

        return {
            "text": text,
            self.PROMPT_KEY: prompt,
        }
