from copy import deepcopy
from typing import Optional

from llmcompressor.transformers.finetune.data import TextGenerationDataset


@TextGenerationDataset.register(name="open_platypus")
class OpenPlatypusDataset(TextGenerationDataset):
    """
    Child text generation class for the Open Platypus dataset

    :param data_args: configuration settings for dataset loading
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

    def __init__(self, data_args, split, processor):
        data_args = deepcopy(data_args)
        data_args.dataset = "garage-bAInd/Open-Platypus"
        super().__init__(
            text_column="text", data_args=data_args, split=split, processor=processor
        )

    def get_raw_dataset(self, cache_dir: Optional[str] = None):
        """
        Load the raw dataset from Hugging Face, using cached copy if available.
        Additionally reformats the entries to fit the alpaca template.

        :param cache_dir: disk location to search for cached dataset
        :return: the requested dataset
        """
        raw_dataset = super().get_raw_dataset(cache_dir=cache_dir)

        # helper fn for restructuring each dataset entry using the alpaca template
        def restructure_fn(sample):
            if "input" in sample and sample["input"] != "":
                sample["text"] = self.ALPACA_TEMPLATE["prompt_input"].format(
                    instruction=sample["instruction"], input=sample["input"]
                )
            else:
                sample["text"] = self.ALPACA_TEMPLATE["prompt_no_input"].format(
                    instruction=sample["instruction"]
                )

            sample[self.PROMPT_KEY] = sample["text"]
            if "output" in sample:
                sample["text"] += sample["output"]
            return sample

        raw_dataset = self.map(
            raw_dataset,
            function=restructure_fn,
            batched=False,
            remove_columns=["input", "output", "instruction", "data_source"],
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Restructuring Platypus Dataset",
        )
        return raw_dataset
