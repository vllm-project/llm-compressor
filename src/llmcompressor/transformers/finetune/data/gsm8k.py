from copy import deepcopy
from typing import Optional

from llmcompressor.transformers.finetune.data import TextGenerationDataset


@TextGenerationDataset.register(name="gsm8k")
class GSM8KDataset(TextGenerationDataset):
    """
    Child text generation class for the Grade School Math 8k dataset

    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param processor: processor or tokenizer to use on dataset
    """

    GSM_TEMPLATE = "Question: {question}\nAnswer:"

    def __init__(self, data_args, split, processor):
        data_args = deepcopy(data_args)
        data_args.dataset = "gsm8k"
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

        # helper fn for restructuring each dataset entry using the gsm template
        def restructure_fn(sample):
            sample["text"] = self.GSM_TEMPLATE.format(question=sample["question"])
            sample[self.PROMPT_KEY] = sample["text"]
            if "answer" in sample:
                sample["text"] += " " + sample["answer"]
            return sample

        raw_dataset = self.map(
            raw_dataset,
            function=restructure_fn,
            batched=False,
            remove_columns=["question", "answer"],
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Restructuring GSM Dataset",
        )
        return raw_dataset
