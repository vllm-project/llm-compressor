from typing import Optional

from llmcompressor.transformers.finetune.data import TextGenerationDataset


@TextGenerationDataset.register(name="mit-han-lab/pile-val-backup", alias=["pile-eval"])
class PileEvalDataset(TextGenerationDataset):
    """
    Child text generation class for the Open Platypus dataset

    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param tokenizer: tokenizer to use on dataset
    """

    def __init__(self, data_args, split, tokenizer):
        super().__init__(
            text_column="text", data_args=data_args, split=split, tokenizer=tokenizer
        )

    def get_raw_dataset(self, cache_dir: Optional[str] = None):
        """
        Load the raw dataset from Hugging Face, using cached copy if available.
        Additionally reformats the entries to fit the template.

        :param cache_dir: disk location to search for cached dataset
        :return: the requested dataset
        """
        raw_dataset = super().get_raw_dataset(cache_dir=cache_dir)

        def restructure_fn(sample):
            sample["text"] = sample["text"].strip()
            return sample

        raw_dataset = self.map(
            raw_dataset,
            function=restructure_fn,
            batched=False,
            remove_columns=["meta"],
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Restructuring Pile Dataset",
        )
        return raw_dataset
