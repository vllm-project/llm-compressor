from copy import deepcopy
from typing import Optional

from llmcompressor.transformers.finetune.data import TextGenerationDataset


@TextGenerationDataset.register(name="cnn_dailymail")
class CNNDailyMailDataset(TextGenerationDataset):
    """
    Text generation class for the CNN/DailyMail dataset

    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param processor: processor or tokenizer to use on dataset
    """

    SAMPLE_TEMPLATE = "Article:\n{article}\n\n### Summarization:\n{highlights}\n"

    def __init__(self, data_args, split, processor):
        data_args = deepcopy(data_args)
        data_args.dataset = "cnn_dailymail"
        data_args.dataset_config_name = "3.0.0"

        super().__init__(
            text_column="text", data_args=data_args, split=split, processor=processor
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
            sample["text"] = self.SAMPLE_TEMPLATE.format(
                article=sample["article"], highlights=sample["highlights"]
            )

            return sample

        raw_dataset = self.map(
            raw_dataset,
            function=restructure_fn,
            batched=False,
            remove_columns=["article", "highlights", "id"],
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Restructuring CNN/DailyMail Dataset",
        )
        return raw_dataset
