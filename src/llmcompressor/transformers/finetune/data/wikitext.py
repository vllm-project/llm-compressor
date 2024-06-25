from llmcompressor.transformers.finetune.data import TextGenerationDataset


@TextGenerationDataset.register(name="wikitext")
class WikiTextDataset(TextGenerationDataset):
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
