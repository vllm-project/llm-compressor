from copy import deepcopy

from llmcompressor.transformers.finetune.data import TextGenerationDataset


@TextGenerationDataset.register(name="c4")
class C4Dataset(TextGenerationDataset):
    """
    Child text generation class for the C4 dataset

    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param tokenizer: tokenizer to use on dataset
    """

    def __init__(self, data_args, split, tokenizer):
        data_args = deepcopy(data_args)
        data_args.dataset = "allenai/c4"
        super().__init__(
            text_column="text", data_args=data_args, split=split, tokenizer=tokenizer
        )
