from copy import deepcopy

from llmcompressor.transformers.finetune.data import TextGenerationDataset


@TextGenerationDataset.register(name="ptb")
class PtbDataset(TextGenerationDataset):
    """
    Child text generation class for the PTB dataset

    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param tokenizer: tokenizer to use on dataset
    """

    def __init__(self, data_args, split, tokenizer):
        data_args = deepcopy(data_args)
        data_args.dataset = "ptb_text_only"
        super().__init__(
            text_column="sentence",
            data_args=data_args,
            split=split,
            tokenizer=tokenizer,
        )
