from copy import deepcopy

from llmcompressor.transformers.finetune.data import DataTrainingArguments as DataArgs
from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.utils import Processor


@TextGenerationDataset.register(name="gsm8k")
class GSM8KDataset(TextGenerationDataset):
    """
    Child text generation class for the Grade School Math 8k dataset

    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param processor: processor or tokenizer to use on dataset
    """

    GSM_TEMPLATE = "Question: {question}\nAnswer:"

    def __init__(self, data_args: DataArgs, split: str, processor: Processor):
        data_args = deepcopy(data_args)
        data_args.dataset = "gsm8k"
        data_args.text_column = "text"

        super().__init__(data_args=data_args, split=split, processor=processor)

    def dataset_template(self, sample):
        prompt = self.GSM_TEMPLATE.format(question=sample["question"])
        text = prompt
        if "answer" in sample:
            text += " " + sample["answer"]

        return {
            "text": text,
            self.PROMPT_KEY: prompt,
        }
