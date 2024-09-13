from trl import SFTConfig as TRLSFTConfig
from trl import SFTTrainer as TRLSFTTrainer

from llmcompressor.transformers import TrainingArguments
from llmcompressor.transformers.finetune.session_mixin import SessionManagerMixIn
from llmcompressor.transformers.finetune.checkpoints_mixin import SafeCheckpointsMixin

__all__ = ["SFTTrainer"]


class SFTTrainer(SafeCheckpointsMixin, SessionManagerMixIn, TRLSFTTrainer):
    def __init__(self, *args, **kwargs):
        sft_config_args = kwargs.get("args")
        if (
            sft_config_args is not None
            and sft_config_args.__class__.__name__ == "TrainingArguments"
        ):
            kwargs["args"] = SFTConfig(**sft_config_args.to_dict())
        super().__init__(*args, **kwargs)

    def _prepare_dataset(self, dataset, *args, **kwargs):
        if "input_ids" in dataset.column_names:
            # dataset is already tokenized, skip preprocessing
            return dataset

        return super()._prepare_dataset(dataset, *args, **kwargs)


class SFTConfig(TrainingArguments, TRLSFTConfig):
    """
    This class is needed to wrap the llmcompressor.transformers.TrainingArguments
    and TRLSFTConfig classes. This allows for the use of arguments and
    configurations from both classes when training a model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
