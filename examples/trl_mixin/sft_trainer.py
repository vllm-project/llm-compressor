from trl import SFTConfig as TRLSFTConfig
from trl import SFTTrainer as TRLSFTTrainer

from llmcompressor.transformers.finetune.session_mixin import SessionManagerMixIn

__all__ = ["SFTTrainer"]


class SFTTrainer(SessionManagerMixIn, TRLSFTTrainer):
    def __init__(self, *args, **kwargs):
        sft_config_args = kwargs.get("args")
        if sft_config_args is not None:
            kwargs["args"] = TRLSFTConfig(**sft_config_args)

        super().__init__(*args, **kwargs)

    def _prepare_dataset(self, dataset, *args, **kwargs):
        if "input_ids" in dataset.column_names:
            # dataset is already tokenized, skip preprocessing
            return dataset

        return super()._prepare_dataset(dataset, *args, **kwargs)
