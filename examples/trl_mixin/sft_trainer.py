from typing import Dict, Optional

from trl import SFTConfig as TRLSFTConfig
from trl import SFTTrainer as TRLSFTTrainer

from llmcompressor.transformers.finetune.session_mixin import SessionManagerMixIn

__all__ = ["SFTTrainer"]


class SFTTrainer(SessionManagerMixIn, TRLSFTTrainer):
    def __init__(self, trl_sft_config_args: Optional[Dict] = None, *args, **kwargs):
        if trl_sft_config_args is not None:
            kwargs["args"] = TRLSFTConfig(**trl_sft_config_args)
        super().__init__(*args, **kwargs)

    def _prepare_dataset(self, dataset, *args, **kwargs):
        if "input_ids" in dataset.column_names:
            # dataset is already tokenized, skip preprocessing
            return dataset

        return super()._prepare_dataset(dataset, *args, **kwargs)
