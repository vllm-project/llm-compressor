from trl import SFTTrainer as TRLSFTTrainer

from llmcompressor.transformers.finetune.session_mixin import SessionManagerMixIn

__all__ = ["SFTTrainer"]


class SFTTrainer(SessionManagerMixIn, TRLSFTTrainer):
    def _prepare_dataset(self, dataset, *args, **kwargs):
        if "input_ids" in dataset.column_names:
            # dataset is already tokenized, skip preprocessing
            return dataset

        return super()._prepare_dataset(dataset, *args, **kwargs)
