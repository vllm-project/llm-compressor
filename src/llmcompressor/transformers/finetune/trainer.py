from transformers import Trainer as HFTransformersTrainer

from llmcompressor.transformers.finetune.checkpoints_mixin import SafeCheckpointsMixin
from llmcompressor.transformers.finetune.session_mixin import SessionManagerMixIn

__all__ = ["Trainer"]


class Trainer(SafeCheckpointsMixin, SessionManagerMixIn, HFTransformersTrainer):
    pass
