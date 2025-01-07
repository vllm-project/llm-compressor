from transformers import Trainer as HFTransformersTrainer

from llmcompressor.transformers.finetune.session_mixin import SessionManagerMixIn

__all__ = ["Trainer", "Calibrator"]


class Trainer(SessionManagerMixIn, HFTransformersTrainer):
    pass


class Calibrator(SessionManagerMixIn):
    pass