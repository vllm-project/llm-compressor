from compressed_tensors.utils import replace_module
from transformers import PreTrainedModel

from llmcompressor.modeling.whisper import replace as replace_WhisperEncoder

__all__ = ["prepare_for_calibration"]

replacements = {
    "WhisperEncoder": replace_WhisperEncoder,
}


def prepare_for_calibration(model: PreTrainedModel) -> PreTrainedModel:
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in replacements:
            new_module = replacements[cls_name](module)
            replace_module(model, name, new_module)

    return model
