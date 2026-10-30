"""Temporary architecture adapters for Transformers' MtpModel.

Remove each adapter once upstream supports its config and checkpoint round-trip.
Keep model loading, offloading, and saving in mtp.py.
"""

from contextlib import ExitStack
from copy import deepcopy

from compressed_tensors.utils import patch_attr
from transformers import PretrainedConfig, PreTrainedModel, modeling_layers
from transformers.modeling_layers import MtpModel

from .qwen3_5_for_conditional_generation import Qwen3_5MtpModel

_MTP_ADAPTERS = {"qwen3_5_text": Qwen3_5MtpModel}


def get_mtp_config(model: PreTrainedModel) -> PretrainedConfig:
    adapter = _MTP_ADAPTERS.get(model.config.get_text_config().model_type)
    if adapter is None:
        return model.config.get_mtp_config()
    return adapter.get_mtp_config(model)


def load_mtp_model(model: PreTrainedModel, config: PretrainedConfig, device_map=None):
    adapter = _MTP_ADAPTERS.get(model.config.get_text_config().model_type)
    if adapter is None:
        return MtpModel.from_pretrained(model, device_map=device_map)

    # Alias only on a temporary config; don't serialize methods or overwrite
    # the backbone's original layer types/count and checkpoint-specific fields.
    load_config = deepcopy(model.config)
    load_config.get_mtp_config = lambda: deepcopy(config)
    load_config.get_text_config().num_mtp_layers = config.num_hidden_layers
    with ExitStack() as stack:
        stack.enter_context(patch_attr(model, "config", load_config))
        if not callable(getattr(MtpModel, "get_weight_conversions", None)):
            # Transformers 5.15 constructs its mapping inline instead.
            original = modeling_layers.get_model_conversion_mapping

            def get_conversions(mtp_model, *args, **kwargs):
                if isinstance(mtp_model, adapter):
                    return adapter.get_mtp_weight_conversions()
                return original(mtp_model, *args, **kwargs)

            stack.enter_context(
                patch_attr(
                    modeling_layers, "get_model_conversion_mapping", get_conversions
                )
            )
        return adapter.from_pretrained(model, device_map=device_map)
