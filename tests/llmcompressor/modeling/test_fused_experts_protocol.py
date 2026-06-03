import pytest
import torch
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts

from llmcompressor.modeling.moe.helpers import FusedExpertsProtocol


@pytest.mark.parametrize(
    "config_cls,experts_cls",
    [
        (DeepseekV4Config, DeepseekV4Experts),
        (Qwen3VLMoeTextConfig, Qwen3VLMoeTextExperts),
    ],
)
def test_protocol_isinstance(config_cls, experts_cls):
    config = config_cls()
    with torch.device("meta"):
        experts = experts_cls(config)

    assert isinstance(experts, FusedExpertsProtocol)
