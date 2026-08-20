import torch
import torch.nn as nn
from loguru import logger

from llmcompressor.core import State
from llmcompressor.modifiers.quantization import QuantizationModifier


class _AuxBlock(nn.Module):
    """A submodule forward decorated with @torch.inference_mode(), mirroring
    a vendored model forward that was written for inference only use before
    being adapted for calibration (see deepseekv32/model.py prior to the
    fix for #2745)."""

    def __init__(self, hidden):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)

    @torch.inference_mode()
    def forward(self, x):
        return self.proj(x)


class _ToyModel(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.aux_block = _AuxBlock(hidden)

    def forward(self, x):
        return self.aux_block(x)


def test_start_calibration_warns_on_inference_mode_forward():
    """
    Regression test for https://github.com/vllm-project/llm-compressor/issues/2745

    A submodule forward decorated with @torch.inference_mode() produces
    output tensors that stay tagged as inference tensors even after the
    call returns. If a later stage of calibration, for example GPTQ's
    weight writeback or the offload cache, tries to update a tensor
    derived from that call in place, the update crashes, potentially after
    a long compression run has already made it through most of the model.
    Warn as soon as calibration starts instead of letting that happen deep
    into a run.
    """
    model = _ToyModel()
    state = State(model=model)
    modifier = QuantizationModifier(targets=["Linear"], scheme="W4A16")

    logs = []
    handler_id = logger.add(logs.append, format="{message}", level="WARNING")
    try:
        modifier.on_initialize(state)
        modifier.on_calibration_start(state, None)
    finally:
        logger.remove(handler_id)

    assert any("inference_mode" in log and "aux_block" in log for log in logs), (
        f"expected a warning naming aux_block, got: {logs}"
    )
