import pytest
import torch
from accelerate import dispatch_model
from compressed_tensors import align_module_device, update_offload_parameter
from torch import nn
from transformers import AutoModelForCausalLM, PretrainedConfig

from llmcompressor.utils import targets_embeddings, untie_word_embeddings
from tests.testing_utils import requires_gpu


def test_untie_word_embeddings_updates_nested_config():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = PretrainedConfig(tie_word_embeddings=True)
            self.config.text_config = PretrainedConfig(tie_word_embeddings=True)
            self.embed_tokens = nn.Embedding(4, 3)
            self.lm_head = nn.Linear(3, 4, bias=False)
            self.lm_head.weight = self.embed_tokens.weight

        def get_input_embeddings(self):
            return self.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

    model = TinyModel()
    untie_word_embeddings(model)

    assert model.lm_head.weight is not model.embed_tokens.weight
    assert model.config.tie_word_embeddings is False
    assert model.config.text_config.tie_word_embeddings is False


@pytest.mark.parametrize(
    "offload,dtype,tie_word_embeddings,device",
    [
        (False, torch.float16, False, "cpu"),
        (False, torch.float32, False, "cpu"),
        (True, torch.float32, False, "cpu"),
        (False, torch.float16, True, "cpu"),
        (False, torch.float32, True, "cpu"),
        (True, torch.float16, True, "cpu"),
        (True, torch.float32, True, "cpu"),
    ],
)
def test_untie_word_embeddings(offload, dtype, tie_word_embeddings, device):
    """
    Test whether model offloading breaks tied/untied embeddings
    """
    # load model
    model_path = "inference-optimization/Llama-3.2-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    if offload:
        model = dispatch_model(model, {"": device}, force_hooks=True)
    else:
        model = model.to(device)

    if not tie_word_embeddings:
        untie_word_embeddings(model)

    # modify lm head
    with torch.no_grad(), align_module_device(model.lm_head):
        update_offload_parameter(model.lm_head, "weight", model.lm_head.weight + 1)

    with (
        align_module_device(model.lm_head),
        align_module_device(model.model.embed_tokens),
    ):
        if tie_word_embeddings:
            assert model.lm_head.weight is model.model.embed_tokens.weight
            assert model.config.tie_word_embeddings
        else:
            assert model.lm_head.weight is not model.model.embed_tokens.weight
            assert not model.config.tie_word_embeddings


@requires_gpu
@pytest.mark.parametrize(
    "offload,dtype,tie_word_embeddings,device",
    [
        (False, torch.float32, False, "cuda:0"),
        (False, torch.float32, True, "cuda:0"),
    ],
)
def test_untie_word_embeddings_gpu(offload, dtype, tie_word_embeddings, device):
    test_untie_word_embeddings(offload, dtype, tie_word_embeddings, device)


def test_targets_embeddings():
    model_path = "inference-optimization/Llama-3.2-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_path)

    targets = {"embed_tokens": model.model.embed_tokens}.items()
    assert targets_embeddings(model, targets, check_input=True, check_output=True)
    assert targets_embeddings(model, targets, check_input=True, check_output=False)
    assert not targets_embeddings(model, targets, check_input=False, check_output=True)
    assert not targets_embeddings(model, targets, check_input=False, check_output=False)

    targets = {"lm_head": model.lm_head}.items()
    assert targets_embeddings(model, targets, check_input=True, check_output=True)
    assert not targets_embeddings(model, targets, check_input=True, check_output=False)
    assert targets_embeddings(model, targets, check_input=False, check_output=True)
    assert not targets_embeddings(model, targets, check_input=False, check_output=False)

    targets = {}.items()
    assert not targets_embeddings(model, targets, check_input=True, check_output=True)
    assert not targets_embeddings(model, targets, check_input=True, check_output=False)
    assert not targets_embeddings(model, targets, check_input=False, check_output=True)
    assert not targets_embeddings(model, targets, check_input=False, check_output=False)
