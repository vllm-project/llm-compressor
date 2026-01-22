import pytest
import torch
from accelerate import dispatch_model
from compressed_tensors import align_module_device, update_offload_parameter
from transformers import AutoModelForCausalLM

from llmcompressor.utils import targets_embeddings, untie_word_embeddings
from tests.testing_utils import requires_gpu


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
    model_path = "nm-testing/tinysmokellama-3.2"
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
    model_path = "nm-testing/tinysmokellama-3.2"
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
