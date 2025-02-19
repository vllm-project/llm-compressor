import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM

from llmcompressor.modifiers.pruning import SparseGPTModifier
from llmcompressor.transformers.finetune.data.data_helpers import (
    format_calibration_data,
)
from llmcompressor.utils.pytorch.module import get_layers


@pytest.mark.integration
def test_infer_owl_layer_sparsity():
    target_sparsity = 0.7
    vocab_size = 512
    seq_len = 2048
    ds_size = 16

    modifier = SparseGPTModifier(
        sparsity=0.7, sparsity_profile="owl", owl_m=5, owl_lmbda=0.05
    )
    model = AutoModelForCausalLM.from_pretrained("Xenova/llama2.c-stories15M")

    dataset = Dataset.from_dict(
        {"input_ids": torch.randint(0, vocab_size, (ds_size, seq_len))}
    )
    dataloader = format_calibration_data(dataset)

    sequential_targets = modifier._infer_sequential_targets(model)
    layers = get_layers(sequential_targets, model)
    sparsities = modifier._infer_owl_layer_sparsity(model, layers, dataloader)
    assert sparsities.keys() == layers.keys()

    for sparsity in sparsities.values():
        assert sparsity == pytest.approx(target_sparsity, abs=0.1)
