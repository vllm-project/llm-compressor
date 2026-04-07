import torch
import torch.fx
from llmcompressor.pipelines.sequential.helpers import get_sequential_ancestors, topological_partition, trace_subgraphs, Subgraph
from llmcompressor.utils.pytorch.module import get_no_split_params
from llmcompressor.utils.dev import skip_weights_download
from tests.llmcompressor.transformers.tracing.test_models import get_target_modules

from compressed_tensors.utils.match import match_named_modules

from contextlib import nullcontext
from transformers import (
    AutoModelForCausalLM,
)

import math
import pytest



class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(torch.nn.Linear(10, 20), torch.nn.ReLU())
        self.fc = torch.nn.Linear(20, 5)

    def forward(self, x):
        x = self.seq(x)
        return self.fc(x)

class DummyModelMultipleSequentialLayers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 10)
        self.layer3 = torch.nn.Linear(10, 10)
        self.layer4 = torch.nn.Linear(10, 10)
        self.layer5 = torch.nn.Linear(10, 10)
        self.layer6 = torch.nn.Linear(10, 10)
        
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
        
def test_get_sequential_ancestors():
    model = DummyModel()

    assert get_sequential_ancestors(model, set()) == set()
    assert get_sequential_ancestors(model, {model}) == set()
    assert get_sequential_ancestors(model, {model.fc}) == {model}
    assert get_sequential_ancestors(model, {model.seq[0]}) == {model, model.seq}
    assert get_sequential_ancestors(model, {model.seq[1]}) == {model, model.seq}

def test_topological_partition_default():
    model = DummyModelMultipleSequentialLayers()
    targets = {model.layer1, model.layer2, model.layer3, model.layer4, model.layer5, model.layer6}
    gm = torch.fx.symbolic_trace(model)
    
    
    assert len(topological_partition(gm, targets)) == 7

def test_topological_partition_multiple_targets():
    model = DummyModelMultipleSequentialLayers()
    gm = torch.fx.symbolic_trace(model)
    targets = {model.layer1, model.layer2, model.layer3, model.layer4, model.layer5, model.layer6}

    
    assert len(topological_partition(gm, targets, 2)) == 4
    

def test_topological_partition_invalid():
    model = DummyModelMultipleSequentialLayers()
    gm = torch.fx.symbolic_trace(model)
    targets = {model.layer1, model.layer2, model.layer3, model.layer4, model.layer5, model.layer6}

    
    with pytest.raises(ValueError):
        topological_partition(gm, targets, 0)


@pytest.mark.parametrize(
    "targets_per_subgraph",
    [   
        (1),
        (2),
        (3),
        (4),
        (5)
    ]
    
)

def test_trace_subgraphs(targets_per_subgraph):
    model_class = AutoModelForCausalLM
    model_id = "Qwen/Qwen3-0.6B"
    device_map = "cpu"
    trust_remote_code = True
    skip_weights = True
    sequential_targets = ['Qwen3DecoderLayer']
    tracer_ignore_dataset_args = [
            "_update_causal_mask",
            "create_causal_mask",
            "_update_mamba_mask",
            "make_causal_mask",
            "get_causal_mask",
            "mask_interface",
            "mask_function",
            "_prepare_4d_causal_attention_mask",
            "_prepare_fsmt_decoder_inputs",
            "_prepare_4d_causal_attention_mask_with_cache_position",
            "_update_linear_attn_mask",
            "project_per_layer_inputs",
        ],
    
    with skip_weights_download(AutoModelForCausalLM) if skip_weights else nullcontext():
        model = model_class.from_pretrained(
            model_id,
            device_map=device_map,
            dtype="auto",
            trust_remote_code=trust_remote_code,
        )
    
    sample = {"input_ids": torch.zeros(1, 32, dtype=torch.long)}
    
    
    if targets_per_subgraph == 0:
        with pytest.raises(ValueError):
            subgraphs = trace_subgraphs(
                model, sample, sequential_targets,tracer_ignore_dataset_args, targets_per_subgraph
            )
    else:
        subgraphs = trace_subgraphs(
                model, sample, sequential_targets,tracer_ignore_dataset_args, targets_per_subgraph
            )
        
        maxSubgraphs = trace_subgraphs(
            model, sample, sequential_targets,tracer_ignore_dataset_args, 1
        )
        
        test_subgraphs_contain_targets(subgraphs, model, 'Qwen3DecoderLayer')
        assert len(subgraphs) == (len(maxSubgraphs) - 1 ) // targets_per_subgraph + 1
        


def test_subgraphs_contain_targets(subgraphs, model, target_class_name):
    # skip prefix subgraph
    for subgraph in subgraphs[1:]:  
        modules = subgraph.submodules(model)
        assert any(
            type(m).__name__ == target_class_name
            for m in modules
        ), f"Expected each non-prefix subgraph to contain at least one {target_class_name}"