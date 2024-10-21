# torchrun --nproc_per_node=1 -m pytest test_hooks.py

import contextlib
import torch
import pytest
import torch.distributed as dist
from transformers import AutoModel

from accelerate import cpu_offload
from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload
from torch.distributed.fsdp._common_utils import (
    HandleTrainingState,
    TrainingState,
)
from compressed_tensors import is_module_offloaded, update_parameter_data

dist.init_process_group(backend="nccl", rank=0, world_size=1)
torch.set_default_device("cuda:0")

def load_model(fsdp, offload):
    model = AutoModel.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if fsdp:
        model = FullyShardedDataParallel(
            model,
            cpu_offload=CPUOffload(offload_params=True) if offload else None
        )
    
    elif offload:
        model = cpu_offload(model)

    return model

parameterize = pytest.mark.parametrize("fsdp,offload", [
    (False, False),
    (False, True),
    (True, False),
    (True, True),
])


@contextlib.contextmanager
def accelerate_writeback_context(module):  # only for this test, do not actually use
    module._hf_hook.pre_forward(module)

    yield

    for name, param in module.named_parameters():
        update_parameter_data(module, param.data, name)

    module._hf_hook.post_forward(module, None)

@contextlib.contextmanager
def modify_params_context(model, module):
    if isinstance(model, FullyShardedDataParallel):
        with (
            model._use_training_state(TrainingState.IDLE, HandleTrainingState.IDLE),
            FullyShardedDataParallel.summon_full_params(model)
        ):
            yield

    elif is_module_offloaded(module):
        with accelerate_writeback_context(module):
            yield

    else:
        yield

""" weight modification """

@parameterize
def test_pre_hook_weight_modify(fsdp, offload):
    model = load_model(fsdp, offload)
    module = model.layers[0].mlp.up_proj

    # register hook being tested
    @torch.no_grad()
    def pre_hook(module, _args):
        with modify_params_context(model, module):
            module.weight *= 0
    module.register_forward_pre_hook(pre_hook)

    # register hook to check result
    def check_hook(_module, _args, output):
        assert torch.all(output == 0.0)
    module.register_forward_hook(check_hook)

    input = torch.randint(0, 32, size=(1, 32))
    #from torch.distributed._composable_state import _get_module_state
    model(input)

@parameterize
def test_post_hook_weight_modify(fsdp, offload):
    model = load_model(fsdp, offload)
    module = model.layers[0].mlp.up_proj

    # register hook being tested
    @torch.no_grad()
    def post_hook(module, _args, _output):
        with modify_params_context(model, module):
            module.weight *= 0
    hook = module.register_forward_hook(post_hook)

    # trigger hook
    input = torch.randint(0, 32, size=(1, 32))
    model(input)
    hook.remove()

    # register hook to check result
    def check_hook(_module, _args, output):
        assert torch.all(output == 0.0)
    module.register_forward_hook(check_hook)

    # trigger checking hook
    input = torch.randint(0, 32, size=(1, 32))
    model(input)

""" weight assignment """

@parameterize
def test_pre_hook_weight_assign(fsdp, offload):
    model = load_model(fsdp, offload)
    module = model.layers[0].mlp.up_proj

    # register hook being tested
    @torch.no_grad()
    def pre_hook(module, args):
        with modify_params_context(model, module):
            module.weight.data = torch.zeros_like(module.weight.data, device=module.weight.device)
    module.register_forward_pre_hook(pre_hook)

    # register hook to check result
    def check_hook(_module, _args, output):
        assert torch.all(output == 0.0)
    module.register_forward_hook(check_hook)

    input = torch.randint(0, 32, size=(1, 32))
    model(input)
    
@parameterize
def test_post_hook_weight_assign(fsdp, offload):
    model = load_model(fsdp, offload)
    module = model.layers[0].mlp.up_proj

    # register hook being tested
    @torch.no_grad()
    def post_hook(module, _args, _output):
        with modify_params_context(model, module):
            module.weight.data = torch.zeros_like(module.weight.data, device=module.weight.device)
    hook = module.register_forward_hook(post_hook)

    # trigger hook
    input = torch.randint(0, 32, size=(1, 32))
    model(input)
    hook.remove()

    # register hook to check result
    def check_hook(_module, _args, output):
        assert torch.all(output == 0.0)
    module.register_forward_hook(check_hook)

    # trigger checking hook
    input = torch.randint(0, 32, size=(1, 32))
    model(input)

""" input return """

@parameterize
def test_pre_hook_input_return(fsdp, offload):
    model = load_model(fsdp, offload)
    module = model.layers[0].mlp.up_proj

    # register hook being tested
    @torch.no_grad()
    def pre_hook(_module, args):
        return (args[0] * 0, )
    module.register_forward_pre_hook(pre_hook)

    # register hook to check result
    def check_hook(_module, args, output):
        assert torch.all(args[0] == 0.0)
        assert torch.all(output == 0.0)
    module.register_forward_hook(check_hook)

    input = torch.randint(0, 32, size=(1, 32))
    model(input)

""" output return """

@parameterize
def test_post_hook_output_return(fsdp, offload):
    model = load_model(fsdp, offload)
    module = model.layers[0].mlp.up_proj

    # register hook being tested
    @torch.no_grad()
    def post_hook(module, _args, output):
        return output * 0.0
    module.register_forward_hook(post_hook)

    # register hook to check result
    def check_hook(_module, _args, output):
        assert torch.all(output == 0.0)
    module.register_forward_hook(check_hook)

    # trigger checking hook
    input = torch.randint(0, 32, size=(1, 32))
    model(input)