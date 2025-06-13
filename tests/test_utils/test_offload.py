# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch
from compressed_tensors.utils import (
    align_module_device,
    align_modules,
    delete_offload_module,
    delete_offload_parameter,
    disable_hf_hook,
    disable_offloading,
    get_execution_device,
    has_offloaded_params,
    offloaded_dispatch,
    register_offload_module,
    register_offload_parameter,
    update_offload_parameter,
)
from compressed_tensors.utils.offload import offload_to_weights_map
from tests.testing_utils import requires_accelerate, requires_gpu


class ExampleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(0).float())
        self.b = torch.nn.Parameter(torch.tensor(0).float())

    def forward(self, x):
        return x * self.a + self.b


class ExampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)

    def forward(self, x):
        return self.linear(x)


@requires_accelerate()
def test_has_offloaded_params():
    from accelerate.hooks import attach_align_device_hook, remove_hook_from_module

    module = ExampleModule()
    assert not has_offloaded_params(module)

    attach_align_device_hook(module, offload=False)
    assert not has_offloaded_params(module)

    remove_hook_from_module(module)
    attach_align_device_hook(module, offload=True, weights_map=module.state_dict())
    assert has_offloaded_params(module)


@requires_gpu
@requires_accelerate()
def test_get_execution_device():
    from accelerate import init_empty_weights
    from accelerate.big_modeling import attach_align_device_hook

    # no offloading
    module = ExampleModule()
    assert get_execution_device(module) == torch.device("cpu")

    # with offloading
    attach_align_device_hook(module, torch.device("cuda:0"))
    assert get_execution_device(module) == torch.device("cuda:0")

    # in meta context
    with torch.device("meta"):
        module = ExampleModule()
        assert get_execution_device(module) == torch.device("meta")

    # offloaded in meta context
    module = ExampleModule()
    attach_align_device_hook(module, torch.device("cuda:0"))
    with torch.device("meta"):
        assert get_execution_device(module) == torch.device("cuda:0")

    # in empty weights context
    with init_empty_weights():
        module = ExampleModule()
        assert get_execution_device(module) == torch.device("meta")

    # offloaded in empty weights context
    module = ExampleModule()
    attach_align_device_hook(module, torch.device("cuda:0"))
    with init_empty_weights():
        assert get_execution_device(module) == torch.device("cuda:0")


@requires_accelerate()
def test_register_offload_parameter():
    from accelerate import init_empty_weights
    from accelerate.hooks import attach_align_device_hook

    module = ExampleModule()
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    # register a param prior to offloading
    register_offload_parameter(module, "c", parameter)
    assert module.c == parameter

    # offloading, check that added param was offloaded
    attach_align_device_hook(module, offload=True, weights_map=module.state_dict())
    assert "c" in module._hf_hook.weights_map

    # register a param after offloading, check that added param was offloaded
    register_offload_parameter(module, "d", parameter)
    assert module.d.device == torch.device("meta")
    assert module._hf_hook.weights_map["d"].device == torch.device("cpu")

    # added parameters can be onloaded and offloaded
    with align_module_device(module, execution_device="cpu"):
        assert module.c.device == torch.device("cpu")
        assert module.d.device == torch.device("cpu")
    assert module.c.device == torch.device("meta")
    assert module.d.device == torch.device("meta")

    # parameters can be added during onload
    with align_module_device(module, execution_device="cpu"):
        register_offload_parameter(module, "e", parameter)
        assert module.e.device == torch.device("cpu")

    # parameters can be added before onload and with explicit offload
    register_offload_parameter(module, "f", parameter, offload_device="cpu")
    assert module._hf_hook.weights_map["f"].device == torch.device("cpu")
    with align_module_device(module, execution_device="cpu"):
        assert module.f.device == torch.device("cpu")
    assert module._hf_hook.weights_map["f"].device == torch.device("cpu")

    # parameters registered in the empty init context are still empty
    with init_empty_weights():
        module = ExampleModule()
        register_offload_parameter(module, "c", parameter)
    assert module.a.device == module.b.device == module.c.device == torch.device("meta")


@requires_accelerate()
def test_update_offload_parameter():
    from accelerate.hooks import attach_align_device_hook

    module = ExampleModule()
    tensor_a = torch.tensor(1.0)
    tensor_b = torch.tensor(2.0)

    # can update modules which are not offloaded
    update_offload_parameter(module, "a", tensor_a)
    assert module.a == tensor_a

    # can update modules which are offloaded
    attach_align_device_hook(module, offload=True, weights_map=module.state_dict())
    update_offload_parameter(module, "b", tensor_b)
    assert module.b.device == torch.device("meta")
    assert module._hf_hook.weights_map["b"] == tensor_b

    # data persists across onloading
    with align_module_device(module, execution_device="cpu"):
        assert module.a.data == tensor_a
        assert module.b.data == tensor_b
        assert module._hf_hook.weights_map["a"] == tensor_a
        assert module._hf_hook.weights_map["b"] == tensor_b

    # data persists across offloading
    assert module.a.device == torch.device("meta")
    assert module.b.device == torch.device("meta")
    assert module._hf_hook.weights_map["a"] == tensor_a
    assert module._hf_hook.weights_map["b"] == tensor_b

    # can update with differnt shape with warning
    with pytest.warns():
        new_data = torch.tensor([3.0])
        update_offload_parameter(module, "a", new_data)
    assert module._hf_hook.weights_map["a"] == new_data


@requires_accelerate()
def test_delete_offload_parameter():
    from accelerate.hooks import attach_align_device_hook

    module = ExampleModule()
    param_c = torch.nn.Parameter(torch.tensor(1.0))
    param_d = torch.nn.Parameter(torch.tensor(2.0))
    register_offload_parameter(module, "c", param_c)
    register_offload_parameter(module, "d", param_d)

    # parameters are deleted
    delete_offload_parameter(module, "a")
    delete_offload_parameter(module, "c")
    assert not hasattr(module, "a")
    assert hasattr(module, "b")
    assert not hasattr(module, "c")
    assert hasattr(module, "d")

    # parameters and their offload are deleted
    attach_align_device_hook(module, offload=True, weights_map=module.state_dict())
    delete_offload_parameter(module, "b")
    delete_offload_parameter(module, "d")
    assert not hasattr(module, "a")
    assert not hasattr(module, "b")
    assert not hasattr(module, "c")
    assert not hasattr(module, "d")
    assert "a" not in module._hf_hook.weights_map
    assert "b" not in module._hf_hook.weights_map
    assert "c" not in module._hf_hook.weights_map
    assert "d" not in module._hf_hook.weights_map


@requires_accelerate()
def test_disable_hf_hook():
    from accelerate.hooks import attach_align_device_hook

    module = ExampleModule()

    def custom_forward():
        pass

    attach_align_device_hook(module, offload=True, weights_map=module.state_dict())
    with disable_hf_hook(module):
        assert not hasattr(module, "_hf_hook")
        module.forward = custom_forward

    assert hasattr(module, "_hf_hook")
    assert module._old_forward == custom_forward


@requires_accelerate()
def test_disable_hf_hook_model_recurse():
    from accelerate.hooks import attach_align_device_hook

    module0 = ExampleModule()
    module1 = ExampleModule()
    module2 = ExampleModule()
    model = torch.nn.Sequential(module0, torch.nn.Sequential(module1, module2))
    attach_align_device_hook(model, offload=True, weights_map=model.state_dict())

    with disable_hf_hook(model):
        assert not hasattr(module0, "_hf_hook")
        assert not hasattr(module1, "_hf_hook")
        assert not hasattr(module2, "_hf_hook")

    assert hasattr(module0, "_hf_hook")
    assert hasattr(module1, "_hf_hook")
    assert hasattr(module2, "_hf_hook")


@requires_accelerate()
def test_align_modules():
    from accelerate.hooks import attach_align_device_hook

    module0 = ExampleModule()
    module1 = ExampleModule()
    module2 = ExampleModule()
    model = torch.nn.Sequential(module0, torch.nn.Sequential(module1, module2))
    attach_align_device_hook(
        model,
        execution_device=torch.device("cpu"),
        offload=True,
        weights_map=model.state_dict(),
    )

    assert module0.a.device == torch.device("meta")
    assert module1.a.device == torch.device("meta")
    assert module2.a.device == torch.device("meta")

    with align_modules((module0, module1)):
        assert module0.a.device != torch.device("meta")
        assert module1.a.device != torch.device("meta")
        assert module2.a.device == torch.device("meta")

    assert module0.a.device == torch.device("meta")
    assert module1.a.device == torch.device("meta")
    assert module2.a.device == torch.device("meta")


@requires_accelerate()
def test_offload_to_weights_map():
    from accelerate.utils import OffloadedWeightsLoader, PrefixedDataset

    name = "name"
    old_value = torch.tensor(0.0)
    new_value = torch.tensor(1.0)
    prefix = "prefix"

    # Dict empty
    weights_map = {}
    with pytest.raises(ValueError):
        offload_to_weights_map(weights_map, name, new_value)
    offload_to_weights_map(weights_map, name, new_value, offload_device="cpu")
    assert weights_map[name] == new_value

    # Dict populated
    weights_map = {name: old_value}
    offload_to_weights_map(weights_map, name, new_value)
    assert weights_map[name] == new_value

    # OffloadedWeightsLoader[Dict] empty
    weights_map = OffloadedWeightsLoader({})
    with pytest.raises(ValueError):
        offload_to_weights_map(weights_map, name, new_value)
    offload_to_weights_map(weights_map, name, new_value, offload_device="cpu")
    assert weights_map[name] == new_value

    # OffloadedWeightsLoader[Dict] populated
    weights_map = OffloadedWeightsLoader({name: old_value})
    offload_to_weights_map(weights_map, name, new_value)
    assert weights_map[name] == new_value

    # PrefixedDataset[Dict] empty
    weights_map = PrefixedDataset({}, prefix)
    with pytest.raises(ValueError):
        offload_to_weights_map(weights_map, name, new_value)
    offload_to_weights_map(weights_map, name, new_value, offload_device="cpu")
    assert weights_map[name] == new_value

    # PrefixedDataset[Dict] populated
    weights_map = PrefixedDataset({name: old_value}, prefix)
    offload_to_weights_map(weights_map, name, new_value)
    assert weights_map[name] == new_value

    # PrefixedDataset[OffloadedWeightsLoader[Dict]] empty
    weights_map = PrefixedDataset(OffloadedWeightsLoader({}), prefix)
    with pytest.raises(ValueError):
        offload_to_weights_map(weights_map, name, new_value)
    offload_to_weights_map(weights_map, name, new_value, offload_device="cpu")
    assert weights_map[name] == new_value

    # PrefixedDataset[OffloadedWeightsLoader[Dict]] populated
    weights_map = PrefixedDataset(OffloadedWeightsLoader({name: old_value}), prefix)
    offload_to_weights_map(weights_map, name, new_value)
    assert weights_map[name] == new_value


@requires_gpu
@requires_accelerate()
@pytest.mark.parametrize("exec_device", [torch.device("cpu"), torch.device("cuda")])
def test_register_offload_module(exec_device):
    # no offloading
    model = ExampleModel()
    child = torch.nn.Linear(2, 3)
    register_offload_module(model, "child", child)
    register_offload_module(model.linear, "child", child)
    assert child in model.children()
    assert child in model.linear.children()

    # with offloading
    model = ExampleModel()
    child = torch.nn.Linear(2, 3)
    offloaded_dispatch(model, exec_device)
    register_offload_module(model, "child", child)
    register_offload_module(model.linear, "child", child)
    assert child in model.children()
    assert child in model.linear.children()

    # can run modules
    model(torch.empty(1))
    child(torch.empty(2, device=exec_device))


@requires_gpu
@requires_accelerate()
@pytest.mark.parametrize("exec_device", [torch.device("cpu"), torch.device("cuda")])
def test_delete_offload_module(exec_device):
    # no offloading
    model = ExampleModel()
    child = torch.nn.Linear(2, 3)
    register_offload_module(model, "child", child)
    register_offload_module(model.linear, "child", child)
    delete_offload_module(model, "child")
    delete_offload_module(model.linear, "child")
    assert not child in model.children()
    assert not child in model.linear.children()

    # with offloading
    model = ExampleModel()
    child = torch.nn.Linear(2, 3)
    offloaded_dispatch(model, exec_device)
    register_offload_module(model, "child", child)
    register_offload_module(model.linear, "child", child)
    delete_offload_module(model, "child")
    delete_offload_module(model.linear, "child")
    assert not child in model.children()
    assert not child in model.linear.children()


@requires_gpu
@requires_accelerate()
@pytest.mark.parametrize(
    "exec_device,offload_device",
    [
        (torch.device("cpu"), torch.device("cpu")),
        (torch.device("cpu"), torch.device("cuda:0")),
        (torch.device("cuda:0"), torch.device("cpu")),
        (torch.device("cuda:0"), torch.device("cuda:0")),
    ],
)
def test_offloaded_dispatch(exec_device, offload_device):
    # single module
    module = torch.nn.Linear(1, 2, device=offload_device)
    module = offloaded_dispatch(module, exec_device, offload_device)
    assert has_offloaded_params(module)
    assert module._hf_hook.offload
    assert module.weight.device == torch.device("meta")
    assert module._hf_hook.weights_map["weight"].device == offload_device
    assert module._hf_hook.tied_params_map is not None

    # can run
    module(torch.empty(1, device=exec_device))

    # model
    model = ExampleModel()
    model = offloaded_dispatch(model, exec_device, offload_device)
    assert not has_offloaded_params(model)

    assert has_offloaded_params(model.linear)
    assert model.linear._hf_hook.offload
    assert model.linear.weight.device == torch.device("meta")
    assert model.linear._hf_hook.weights_map["weight"].device == offload_device
    assert model.linear._hf_hook.tied_params_map is not None

    # can run
    model(torch.empty(1, device=exec_device))

    # can add new params
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    register_offload_parameter(module, "new_param", parameter)
    assert module.new_param.device == torch.device("meta")
    assert module._hf_hook.weights_map["new_param"].device == offload_device


@requires_gpu
@requires_accelerate()
@pytest.mark.parametrize(
    "exec_device,offload_device",
    [
        (torch.device("cpu"), torch.device("cpu")),
        (torch.device("cpu"), torch.device("cuda:0")),
        (torch.device("cuda:0"), torch.device("cpu")),
        (torch.device("cuda:0"), torch.device("cuda:0")),
    ],
)
def test_disable_offloading(exec_device, offload_device):
    module = torch.nn.Linear(1, 2, device=exec_device)

    # non-offloaded modules are unaffected
    with disable_offloading():
        output = module(torch.empty(1, device=exec_device))
        assert module.weight.device == exec_device
        assert output.device == exec_device

    # offloaded modules stay on device until context exit
    offloaded_dispatch(module, exec_device, offload_device)
    assert module.weight.device == torch.device("meta")
    assert module._hf_hook.weights_map["weight"].device == offload_device

    with disable_offloading():
        assert module.weight.device == torch.device("meta")
        output = module(torch.empty(1, device=exec_device))
        assert module.weight.device == exec_device
        assert output.device == exec_device

        output = module(torch.empty(1, device=exec_device))
        assert module.weight.device == exec_device
        assert output.device == exec_device

    assert module.weight.device == torch.device("meta")
    assert module._hf_hook.weights_map["weight"].device == offload_device
