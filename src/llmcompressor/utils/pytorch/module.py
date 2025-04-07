"""
Utility / helper functions
"""

import difflib
import re
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import tqdm
from accelerate.hooks import attach_align_device_hook
from compressed_tensors.quantization.utils import is_module_quantized
from compressed_tensors.utils import (
    align_module_device,
    delete_offload_parameter,
    has_offloaded_params,
    register_offload_parameter,
)
from packaging import version
from torch.nn import Linear, Module, Parameter
from torch.nn.modules.conv import _ConvNd
from transformers import PreTrainedModel
from transformers.models.llama4.configuration_llama4 import Llama4Config
from transformers.models.llama4.modeling_llama4 import Llama4TextExperts
from typing_extensions import Self

from llmcompressor.core import ModelParameterizedLayer
from llmcompressor.utils.dev import skip_weights_initialize
from llmcompressor.utils.fsdp.context import (
    fix_fsdp_module_name,
    summon_full_params_context,
)

try:
    quant_err = None
    from torch.nn.qat import Conv2d as QATConv2d
    from torch.nn.qat import Linear as QATLinear
    from torch.quantization import QuantWrapper
except Exception as _err:
    quant_err = _err
    QuantWrapper = None
    QATLinear = None
    QATConv2d = None

try:
    from torch.nn.qat import Conv3d as QATConv3d
except Exception as _err:
    quant_conv3d_err = _err
    QATConv3d = None


try:
    from transformers.modeling_utils import Conv1D as TransformerConv1D
except Exception as _err:
    gpt_conv1d_err = _err
    TransformerConv1D = None


__all__ = [
    "match_targets",
    "get_default_params",
    "match_layers_params",
    "get_layers",
    "get_layer",
    "set_layer",
    "get_params",
    "get_param",
    "set_param",
    "get_terminal_layers",
    "get_prunable_layers",
    "get_quantizable_layers",
    "qat_active",
    "get_layers_params",
    "get_matching_layer",
    "get_no_split_params",
    "module_bfs",
    "replace_llama_moe",
]


_PARSED_TORCH_VERSION = version.parse(torch.__version__)


ALL_TARGET = "__ALL__"
ALL_PRUNABLE_TARGET = "__ALL_PRUNABLE__"
ALL_QUANTIZABLE_TARGET = "__ALL_QUANTIZABLE__"


def match_targets(name: str, targets: Union[str, List[str]]) -> Tuple[bool, int]:
    if isinstance(targets, str):
        targets = [targets]

    for index, target in enumerate(targets):
        if target[:3] == "re:":
            pattern = target[3:]
            if re.match(pattern, name):
                return True, index
        elif name == target:
            return True, index

    return False, -1


def match_class(layer: Module, targets: Union[str, List[str]]) -> Tuple[bool, int]:
    if isinstance(targets, str):
        targets = [targets]

    for index, target in enumerate(targets):
        if layer.__class__.__name__ == target:
            return True, index

    return False, -1


def get_default_params(layers: Dict[str, Module]) -> Dict[str, Parameter]:
    params = {}
    for name, layer in layers.items():
        for param_name, param in layer.named_parameters():
            if param_name == "weight":
                params[name] = param
                break
    return params


def match_layers_params(
    targets: Union[str, List[str]], module: Module, params: bool = False
) -> Dict[str, Union[Module, Parameter]]:
    if targets == ALL_TARGET:
        values = get_terminal_layers(module)

        return values if not params else get_default_params(values)

    if targets == ALL_PRUNABLE_TARGET:
        values = get_prunable_layers(module)

        return values if not params else get_default_params(values)

    if targets == ALL_QUANTIZABLE_TARGET:
        values = get_quantizable_layers(module)

        return values if not params else get_default_params(values)

    if isinstance(targets, str):
        targets = [targets]

    resolved = {}
    targets_found = [False for _ in range(len(targets))]

    for name, layer in module.named_modules():
        # due to nesting, FSDP may not be the top layer
        name = fix_fsdp_module_name(name)
        match, match_index = match_targets(name, targets)
        if match and not params:
            targets_found[match_index] = True
            resolved[name] = layer
        else:
            match, match_index = match_class(layer, targets)
            if match:
                targets_found[match_index] = True
                resolved[name] = layer

        for param_name, param in layer.named_parameters():
            if "." in param_name:  # skip parameters of nested layers
                continue

            param_match, param_match_index = match_targets(
                f"{name}.{param_name}", targets
            )
            if param_match:
                targets_found[param_match_index] = True
                resolved[f"{name}"] = layer if not params else param

    missed = [target for found, target in zip(targets_found, targets) if not found]
    if len(missed) > 0:
        raise ValueError(f"Could not find targets {missed} in module {module}")

    return resolved


def get_layers(targets: Union[str, List[str]], module: Module) -> Dict[str, Module]:
    return match_layers_params(targets, module)


def get_layer(target: str, module: Module) -> Tuple[str, Module]:
    layers = get_layers(target, module)
    if len(layers) != 1:
        raise ValueError(f"Expected 1 layer for target {target}, found {len(layers)}")
    name, layer = next(iter(layers.items()))

    return name, layer


def set_layer(target: str, layer: Module, module: Module) -> Module:
    with summon_full_params_context(module):
        # importing here to avoid circular import
        from llmcompressor.utils.fsdp.helpers import maybe_get_wrapped

        parent_target = ".".join(target.split(".")[:-1])
        if parent_target != "":
            parent_layer = get_layer(parent_target, module)[1]
        else:
            parent_layer = maybe_get_wrapped(module)
        old_layer = getattr(parent_layer, target.split(".")[-1])
        setattr(parent_layer, target.split(".")[-1], layer)

    return old_layer


def get_params(targets: Union[str, List[str]], module: Module) -> Dict[str, Parameter]:
    return match_layers_params(targets, module, params=True)


def get_param(target: str, module: Module) -> Tuple[str, Parameter]:
    params = get_params(target, module)
    if len(params) != 1:
        raise ValueError(
            f"Expected 1 parameter for target {target}, found {len(params)}"
        )
    name, param = next(iter(params.items()))

    return name, param


def set_param(target: str, param: Parameter, module: Module) -> Parameter:
    layer_name, param_name = target.rsplit(".", 1)
    layer = get_layer(layer_name, module)[1]
    old_param = getattr(layer, param_name)
    setattr(layer, param_name, param)

    return old_param


def get_terminal_layers(module: Module) -> Dict[str, Module]:
    terminal = {}

    for name, layer in module.named_modules():
        if len(list(layer.named_modules())) > 1:
            continue

        terminal[name] = layer

    return terminal


def get_prunable_layers(module: Module) -> Dict[str, Module]:
    prunable = {}

    for name, layer in module.named_modules():
        if (
            isinstance(layer, Linear)
            or isinstance(layer, _ConvNd)
            or (QATLinear and isinstance(layer, QATLinear))
            or (QATConv2d and isinstance(layer, QATConv2d))
            or (QATConv3d and isinstance(layer, QATConv3d))
            or (TransformerConv1D and isinstance(layer, TransformerConv1D))
        ):
            prunable[name] = layer

    return prunable


def get_quantizable_layers(module: Module) -> Dict[str, Module]:
    if QATLinear is None:
        raise ImportError(
            "PyTorch version is not setup for Quantization. "
            "Please install a QAT compatible version of PyTorch"
        )

    quantizable = {}

    for name, layer in module.named_modules():
        if isinstance(layer, Linear) or isinstance(layer, _ConvNd):
            quantizable[name] = layer

    return quantizable


def qat_active(module: Module) -> bool:
    """
    Determines if any layers in the model have quantization enabled by checking for
    weight_fake_quant attributes

    :param module: PyTorch model to check for quantization
    :return: True if quantization is active anywhere in the model, False otherwise
    """
    for _, layer in module.named_modules():
        if isinstance(layer, torch.quantization.FakeQuantize):
            return True
        if is_module_quantized(layer):
            return True

    return False


def get_layers_params(
    targets: Union[str, List[str]], module: Module
) -> Dict[str, ModelParameterizedLayer]:
    params = get_params(targets, module)
    layers = get_layers(targets, module)

    parameterized_layers = {}
    for name, param in params.items():
        param_layer = ModelParameterizedLayer(
            layer_name=name, layer=layers[name], param_name=name, param=param
        )
        parameterized_layers[name] = param_layer

    return parameterized_layers


def get_matching_layer(
    target: str, name_to_match: str, module: Module
) -> Optional[Tuple[str, Module]]:
    """
    Given a target regex, find the layer name in the module that most closely matches
    the name_to_match string. This is used to matches submodules in the same layer, for
    instance matching "re.*k_proj" to "model.decoder.layer.0.q_proj" to find the k_proj
    that exists in layer 0.

    :param target: regex to search for
    :param name_to_match: full layer name to match to, should exist in module
    :param module: module to search for target in
    :return: Tuple containing the layer name and module that fits the target regex and
    best matches name_to_match, or None if no match can be found
    """
    potential_matches = get_layers(target, module)
    largest_substring = 0
    match = None
    for name, module in potential_matches.items():
        seq_matcher = difflib.SequenceMatcher(None, name, name_to_match)
        _, _, match_length = seq_matcher.find_longest_match(
            0, len(name), 0, len(name_to_match)
        )
        if match_length > largest_substring:
            match = (name, module)
            largest_substring = match_length

    return match


def get_no_split_params(module: PreTrainedModel) -> Union[str, List[str]]:
    """
    Get list of module classes that shouldn't be split when sharding. For
    Hugging Face Transformer models, this is the decoder layer type. For other
    types of models, this just returns all module names.

    :return: list of class names that shouldn't be split
    """
    # importing here to avoid circular import
    from llmcompressor.utils.fsdp.helpers import maybe_get_wrapped

    model = maybe_get_wrapped(module)
    ret = model._get_no_split_modules("auto")
    return ret or ALL_TARGET


def module_bfs(
    module: Module,
    func: Callable[[Module], Module],
    pre: bool = True,
    progress: Union[bool, tqdm.tqdm] = False,
) -> Module:
    if progress is True:
        total = len(list(module.modules()))
        progress = tqdm.tqdm(total=total)

    if pre:
        module = func(module)

    for name, child in list(module.named_children()):
        module.add_module(name, module_bfs(child, func, pre, progress))

    if not pre:
        module = func(module)

    if isinstance(progress, tqdm.tqdm):
        progress.update(1)

    return module


class Hook:
    def __call__(self, module, state_dict, prefix, local_metadata):
        gate_up_proj = torch.empty(
            module.num_experts, module.hidden_size, module.intermediate_size * 2
        )
        down_proj = torch.empty(
            module.num_experts, module.intermediate_size, module.hidden_size
        )
        for expert_index in range(module.num_experts):
            gate_up_key = f"{prefix}experts.{expert_index}.gate_up_proj.weight"
            down_key = f"{prefix}experts.{expert_index}.down_proj.weight"

            gate_up_proj[expert_index] = state_dict[gate_up_key].T
            down_proj[expert_index] = state_dict[down_key].T

            del state_dict[gate_up_key]
            del state_dict[down_key]

        state_dict[f"{prefix}gate_up_proj"] = gate_up_proj
        state_dict[f"{prefix}down_proj"] = down_proj


class Llama4TextMLP(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, act_fn):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.act_fn = act_fn

        self.gate_up_proj = torch.nn.Linear(
            self.hidden_size, 2 * self.intermediate_size, bias=False
        )
        self.down_proj = torch.nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states)
        gate, up = (
            gate_up[:, : self.intermediate_size],
            gate_up[:, self.intermediate_size :],
        )
        return self.down_proj(up * self.act_fn(gate))


class Llama4TextExpertsLinear(torch.nn.Module):
    def __init__(
        self,
        config: Llama4Config,
        act_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size

        self.experts = torch.nn.ModuleList(
            [
                Llama4TextMLP(self.hidden_size, self.intermediate_size, act_fn)
                for _ in range(self.num_experts)
            ]
        )

        self.register_state_dict_post_hook(Hook())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        outputs = [expert(hidden_states) for expert in self.experts]
        return torch.cat(outputs)

    @classmethod
    def from_module(cls, module: Llama4TextExperts) -> Self:
        config = Llama4Config(
            num_local_experts=module.num_experts,
            intermediate_size=module.intermediate_size,
            hidden_size=module.hidden_size,
        )
        with skip_weights_initialize():
            instance = cls(config, module.act_fn)

        if has_offloaded_params(module):
            weights_map = module._hf_hook.weights_map
            for name, expert in instance.experts.named_children():
                attach_align_device_hook(
                    module=expert,
                    execution_device=module._hf_hook.execution_device,
                    offload=module._hf_hook.offload,
                    weights_map=weights_map,
                    module_name=f"experts.{name}",
                    skip_keys=module._hf_hook.skip_keys,
                )

        with align_module_device(module):
            gate_up_proj = module.gate_up_proj.data.transpose(-2, -1)
            down_proj = module.down_proj.data.transpose(-2, -1)

            for expert_index in range(module.num_experts):
                register_offload_parameter(
                    instance.experts[expert_index].gate_up_proj,
                    "weight",
                    torch.nn.Parameter(gate_up_proj[expert_index]),
                )
                register_offload_parameter(
                    instance.experts[expert_index].down_proj,
                    "weight",
                    torch.nn.Parameter(down_proj[expert_index]),
                )

        for name, _ in list(module.named_parameters()):
            delete_offload_parameter(module, name)

        return instance


def replace_llama_moe(module: torch.nn.Module) -> torch.nn.Module:
    if module.__class__.__name__ == "Llama4TextExperts":
        return Llama4TextExpertsLinear.from_module(module)
    else:
        return module
