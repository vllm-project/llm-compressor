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

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple
from weakref import ReferenceType, ref

from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.utils import getattr_chain
from compressed_tensors.utils.internal import InternalModule
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle
from transformers import Cache, PretrainedConfig, PreTrainedModel


__all__ = [
    "QuantizedKVCache",
    "initialize_hooked_kv_cache",
    "register_key_hook",
    "register_value_hook",
    "KV_CACHE_ATTR",
]


KV_CACHE_ATTR = "kv_cache"


class QuantizedKVCache(InternalModule):
    """
    QuantizedKVCache module which wraps the functionality of any existing kvcache args.
    Unlike transform Cache instances, this cache is a `torch.nn.Module` which can be
    hooked to trigger transforms and calibration hooks.

    This module works by being registered as a submodule to attention modules via
    `initialize_hooked_kv_cache`, then adding a hook which replaces `past_key_values`
    kwargs with this module. This module adopts the functionality of the replaced cache,
    preserving caching functionality such as sliding window attention, ect.

    :param attn_module: parent attention module
    """

    def __init__(self, config: PretrainedConfig, attn_module: Module):
        super().__init__()
        self.config = config
        self.attn_module = ref(attn_module)  # avoid circular reference
        self.past_key_values: Optional[ReferenceType[Cache]] = None

    def update(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        return self(*args, **kwargs)

    def forward(
        self,
        key_states: Tensor,
        value_states: Tensor,
        *args,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        # quantization
        module = self.attn_module()
        quant_args_attr = "quantization_scheme.input_activations"
        quant_args = getattr_chain(module, quant_args_attr, None)
        quant_enabled = getattr(module, "quantization_enabled", True)
        if quant_args is not None and quant_enabled:
            key_states = forward_quantize(module, key_states, "k", quant_args)
            value_states = forward_quantize(module, value_states, "v", quant_args)

        # original cache
        if self.past_key_values is not None:
            ret = self.past_key_values().update(
                key_states, value_states, *args, **kwargs
            )
        else:
            ret = (key_states, value_states)
        self.past_key_values = None

        return ret

    def add_past_key_values(self, past_key_values: Optional[Cache]):
        if past_key_values is not None:
            self.past_key_values = ref(past_key_values)
        else:
            self.past_key_values = None


# ----- initialize ----- #


def _kv_cache_attention_hook(
    module: Module, args: List[Any], kwargs: Dict[str, Any]
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Hook which should be called before each quantized attention forward pass.
    This hook dynamically replaces the `past_key_values` kwarg to the attention
    forward function.

    The original kvcache object is assigned to QuantizedKVCache().past_key_values
    as a weakref to maintain original cache functionality and compute savings
    """
    _past_kv_name = (
        "past_key_values"  # transformers#39956
        if "past_key_values" in inspect.signature(module.forward).parameters
        else "past_key_value"
    )
    past_key_values: Optional[Cache] = kwargs.get(_past_kv_name, None)

    cache: QuantizedKVCache = getattr(module, KV_CACHE_ATTR)
    cache.add_past_key_values(past_key_values)
    kwargs[_past_kv_name] = cache

    return args, kwargs


def initialize_hooked_kv_cache(model: PreTrainedModel, module: Module):
    """
    Initialize a `QuantizedKVCache` instance attached to attention

    :param model: parent model of attention module
    :param module: attention module to initialize with
    """
    if not hasattr(module, KV_CACHE_ATTR):
        module.register_module(KV_CACHE_ATTR, QuantizedKVCache(model.config, module))
        module.register_forward_pre_hook(_kv_cache_attention_hook, with_kwargs=True)


# ----- hooks ----- #


def register_key_hook(
    module: Module, hook: Callable[[Module, Tensor], Optional[Tensor]]
) -> RemovableHandle:
    """
    Register a hook which takes post-rope key states as an argument and
    returns the modified key states or `None`

    :param module: attention module to add hook to
    :param hook: key hook function
    """
    kv_cache: QuantizedKVCache = getattr(module, KV_CACHE_ATTR)

    def _hook(cache: QuantizedKVCache, args, kwargs):
        bound = inspect.signature(cache.forward).bind(*args, **kwargs)
        value = hook(module, bound.arguments["key_states"])
        if value is not None:
            bound.arguments["key_states"] = value

        return bound.args, bound.kwargs

    return kv_cache.register_forward_pre_hook(_hook, with_kwargs=True)


def register_value_hook(
    module: Module, hook: Callable[[Module, Tensor], Optional[Tensor]]
) -> RemovableHandle:
    """
    Register a hook which takes value states as an argument and
    returns the modified value states or `None`

    :param module: attention module to add hook to
    :param hook: value hook function
    """
    kv_cache: QuantizedKVCache = getattr(module, KV_CACHE_ATTR)

    def _hook(cache: QuantizedKVCache, args, kwargs):
        bound = inspect.signature(cache.forward).bind(*args, **kwargs)
        value = hook(module, bound.arguments["value_states"])
        if value is not None:
            bound.arguments["value_states"] = value

        return bound.args, bound.kwargs

    return kv_cache.register_forward_pre_hook(_hook, with_kwargs=True)
