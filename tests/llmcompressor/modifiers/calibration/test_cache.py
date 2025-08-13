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

import torch
from compressed_tensors.quantization.quant_args import QuantizationArgs

from llmcompressor.modifiers.quantization.cache import QuantizedKVParameterCache
from llmcompressor.observers import Observer


def test_is_quantized_cache_singleton():
    """
    Check if quantized_cache is a singleton, used for
    passing in QuantizedKVParameterCache to the forward call of
    the model's self_attn
    """

    args = QuantizationArgs()
    cache = QuantizedKVParameterCache(args)
    observer = args.observer
    observer = Observer.load_from_registry(observer, quantization_args=args)

    tensor = torch.tensor([1, 2, 3])
    cache.k_scales.append(tensor)
    cache.k_observers.append(observer)

    same_cache = QuantizedKVParameterCache(args)

    assert len(cache.k_scales) == len(same_cache.k_scales)
    assert torch.equal(cache.k_scales[0], same_cache.k_scales[0])

    assert cache.k_observers == same_cache.k_observers
    assert hex(id(cache.k_observers[0])) == hex(id(same_cache.k_observers[0]))

    cache.reset()


def test_update():
    num_bits = 8
    args = QuantizationArgs(num_bits=num_bits, symmetric=True)
    cache = QuantizedKVParameterCache(args)

    max_key_states_val = 1.0
    max_value_states_val = 2.0
    key_states = torch.cat(
        (max_key_states_val * torch.ones(1, 2, 2), torch.ones(1, 2, 2)), dim=0
    )
    value_states = torch.cat(
        (max_value_states_val * torch.ones(1, 2, 2), torch.ones(1, 2, 2)), dim=0
    )
    layer_idx = 0

    cache.update(key_states, value_states, layer_idx)
    denom = (2 ** (num_bits) - 1) / 2
    expected_k_scale = torch.tensor([max_key_states_val / denom])
    expected_v_scale = torch.tensor([max_value_states_val / denom])

    assert cache.k_scales[0] == expected_k_scale
    assert cache.v_scales[0] == expected_v_scale

    # new attn layer
    layer_idx = 1
    cache.update(key_states, value_states, layer_idx)

    assert len(cache.k_scales) == 2
    assert len(cache.v_scales) == 2

    assert len(cache.k_observers) == 2
    assert len(cache.v_observers) == 2

    cache.reset()


def test_cache_reset():
    num_bits = 8
    args = QuantizationArgs(num_bits=num_bits, symmetric=True)
    cache = QuantizedKVParameterCache(args)

    max_key_states_val = 1.0
    max_value_states_val = 2.0
    key_states = torch.cat(
        (max_key_states_val * torch.ones(1, 2, 2), torch.ones(1, 2, 2)), dim=0
    )
    value_states = torch.cat(
        (max_value_states_val * torch.ones(1, 2, 2), torch.ones(1, 2, 2)), dim=0
    )
    layer_idx = 0

    cache.update(key_states, value_states, layer_idx)
    assert len(cache.k_scales) == 1
    assert len(cache.v_scales) == 1

    assert len(cache.k_observers) == 1
    assert len(cache.v_observers) == 1

    cache.reset()

    # new instance, different memory addr
    different_cache = QuantizedKVParameterCache(args)

    assert len(different_cache.k_scales) == 0
    assert len(different_cache.v_scales) == 0

    assert len(different_cache.k_observers) == 0
    assert len(different_cache.v_observers) == 0

    assert hex(id(cache)) != hex(id(different_cache))
