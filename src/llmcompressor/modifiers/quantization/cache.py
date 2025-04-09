from typing import Any, Dict, List, Optional, Tuple

from compressed_tensors.quantization.lifecycle import KVCacheScaleType
from compressed_tensors.quantization.quant_args import QuantizationArgs
from torch import Tensor
from transformers import DynamicCache

from llmcompressor.observers import Observer


class QuantizedKVParameterCache(DynamicCache):
    """
    Quantized KV cache used in the forward call based on HF's dynamic cache.
    Quantization strategy (tensor, group, channel) set from Quantization arg's strategy
    Singleton, so that the same cache gets reused in all forward call of self_attn.
    Each time forward is called, .update() is called, and ._quantize(), ._dequantize()
     gets called appropriately.
    The size of tensor is
     `[batch_size, num_heads, seq_len - residual_length, head_dim]`.


    Triggered by adding kv_cache_scheme in the recipe.

    Example:

    ```python3
    recipe = '''
    quant_stage:
        quant_modifiers:
            QuantizationModifier:
                kv_cache_scheme:
                    num_bits: 8
                    type: float
                    strategy: tensor
                    dynamic: false
                    symmetric: true
    '''

    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """Singleton"""
        if cls._instance is None:
            cls._instance = super(QuantizedKVParameterCache, cls).__new__(cls)
        return cls._instance

    def __init__(self, quantization_args: QuantizationArgs):
        if not self._initialized:
            super().__init__()

            self.quantization_args = quantization_args

            self.k_observers: List[Observer] = []
            self.v_observers: List[Observer] = []

            # each index corresponds to layer_idx of the attention layer
            self.k_scales: List[Tensor] = []
            self.v_scales: List[Tensor] = []

            self.k_zps: List[Tensor] = []
            self.v_zps: List[Tensor] = []

            self._initialized = True

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get the k_scale and v_scale and output the
         fakequant-ed key_states and value_states
        """

        if len(self.k_observers) <= layer_idx:
            k_observer_name = self.quantization_args.observer
            k_observer = Observer.load_from_registry(
                k_observer_name, quantization_args=self.quantization_args
            )
            v_observer_name = self.quantization_args.observer
            v_observer = Observer.load_from_registry(
                v_observer_name, quantization_args=self.quantization_args
            )

            # NOTE: User may ignore some layers in configuration,
            # meaning len(self.k_observers) <= layer_idx-1
            # Must account for that case by padding list so that
            # index of lists corresponds to layer_idx
            _pad_and_append_at_idx_(self.k_observers, layer_idx, k_observer)
            _pad_and_append_at_idx_(self.v_observers, layer_idx, v_observer)

        q_key_states = self._quantize(
            key_states.contiguous(), KVCacheScaleType.KEY, layer_idx
        )
        q_value_states = self._quantize(
            value_states.contiguous(), KVCacheScaleType.VALUE, layer_idx
        )

        qdq_key_states = self._dequantize(q_key_states, KVCacheScaleType.KEY, layer_idx)
        qdq_value_states = self._dequantize(
            q_value_states, KVCacheScaleType.VALUE, layer_idx
        )

        keys_to_return, values_to_return = qdq_key_states, qdq_value_states

        return keys_to_return, values_to_return

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        Returns the sequence length of the cached states.
        A layer index can be optionally passed.
        """
        if len(self.key_cache) <= layer_idx:
            return 0
        # since we cannot get the seq_length of each layer directly and
        # rely on `_seen_tokens` which is updated every "layer_idx" == 0,
        # this is a hack to get the actual seq_length for the given layer_idx
        # this part of code otherwise fails when used to
        # verify attn_weight shape in some models
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1

    def reset_states(self):
        """reset the kv states (used in calibration)"""
        self.key_cache: List[Tensor] = []
        self.value_cache: List[Tensor] = []
        # Used in `generate` to keep tally of how many tokens the cache has seen
        self._seen_tokens = 0
        self._quantized_key_cache: List[Tensor] = []
        self._quantized_value_cache: List[Tensor] = []

    def reset(self):
        """
        Reset the instantiation, create new instance on init
        """
        QuantizedKVParameterCache._instance = None
        QuantizedKVParameterCache._initialized = False

    def _quantize(self, tensor, kv_type, layer_idx):
        """Quantizes a key/value using a defined quantization method."""
        from compressed_tensors.quantization.lifecycle.forward import quantize

        if kv_type == KVCacheScaleType.KEY:  # key type
            observer = self.k_observers[layer_idx]
            scales = self.k_scales
            zps = self.k_zps
        else:
            assert kv_type == KVCacheScaleType.VALUE
            observer = self.v_observers[layer_idx]
            scales = self.v_scales
            zps = self.v_zps

        scale, zp = observer(tensor)
        _pad_and_append_at_idx_(scales, layer_idx, scale)
        _pad_and_append_at_idx_(zps, layer_idx, zp)

        q_tensor = quantize(
            x=tensor,
            scale=scale,
            zero_point=zp,
            args=self.quantization_args,
        )
        return q_tensor

    def _dequantize(self, qtensor, kv_type, layer_idx):
        """Dequantizes back the tensor that was quantized by `self._quantize()`"""
        from compressed_tensors.quantization.lifecycle.forward import dequantize

        if kv_type == KVCacheScaleType.KEY:
            scale = self.k_scales[layer_idx]
            zp = self.k_zps[layer_idx]
        else:
            assert kv_type == KVCacheScaleType.VALUE
            scale = self.v_scales[layer_idx]
            zp = self.v_zps[layer_idx]

        qdq_tensor = dequantize(
            x_q=qtensor,
            scale=scale,
            zero_point=zp,
            args=self.quantization_args,
        )
        return qdq_tensor


# NOTE: Using _ suffix to denote l is modified in place
def _pad_and_append_at_idx_(lst: List, idx: int, val: Any) -> list:
    """
    Append value val to list lst at index idx, right padding if necessary
    Needed because user may ignore some layers in configuration, meaning
    len(lst) <= idx-1

    >>> _pad_and_append_at_idx_([0,1,2], 5, 5)
    [0, 1, 2, None, None, 5]
    >>> _pad_and_append_at_idx_([0,1,2], 3, 8)
    [0, 1, 2, 8]
    >>> _pad_and_append_at_idx_([0,1,2], 1, 5)
    [0, 5, 2]
    """
    num_to_pad = idx - len(lst) + 1
    if num_to_pad > 0:
        lst += [None] * num_to_pad
    lst[idx] = val
    return lst
