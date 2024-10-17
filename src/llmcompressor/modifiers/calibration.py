import logging

import torch
from compressed_tensors.quantization import QuantizationStatus, is_attention_module
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.utils.offload import is_module_offloaded, update_parameter_data
from torch.nn import Module

from llmcompressor.observers import Observer

__all__ = [
    "initialize_observer",
    "update_weight_zp_scale",
    "calibrate_input_hook",
    "calibrate_output_hook",
    "calibrate_kv_cache_input_hook",
    "calibrate_kv_cache_output_hook"
]

_LOGGER = logging.getLogger(__name__)


def initialize_observer(
    module: Module,
    base_name: str,
):
    # initialize observer module and attach as submodule
    arg_name = "weights" if base_name == "weight" else f"{base_name}_activations"
    quantization_scheme = getattr(module, "quantization_scheme", None)
    if not quantization_scheme:
        # no quantization scheme nothing to do
        return

    # observers have a different lifecycle for kv_cache
    if is_attention_module(module):
        return 

    quantization_args = getattr(quantization_scheme, arg_name, None)
    if quantization_args:
        observer = quantization_args.get_observer()
        observer = Observer.load_from_registry(
            observer, quantization_args=quantization_args
        )
        module.register_module(f"{base_name}_observer", observer)


def call_observer(module: Module, base_name: str, value: torch.Tensor):
    observer = getattr(module, f"{base_name}_observer")
    # TODO: what cases require the g_idx?
    g_idx = getattr(module, "weight_g_idx", None)

    updated_scale, updated_zero_point = observer(value, g_idx=g_idx)

    # update scale and zero point
    update_parameter_data(module, updated_scale, f"{base_name}_scale")
    update_parameter_data(module, updated_zero_point, f"{base_name}_zero_point")


def update_weight_zp_scale(module: Module):
    """
    marks a layer as ready for calibration which activates observers
    to update scales and zero points on each forward pass

    apply to full model with `model.apply(set_module_for_calibration)`

    :param module: module to set for calibration
    :param quantize_weights_upfront: whether to automatically
       run weight quantization at the start of calibration
    """
    if not getattr(module, "quantization_scheme", None):
        # no quantization scheme nothing to do
        return

    status = getattr(module, "quantization_status", None)
    if not status or status != QuantizationStatus.INITIALIZED:
        _LOGGER.warning(
            f"Attempting set module with status {status} to calibration mode. "
            f"but status is not {QuantizationStatus.INITIALIZED} - you may "
            "be calibrating an uninitialized module which may fail or attempting "
            "to re-calibrate a frozen module"
        )

    if module.quantization_scheme.weights is not None:
        # set weight scale and zero_point up front, calibration data doesn't affect it
        offloaded = is_module_offloaded(module)
        if offloaded:
            module._hf_hook.pre_forward(module)

        call_observer(module=module, base_name="weight", value=module.weight)

        if offloaded:
            module._hf_hook.post_forward(module, None)

    module.quantization_status = QuantizationStatus.CALIBRATION


def calibrate_activations(module: Module, value: torch.Tensor, base_name: str):
    # If empty tensor, can't update zp/scale
    # Case for MoEs
    if value.numel() == 0:
        return

    call_observer(
        module=module,
        base_name=base_name,
        value=value,
    )


def calibrate_input_hook():
    def hook_fn(module: Module, inp):
        # Why does the hook wrap the input as a tuple?
        inp = inp[0] if isinstance(inp, tuple) else inp
        calibrate_activations(module, value=inp, base_name="input")

    return hook_fn


def calibrate_output_hook():
    def hook_fn(module: Module, inp, output: torch.Tensor):
        calibrate_activations(
            module,
            value=output,
            base_name="output",
        )
        output = forward_quantize(
            module=module,
            value=output,
            base_name="output",
            args=module.quantization_scheme.output_activations,
        )
        return output

    return hook_fn

def calibrate_kv_cache_input_hook():
    def hook_fn(module: Module, inp):
        kv_cache = module.getattr(module, "kv_cache")
        # update inputs/args/kwargs
        print("inp", inp)
        breakpoint()
        return inp
    
    return hook_fn


def calibrate_kv_cache_output_hook():
    def hook_fn(module: Module, inp, output: torch.Tensor):
        kv_cache = module.getattr(module, "kv_cache")
        update_parameter_data(
            module, kv_cache.k_scales[module.layer_idx], "k_scale"
        )
        update_parameter_data(
            module, kv_cache.v_scales[module.layer_idx], "v_scale"
        )
    return hook_fn
