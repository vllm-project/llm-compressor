import torch
from compressed_tensors.quantization import QuantizationStatus, is_attention_module
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.quantization.utils import is_kv_cache_quant_scheme
from compressed_tensors.utils.offload import is_module_offloaded, update_parameter_data
from loguru import logger
from torch.nn import Module

from llmcompressor.modifiers.quantization.cache import QuantizedKVParameterCache
from llmcompressor.observers import Observer

__all__ = [
    "initialize_observer",
    "update_weight_zp_scale",
    "calibrate_input_hook",
    "calibrate_output_hook",
    "calibrate_kv_cache_input_hook",
    "calibrate_kv_cache_output_hook",
    "set_unset_kv_cache",
    "freeze_module_quantization",
    "apply_calibration_status",
]


def initialize_observer(
    module: Module,
    base_name: str,
):
    """
    Initialize observer module and attach as submodule.
    The name of the observer is fetched from the quantization_args.
    The name is then used to load the observer from the registry and attached
    to the module. The name of the observer uses the base_name provided.

    :param module: torch.nn.Module that the observer is being attached to
    :param base_name: str used to name the observer attribute

    """

    arg_name = "weights" if base_name == "weight" else f"{base_name}_activations"
    quantization_scheme = getattr(module, "quantization_scheme", None)
    if not quantization_scheme:
        # no quantization scheme nothing to do
        return

    # observers have a different lifecycle for kv_cache
    if is_attention_module(module):
        return

    quantization_args = getattr(quantization_scheme, arg_name, None)
    # dont need observers for dynamic
    if quantization_args is not None and not quantization_args.dynamic:
        observer = Observer.load_from_registry(
            quantization_args.observer, quantization_args=quantization_args
        )
        module.register_module(f"{base_name}_observer", observer)


def call_observer(module: Module, base_name: str, value: torch.Tensor):
    """
    Call a module's attached input/output observer using a provided value.
    Update the module's scale and zp using the observer's return
    values.

    :param module: torch.nn.Module
    :param base_name: substring used to fetch the observer, scales, and zp
    :param value: torch.Tensor to be passed to the observer
    """
    offloaded = is_module_offloaded(module)
    if offloaded:
        module._hf_hook.pre_forward(module)

    observer = getattr(module, f"{base_name}_observer")
    g_idx = getattr(module, "weight_g_idx", None)

    if base_name == "weight":
        updated_scale, updated_zero_point = observer(value, g_idx=g_idx)
    else:
        updated_scale, updated_zero_point = observer(value)

    # update scale and zero point
    update_parameter_data(module, updated_scale, f"{base_name}_scale")
    update_parameter_data(module, updated_zero_point, f"{base_name}_zero_point")

    if offloaded:
        module._hf_hook.post_forward(module, None)


def update_weight_zp_scale(module: Module):
    """
    marks a layer as ready for calibration which activates observers
    to update scales and zero points on each forward pass

    apply to full model with `model.apply(update_weight_zp_scale)`

    :param module: module to set for calibration
    :param quantize_weights_upfront: whether to automatically
       run weight quantization at the start of calibration
    """
    if not getattr(module, "quantization_scheme", None):
        # no quantization scheme nothing to do
        return

    status = getattr(module, "quantization_status", None)
    if not status:
        # not set to initialize; no scales/zp to update
        return
    if status != QuantizationStatus.INITIALIZED:
        logger.warning(
            f"Attempting set module with status {status} to calibration mode. "
            f"but status is not {QuantizationStatus.INITIALIZED} - you may "
            "be calibrating an uninitialized module which may fail or attempting "
            "to re-calibrate a frozen module"
        )

    if module.quantization_scheme.weights is not None:
        # set weight scale and zero_point up front, calibration data doesn't affect it
        call_observer(module=module, base_name="weight", value=module.weight)


def calibrate_activations(module: Module, value: torch.Tensor, base_name: str):
    """
    Calibrate input or output activations by calling the a module's attached
    observer.

    :param module: torch.nn.Module
    :param base_name: substring used to fetch the observer, scales, and zp
    :param value: torch.Tensor to be passed to the observer

    """
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
    """
    Hook to calibrate input activations.
    Will call the observers to update the scales/zp before applying
    input QDQ in the module's forward pass.
    """

    def hook_fn(module: Module, inp):
        inp = inp[0] if isinstance(inp, tuple) else inp
        calibrate_activations(module, value=inp, base_name="input")

    return hook_fn


def calibrate_output_hook():
    """
    Hook to calibrate output activations.
    Will call the observers to update the scales/zp before applying
    output QDQ.
    """

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
    """
    Hook to update inputs to attention layers when running
    kv_cache quantization. Will update the passed in
    kv_cache to singleton QuantizedKVParameterCache.
    """

    def hook_fn(module: Module, args, kwargs):
        kv_cache = getattr(module, "kv_cache")
        kwargs["past_key_value"] = kv_cache
        kwargs["use_cache"] = False
        return args, kwargs

    return hook_fn


def calibrate_kv_cache_output_hook():
    """
    Hook to update k_scale and v_scale parameters when running kv_cache quantization.
    """

    def hook_fn(module: Module, inpt, output: torch.Tensor):
        kv_cache = getattr(module, "kv_cache")
        update_parameter_data(module, kv_cache.k_scales[module.layer_idx], "k_scale")
        update_parameter_data(module, kv_cache.v_scales[module.layer_idx], "v_scale")

    return hook_fn


def set_unset_kv_cache(module: Module):
    """
    Set or unset singleton QuantizedKVParameterCache for each
    attn module when running kv_cache quantization.
    """
    if not hasattr(module, "quantization_scheme"):
        return

    if is_kv_cache_quant_scheme(module.quantization_scheme):
        output_args = module.quantization_scheme.output_activations
        kv_cache = QuantizedKVParameterCache(output_args)
        if hasattr(module, "kv_cache"):
            delattr(module, "kv_cache")
        else:
            setattr(module, "kv_cache", kv_cache)


def apply_calibration_status(module: Module):
    scheme = getattr(module, "quantization_scheme", None)
    if not scheme:
        # no quantization scheme nothing to do
        return
    module.quantization_status = QuantizationStatus.CALIBRATION


def freeze_module_quantization(module: Module):
    """
    deletes observers when calibration is complete.

    apply to full model with `model.apply(freeze_module_quantization)`

    :param module: module to freeze quantization for
    """
    scheme = getattr(module, "quantization_scheme", None)
    if not scheme:
        # no quantization scheme nothing to do
        return

    if module.quantization_status == QuantizationStatus.FROZEN:
        # nothing to do, already frozen
        return

    for name in ("input", "weight", "output"):
        obs_name = f"{name}_observer"
        if hasattr(module, obs_name):
            delattr(module, obs_name)

    module.quantization_status = QuantizationStatus.FROZEN
