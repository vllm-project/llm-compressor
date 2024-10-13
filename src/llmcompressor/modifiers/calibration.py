import torch
from compressed_tensors.quantization import QuantizationArgs
from torch.nn import Module

__all__ = ["initialize_observer" "update_weight_zp_scale"]


def initialize_observer(
    module: Module,
    base_name: str,
):
    # initialize observer module and attach as submodule
    arg_name = "weights" if base_name == "weight" else base_name
    quantization_args = getattr(module.quantization_scheme, arg_name)
    observer = quantization_args.get_observer()
    observer = Observer.load_from_registry(
        observer, quantization_args=quantization_args
    )
    module.register_module(f"{base_name}_observer", observer)


def call_observer(self, module: Module, base_name: str, value: torch.Tensor):
    observer = getattr(module, f"{base_name}_observer")
    g_idx = getattr(module, "weight_g_idx", None)

    updated_scale, updated_zero_point = observer(value, g_idx=g_idx)

    # update scale and zero point
    update_parameter_data(module, updated_scale, f"{base_name}_scale")
    update_parameter_data(module, updated_zero_point, f"{base_name}_zero_point")


def update_weight_zp_scale(moudle: Module):
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

    if quantize_weights_upfront and module.quantization_scheme.weights is not None:
        # set weight scale and zero_point up front, calibration data doesn't affect it
        offloaded = is_module_offloaded(module)
        if offloaded:
            module._hf_hook.pre_forward(module)

        call_observer(module=module, base_name="weight", value=module.weight)

        if offloaded:
            module._hf_hook.post_forward(module, None)

    module.quantization_status = QuantizationStatus.CALIBRATION


def calibrate_activations(
    module: Module,
    value: torch.Tensor,
    base_name: str,
    quantization_args: QuantizationArgs,
):
    # If empty tensor, can't update zp/scale
    # Case for MoEs
    if value.numel() == 0:
        return
    # calibration mode - get new quant params from observer
    if not hasattr(module, f"{base_name}_observer"):
        from compressed_tensors.quantization.lifecycle import initialize_observers

        initialize_observers(
            module=module, base_name=base_name, quantization_args=quantization_args
        )

    call_observer(
        module=module,
        base_name=base_name,
        value=value,
        quantization_args=quantization_args,
    )


def _calibrate_input_hook():
    def hook_fn(module, inp):
        if module.input_activations:
            calibrate_activations(
                module,
                value=inp,
                base_name="input",
                quantization_args=module.input_activations,
            )
    return hook_fn


def _calibrate_output_hook():
    def hook_fn(module, inp, output):
        if module.output_activations:
            calibrate_activations(
                module,
                value=output,
                base_name="output",
                quantization_args=module.output_activations,
            )
            output = forward_quantize(module, output)
            return output
    return hook_fn
