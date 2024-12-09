from dataclasses import dataclass
from typing import Dict

import torch
from torch.nn import Parameter

from llmcompressor.core import ModelParameterizedLayer
from llmcompressor.modifiers.utils.hooks import HooksMixin

__all__ = ["LayerParamMasking", "param_mask_name"]


def param_mask_name() -> str:
    """
    Name to use for mask buffer on a sparse layer
    """
    return "mask"


def setup_mask_for_param(param: Parameter, mask: torch.Tensor) -> torch.Tensor:
    if mask is None:
        raise ValueError("Mask cannot be None")

    if mask.shape != param.data.shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match " f"param shape {param.data.shape}"
        )

    if mask.dtype != torch.bool:
        raise ValueError("Mask must be a boolean tensor")

    return param.data.new_tensor(mask, dtype=torch.bool)


@dataclass
class ParameterizedLayerMaskSettings:
    persistent: bool = False
    use_hooks: bool = False


class LayerParamMasking(HooksMixin):
    _mask_settings: Dict[str, ParameterizedLayerMaskSettings] = {}
    _masked_layer_params: Dict[str, ModelParameterizedLayer] = {}
    enabled_: bool = False

    def add_mask(
        self,
        layer_param_name: str,
        parameterized_layer: ModelParameterizedLayer,
        init_mask: torch.Tensor = None,
        persistent: bool = False,
        add_hooks: bool = False,
    ):
        if layer_param_name in self._masked_layer_params:
            raise ValueError(f"Layer param {layer_param_name} already has a mask")

        mask_name = param_mask_name()

        try:
            parameterized_layer.layer.get_buffer(mask_name)
        except AttributeError:
            # add the mask buffer to the layer
            parameterized_layer.layer.register_buffer(
                mask_name,
                torch.ones_like(parameterized_layer.param.data, dtype=torch.bool),
                persistent=persistent,
            )

        if init_mask is not None:
            parameterized_layer.layer.get_buffer(mask_name).fill_(
                setup_mask_for_param(parameterized_layer.param, init_mask)
            )

        self._masked_layer_params[layer_param_name] = parameterized_layer
        self._mask_settings[layer_param_name] = ParameterizedLayerMaskSettings(
            persistent=persistent, use_hooks=add_hooks
        )

        if add_hooks:

            def _forward_hook_fn(module, input, output):
                if not self.enabled_:
                    return output

                mask = module.get_buffer(mask_name)
                parameterized_layer.param.data = parameterized_layer.param.data * mask

                return output

            def _backward_hook_fn(gradients):
                if not self.enabled_:
                    return

                mask = parameterized_layer.layer.get_buffer(mask_name)
                if gradients[0] is not None:
                    gradients[0] *= mask

                return gradients

            self.register_hook(parameterized_layer.layer, _forward_hook_fn, "forward")
            self.register_hook(parameterized_layer.param, _backward_hook_fn, "")

    def update_mask(
        self,
        layer_param_name: str,
        mask: torch.Tensor,
    ):
        parameterized_layer = self._masked_layer_params[layer_param_name]
        mask_name = param_mask_name()
        mask_tensor = parameterized_layer.layer.get_buffer(mask_name)
        mask_tensor[:] = mask

    def remove_mask(self, layer_param_name: str):
        mask_settings = self._mask_settings[layer_param_name]
        parameterized_layer = self._masked_layer_params[layer_param_name]

        if not mask_settings.persistent:
            delattr(
                parameterized_layer.layer,
                param_mask_name(),
            )

        del self._masked_layer_params[layer_param_name]
        del self._mask_settings[layer_param_name]

        if mask_settings.use_hooks:
            self.remove_hooks()

    def apply_mask_weight(self, layer_param_name: str):
        if not self.enabled_:
            return

        parameterized_layer = self._masked_layer_params[layer_param_name]
        mask_name = param_mask_name()
        mask = parameterized_layer.layer.get_buffer(mask_name)
        parameterized_layer.param.data = parameterized_layer.param.data * mask

    def apply_mask_gradient(self, layer_param_name: str):
        if not self.enabled_:
            return

        parameterized_layer = self._masked_layer_params[layer_param_name]
        mask_name = param_mask_name()
        mask = parameterized_layer.layer.get_buffer(mask_name)

        if parameterized_layer.param.grad is not None:
            parameterized_layer.param.grad = parameterized_layer.param.grad * mask

    def enable_masks(self):
        self.enabled_ = True

    def disable_masks(self):
        self.enabled_ = False
