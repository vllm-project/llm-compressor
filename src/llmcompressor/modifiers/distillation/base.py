import inspect

import torch
import torch.nn as nn
from compressed_tensors.quantization import enable_quantization
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.quantization.utils import is_module_quantized
from compressed_tensors.utils import (
    align_module_device,
    update_offload_parameter,
)
from loguru import logger
from pydantic import PrivateAttr

from llmcompressor.core import Event, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.calibration import (
    observe,
    update_qparams,
)
from llmcompressor.modifiers.quantization.quantization import QuantizationMixin
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.utils.pytorch import infer_sequential_targets

__all__ = ["QuantizationAwareDistillationModifier"]


class QuantizationAwareDistillationModifier(Modifier, QuantizationMixin):
    """
    Block-wise quantization-aware distillation. For each subgraph block in the
    sequential pipeline, this modifier:

    1. Captures inputs and outputs to decoder layers during calibration
    2. Uses cached outputs as teacher targets (pre-quantization)
    3. Runs a distillation loop using the standard CT quantized forward
       (implicit STE via weight data patching), minimizing MSE against teacher
    4. Re-observes weights after each epoch to keep scales/zero-points aligned
    5. Applies final fake quantization to weight data for error propagation

    The distillation loss is computed on the full block output, not per-layer,
    so quantization error is minimized across the entire subgraph.

    Sample usage::

        from llmcompressor import oneshot
        from llmcompressor.modifiers.distillation import (
            QuantizationAwareDistillationModifier,
        )

        recipe = QuantizationAwareDistillationModifier(
            targets="Linear", scheme="W4A16", ignore=["lm_head"],
            epochs=10, lr=1e-5,
        )
        oneshot(model=model, dataset=ds, recipe=recipe, ...)

    :param epochs: number of distillation epochs per block. Defaults to 10.
    :param lr: learning rate for the Adam optimizer. Defaults to 1e-5.
    :param master_precision: dtype for master weights used by the optimizer.
        Set to "float32" (default) for stable optimization with small learning
        rates. Set to None to optimize in the model's native dtype.
    :param offload_device: device to offload cached inputs/outputs to between
        uses. Defaults to "cpu" to save GPU memory. Set to None to keep
        cached tensors on the original device.
    """

    requires_calibration_data: bool = True

    epochs: int = 10
    lr: float = 1e-5
    master_precision: str | None = "float32"
    offload_device: str | None = "cpu"

    # module instance -> IntermediatesCache of bound forward kwargs
    _cached_inputs: dict[nn.Module, IntermediatesCache] = PrivateAttr(
        default_factory=dict
    )
    # module instance -> IntermediatesCache of output hidden states
    _cached_outputs: dict[nn.Module, IntermediatesCache] = PrivateAttr(
        default_factory=dict
    )
    _sequential_targets: list[str] = PrivateAttr(default_factory=list)
    _module_names: dict[nn.Module, str] = PrivateAttr(default_factory=dict)

    def on_initialize(self, state: State, **kwargs) -> bool:
        if QuantizationMixin.has_config(self):
            QuantizationMixin.initialize_quantization(self, state.model)

        self._module_names = {
            mod: name for name, mod in state.model.named_modules()
        }
        state.model.requires_grad_(False)
        self._sequential_targets = infer_sequential_targets(
            state.model, sequential_targets=kwargs.get("sequential_targets")
        )
        return True

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        if QuantizationMixin.has_config(self):
            QuantizationMixin.start_calibration(self, state.model)
        for _, module in state.model.named_modules():
            if self._is_decoding_layer(module):
                self.register_hook(
                    module, self._capture_input, "forward_pre", with_kwargs=True
                )
                self.register_hook(
                    module, self._capture_output, "forward"
                )

    def _capture_input(self, module, args, kwargs):
        if module not in self._cached_inputs:
            self._cached_inputs[module] = IntermediatesCache(
                offload_device=self.offload_device
            )
        bound = inspect.signature(module.forward).bind(*args, **kwargs)
        bound.apply_defaults()
        self._cached_inputs[module].append(bound.arguments)

    def _capture_output(self, module, args, output):
        if module not in self._cached_outputs:
            self._cached_outputs[module] = IntermediatesCache(
                offload_device=self.offload_device
            )
        hidden = output[0] if isinstance(output, tuple) else output
        self._cached_outputs[module].append({"hidden": hidden.detach()})

    def on_sequential_epoch_end(
        self, state: State, event: Event, modules: list[nn.Module], **kwargs
    ):
        decoding_layers = [m for m in modules if self._is_decoding_layer(m)]
        # drop cached inputs for non-first layers (only first layer's are used)
        for layer in decoding_layers[1:]:
            self._cached_inputs.pop(layer, None)
            self._cached_outputs.pop(layer, None)
        self._distill_block(state, modules)
        # clean up the first layer's cache
        if decoding_layers:
            self._cached_inputs.pop(decoding_layers[0], None)
            self._cached_outputs.pop(decoding_layers[0], None)

    def _distill_block(self, state: State, modules: list[nn.Module]):
        decoding_layers = [m for m in modules if self._is_decoding_layer(m)]
        if not decoding_layers:
            return

        first_layer = decoding_layers[0]
        last_layer = decoding_layers[-1]
        if first_layer not in self._cached_inputs:
            return
        if last_layer not in self._cached_outputs:
            return
        input_cache = self._cached_inputs[first_layer]
        output_cache = self._cached_outputs[last_layer]
        num_samples = len(input_cache.batch_intermediates)
        if num_samples == 0:
            return

        layer_names = [self._module_names.get(l, "?") for l in decoding_layers]
        logger.info("Distilling block [{}]", ", ".join(layer_names))

        quantized_modules = [m for m in modules if is_module_quantized(m)]
        if not quantized_modules:
            logger.warning("No quantized modules in block")
            return

        with align_module_device(first_layer), torch.enable_grad():
            teacher_outputs = [
                batch["hidden"] for batch in output_cache.iter()
            ]

            # enable CT quantized forwards (implicit STE via weight.data patching)
            for m in quantized_modules:
                enable_quantization(m)

            observe(quantized_modules, "weight")
            update_qparams(quantized_modules, "weight")

            master_dtype = (
                getattr(torch, self.master_precision)
                if self.master_precision
                else None
            )

            # trainable weights: either fp32 master copies or native parameters
            trainable = []
            master_weights = {}
            for m in quantized_modules:
                if not hasattr(m, "weight"):
                    continue
                m.weight.requires_grad_(True)
                if master_dtype and m.weight.dtype != master_dtype:
                    w_master = m.weight.data.to(master_dtype).clone()
                    w_master.requires_grad_(True)
                    master_weights[id(m)] = w_master
                    trainable.append(w_master)
                else:
                    trainable.append(m.weight)

            optimizer = torch.optim.Adam(trainable, lr=self.lr)

            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for batch_idx, batch_kwargs in enumerate(input_cache.iter()):
                    optimizer.zero_grad()

                    # sync master weights to module before forward
                    for m in quantized_modules:
                        if id(m) in master_weights:
                            m.weight.data.copy_(
                                master_weights[id(m)].detach().to(m.weight.dtype)
                            )

                    with HooksMixin.disable_hooks():
                        student_hidden = self._chain_layers(
                            decoding_layers, batch_kwargs
                        )

                    teacher_out = teacher_outputs[batch_idx]
                    loss = torch.nn.functional.mse_loss(
                        student_hidden.float(), teacher_out.float()
                    )
                    loss.backward()

                    # copy gradients to master weights
                    for m in quantized_modules:
                        if id(m) in master_weights and m.weight.grad is not None:
                            master_weights[id(m)].grad = m.weight.grad.to(
                                master_dtype
                            )

                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / max(num_samples, 1)
                logger.info(
                    "  epoch {}/{}: loss = {:.6f}", epoch + 1, self.epochs, avg_loss
                )

                # sync master weights back and re-observe
                for m in quantized_modules:
                    if id(m) in master_weights:
                        m.weight.data.copy_(
                            master_weights[id(m)].detach().to(m.weight.dtype)
                        )
                observe(quantized_modules, "weight")
                update_qparams(quantized_modules, "weight")

            # final sync
            for m in quantized_modules:
                if id(m) in master_weights:
                    m.weight.data.copy_(
                        master_weights[id(m)].detach().to(m.weight.dtype)
                    )
                m.weight.requires_grad_(False)

            # apply fake quantization for error propagation
            observe(quantized_modules, "weight")
            update_qparams(quantized_modules, "weight")
            for m in quantized_modules:
                scheme = getattr(m, "quantization_scheme", None)
                if scheme and scheme.weights and hasattr(m, "weight"):
                    fake_w = forward_quantize(
                        m, m.weight.data, "weight", scheme.weights
                    )
                    update_offload_parameter(m, "weight", fake_w)

    @staticmethod
    def _chain_layers(
        layers: list[nn.Module], batch_kwargs: dict
    ) -> torch.Tensor:
        """Run input through all decoder layers sequentially, return final hidden."""
        out = layers[0](**batch_kwargs)
        hidden = out[0] if isinstance(out, tuple) else out
        if len(layers) > 1:
            # subsequent layers get the same kwargs with updated hidden states
            rest_kwargs = {
                k: v for k, v in batch_kwargs.items()
                if k != next(iter(batch_kwargs))
            }
            for layer in layers[1:]:
                out = layer(hidden, **rest_kwargs)
                hidden = out[0] if isinstance(out, tuple) else out
        return hidden

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        if QuantizationMixin.has_config(self):
            QuantizationMixin.end_calibration(self, state.model)
        self.remove_hooks()
        self._cached_inputs.clear()
        self._cached_outputs.clear()
        self._module_names.clear()

    def _is_decoding_layer(self, module: nn.Module) -> bool:
        return module.__class__.__name__ in self._sequential_targets
