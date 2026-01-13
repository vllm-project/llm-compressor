from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate.hooks import add_hook_to_module, remove_hook_from_submodules
from auto_round import AutoRound
from auto_round.schemes import QuantizationScheme as ARQuantizationScheme
from auto_round.wrapper import WrapperWALayer
from compressed_tensors.quantization import (
    QuantizationScheme,
    QuantizationStrategy,
    enable_quantization,
)
from compressed_tensors.utils import (
    align_module_device,
    match_named_modules,
    update_offload_parameter,
)
from loguru import logger
from pydantic import PrivateAttr

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.calibration import apply_calibration_status
from llmcompressor.modifiers.quantization.quantization import QuantizationMixin
from llmcompressor.utils import targets_embeddings, untie_word_embeddings
from llmcompressor.utils.pytorch import get_no_split_params

__all__ = ["AutoRoundModifier"]


class _LLModelWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList()

    def forward(self, *args, **kwargs):
        for layer in self.layers:
            res = layer(*args, **kwargs)
        return res


class _PretrainModelWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _LLModelWrapper()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def _wrap_decoding_layer(layer: torch.nn.Module) -> _PretrainModelWrapper:
    wrapped_model = _PretrainModelWrapper()
    wrapped_model.model.layers.append(layer)
    first_param = next(layer.parameters())
    wrapped_model.dtype = first_param.dtype
    return wrapped_model


@contextmanager
def suspend_accelerate_hooks(model: nn.Module):
    """
    Temporarily suspend Accelerate hooks from a model.

    This context manager detaches all Accelerate hooks (used for device offloading,
    dtype casting, etc.) from the model, allowing Autoround to operate without
    interference. On exit, the model is restored to its original device
    and all hooks are re-attached.
    """
    saved_hooks = {}
    original_device = next(model.parameters()).device
    for name, module in model.named_modules():
        if hasattr(module, "_hf_hook"):
            saved_hooks[name] = module._hf_hook

    remove_hook_from_submodules(model)
    try:
        yield
    finally:
        remove_hook_from_submodules(model)
        model.to(original_device)
        for name, module in model.named_modules():
            if name in saved_hooks:
                logger.info("Restoring Accelerate hook for module: {}", name)
                add_hook_to_module(module, saved_hooks[name], append=True)


class AutoRoundModifier(Modifier, QuantizationMixin):
    """
    Implements the AutoRound algorithm from https://aclanthology.org/2024.findings-emnlp.662.pdf.
    This modifier leverages signed gradient descent (SignSGD) optimizer and
    block-wise loss to optimize rounding values and weight clipping in a few steps.

    Sample yaml:

    ```yaml
    test_stage:
      modifiers:
        AutoRoundModifier:
          iters: 200
          config_groups:
            group_0:
              targets:
                - "Linear"
              input_activations: null
              output_activations: null
              weights:
                num_bits: 4
                type: "int"
                symmetric: true
                strategy: group
                group_size: 128
    ```

    Lifecycle:

    - on_initialize
        - apply config to model
    - on_start
        - add input capture hooks to decoding layers
    - on_sequential_epoch_end
        - apply_autoround
        - post_autoround_cleanup
    - on_finalize
        - remove_hooks()
        - model.apply(freeze_module_quantization)

    :param config_groups: dictionary specifying quantization schemes to apply to target
        modules. Modules not matching a scheme target will NOT be quantized.
    :param targets: list of layer names to quantize if a scheme is provided. Defaults
        to Linear layers
    :param ignore: optional list of module class names or submodule names to not
        quantize even if they match a target in config_groups. Defaults to empty list.
    :param scheme: a single quantization scheme to apply to the model. This is a
        dictionary that supports all keys from QuantizationScheme except targets, which
        will be set to the targets parameter set at the modifier level.
    :param sequential_targets: class names of decoding layers to tune sequentially. If
        None, targets are inferred via `get_no_split_params()` to respect no-split
        constraints for large models. Defaults to None.
    :param iters: number of tuning iterations per block (decoding layer). Higher values
        typically improve accuracy at the cost of longer tuning time. Defaults to 200.
    :param enable_torch_compile: whether to enable `torch.compile` to accelerate the
        tuning loop. Disable if your environment or model encounters compilation issues.
        Defaults to True.
    :param batch_size: calibration/tuning batch size used by AutoRound when optimizing
        rounding/clipping parameters. Larger values can improve stability but require
        more memory. Defaults to 8.
    :param device_ids: optional device map string for layer dispatch during tuning.
        Examples: "0,1" for cuda:0 and cuda:1, or "auto" to use all available GPUs.
        When None, no dispatching occurs and the model remains on its current device.
        Defaults to None.
    """

    sequential_targets: Union[str, List[str], None] = None
    # AutoRound modifier arguments
    iters: int = 200
    enable_torch_compile: bool = True
    batch_size: int = 8
    lr: Optional[float] = None
    device_ids: Optional[str] = None

    # private variables
    _all_module_input: Dict[str, List[Tuple]] = PrivateAttr(default_factory=dict)
    _q_input: Optional[torch.Tensor] = PrivateAttr(default=None)

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize the model state for quantization and calibration.

        :param state: session state storing input model and calibration data
        """
        # apply config to model and prepare calibration hooks
        if QuantizationMixin.has_config(self):
            QuantizationMixin.initialize_quantization(self, state.model)

        # prepare module names
        self._add_temporary_names(state.model)
        # freeze all model parameters
        for _, param in state.model.named_parameters():
            param.requires_grad_(False)

        self.sequential_targets = self._infer_sequential_targets(state.model)
        return True

    def start_calibration(self, model: torch.nn.Module):
        """
        Register activation calibration hooks and enable quantization as we calibrate

        :param model: model to prepare for calibration
        """
        targets = match_named_modules(model, self.targets, self.ignore)
        if targets_embeddings(model, targets):
            untie_word_embeddings(model)

        for _, module in match_named_modules(model, self.targets, self.ignore):
            # skip register observers for auto-round
            apply_calibration_status(module)

        model.apply(enable_quantization)  # quantize at the same time as calibrate

    def input_capture_hook(self, module, *args, **kwargs):
        if module._tmp_name not in self._all_module_input:
            self._all_module_input[module._tmp_name] = []
        self._all_module_input[module._tmp_name].append((args, kwargs))

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

        # register quantization calibration hooks
        # assume quantization has been initialized by this modifier or one before it
        self.start_calibration(state.model)
        for _, module in state.model.named_modules():
            if self._is_decoding_layer(module):
                # register input capture hook for decoding layers
                self.register_hook(
                    module, self.input_capture_hook, "forward_pre", with_kwargs=True
                )

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        if event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            subgraph = kwargs.pop("subgraph", None)
            self.apply_autoround(state, subgraph)
            self.post_autoround_cleanup()

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            if not self.ended_:
                self.on_end(state, None)

    def apply_autoround(self, state, subgraph):
        """
        Applies AutoRound quantization tuning on the current decoding layer.

        The tuning logic is as follows:
        for iter in range(iters):
            quant_output = forward(layer, cached_inputs)
            loss = mse_loss(quant_output, original_output)
            loss.backward()
            optimizer.step()
            if loss < best_loss:
                best_params = update_params(layer)

        For more details, please refer to the AutoRound repository:
        https://github.com/intel/auto-round/
        """
        modules = list(subgraph.submodules(model=state.model))

        decoding_layers = [m for m in modules if self._is_decoding_layer(m)]
        if len(decoding_layers) == 0:
            return
        assert len(decoding_layers) == 1, (
            "Only one decoding layer is expected in the subgraph, "
            f"found {len(decoding_layers)}."
        )
        decoding_layer = decoding_layers[0]

        logger.info("Applying AutoRound on layer {}", decoding_layer._tmp_name)

        wrapped_model = _wrap_decoding_layer(decoding_layer)
        wrapped_model.name_or_path = state.model.name_or_path
        wrapped_model.config = state.model.config

        with (
            torch.enable_grad(),
            align_module_device(decoding_layer),
            suspend_accelerate_hooks(wrapped_model),
        ):
            ar_quant_scheme = self._mapping_config_to_autoround()
            fp_layers = self.get_unquantized_layer_names(decoding_layer)
            ar = AutoRound(
                model=wrapped_model,
                tokenizer="",
                scheme=ar_quant_scheme,
                iters=self.iters,
                enable_torch_compile=self.enable_torch_compile,
                batch_size=self.batch_size,
                lr=self.lr,
                device_map=self.device_ids,
                fp_layers=",".join(fp_layers) if fp_layers else "",
            )
            # TODO: configure layer-wise config based on self.resolved_config
            ar.configure_layer_config(enable_gguf_official_mixed=False)
            ar.batch_dim = 0
            first_param = next(decoding_layer.parameters())
            device = first_param.device
            cur_inputs = self._all_module_input[decoding_layer._tmp_name]
            decoding_layer.tuning_device = device
            # Leave offload for LLMC to handle if `device_ids` is not set
            auto_offload = False
            if self.device_ids is not None:
                # When device_ids is set, we move decoding layer to CPU first,
                # then the submodules will be re-dispatched by AutoRound.
                decoding_layer.to("cpu")
                auto_offload = True

            q_input, _ = ar.quantize_block(
                block=decoding_layer,
                inputs=cur_inputs,
                q_input=self._q_input,
                device=str(device),
                auto_offload=auto_offload,
            )
            self._q_input = q_input

            decoding_layer = self._unwrapper_quantized_layer(decoding_layer)

            # Update offload parameters and remove temporary attributes
            for name, module in decoding_layer.named_modules():
                if (
                    hasattr(module, "weight_scale")
                    and hasattr(module, "weight_zero_point")
                    and hasattr(module, "scale")
                ):
                    # Note: The model's weight is already q-dq in-place by auto-round.
                    weight_scale = module.scale
                    del module.scale
                    # TODO: update zero_point after supporting asymmetric quantization
                    update_offload_parameter(module, "weight_scale", weight_scale)

                if hasattr(module, "act_scale") and hasattr(module, "input_scale"):
                    act_scale = module.act_scale
                    assert act_scale.numel() == module.input_scale.numel(), (
                        f"Expected act_scale of size {module.input_scale.numel()}, "
                        "got {act_scale.numel()}"
                    )
                    del module.act_scale

                    # activation scale shape maybe different
                    update_offload_parameter(
                        module,
                        "input_scale",
                        act_scale.reshape(module.input_scale.shape),
                    )

        decoding_layer.eval()

    def post_autoround_cleanup(self):
        self._all_module_input.clear()

    def on_end(self, state: State, event: Event, **kwargs):
        """
        Finish calibrating by removing observers and calibration hooks
        """
        self.ended_ = True
        QuantizationMixin.end_calibration(self, state.model)
        self._remove_temporary_names(state.model)
        self.remove_hooks()
        self._q_input = None

    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        disable the quantization observers used by the AutoRound algorithm

        :param state: session state storing input model and calibration data
        """
        if not self.ended_:
            self.on_end(state, None)

        return True

    def get_unquantized_layer_names(self, wrapped_model: torch.nn.Module) -> List[str]:
        unquantized_layers = []

        for name, module in wrapped_model.named_modules():
            if (
                module.__class__.__name__ in self.resolved_targets
                and getattr(module, "quantization_scheme", None) is None
            ):
                unquantized_layers.append(name)
        return unquantized_layers

    def _unwrapper_quantized_layer(self, model: torch.nn.Module):
        # auto-round will return WrapperWALayer if activation is quantized
        for name, module in model.named_modules():
            if isinstance(module, WrapperWALayer):
                if "." in name:
                    parent, child = name.rsplit(".", maxsplit=1)
                    parent = model.get_submodule(parent)
                    setattr(parent, child, module.orig_layer)
                else:
                    # It's a top-level module
                    setattr(model, name, module.orig_layer)
        return model

    def _add_temporary_names(self, model: torch.nn.Module):
        for name, mod in model.named_modules():
            mod._tmp_name = name

    def _remove_temporary_names(self, model: torch.nn.Module):
        for _, mod in model.named_modules():
            if hasattr(mod, "_tmp_name"):
                del mod._tmp_name

    def _is_decoding_layer(self, module: torch.nn.Module) -> bool:
        return module.__class__.__name__ in self.sequential_targets

    def _infer_sequential_targets(self, model: torch.nn.Module) -> str | list[str]:
        match self.sequential_targets:
            case None:
                return get_no_split_params(model)
            case str():
                return [self.sequential_targets]
            case _:
                return self.sequential_targets

    def _mapping_config_to_autoround(self):
        resolved_config = self.resolved_config
        quant_scheme = None
        # TODO: release below constraint in later PRs
        assert len(resolved_config.config_groups) == 1, (
            "AutoRoundModifier only supports one quantization scheme for now, "
            f"got {len(resolved_config.config_groups)}"
        )

        for scheme in resolved_config.config_groups.values():
            assert isinstance(
                scheme, QuantizationScheme
            ), f"Expected QuantizationScheme, got {type(scheme)}"
            quant_scheme = scheme
        weight_args = quant_scheme.weights
        activation_args = quant_scheme.input_activations
        assert quant_scheme.output_activations is None, (
            "Output activation quantization is not supported in AutoRoundModifier, "
            f"got {quant_scheme.output_activations}"
        )
        group_size = weight_args.group_size
        data_type = weight_args.type
        if group_size is None:
            if weight_args.strategy == QuantizationStrategy.CHANNEL:
                group_size = -1
            elif weight_args.strategy == QuantizationStrategy.TENSOR:
                group_size = 0
            else:
                raise ValueError(
                    "AutoRoundModifier only supports channel-wise and tensor-wise "
                    "weight quantization"
                )

        if data_type == "float":
            data_type = "fp"

        if activation_args is None:
            act_bits = 16
            act_group_size = None
            act_symmetric = None
            act_dynamic = None
            act_data_type = None
        else:
            act_dynamic = activation_args.dynamic
            act_group_size = activation_args.group_size
            act_symmetric = activation_args.symmetric
            act_bits = activation_args.num_bits

            # activation is quantized dynamically, skip collecting scale in auto-round
            if act_dynamic:
                act_bits = 16

            act_data_type = activation_args.type
            assert activation_args.strategy != QuantizationStrategy.GROUP, (
                "Input activation group-wise quantization is not supported "
                "in AutoRoundModifier"
            )
            if act_group_size is None:
                if activation_args.strategy in [
                    QuantizationStrategy.CHANNEL,
                    QuantizationStrategy.TOKEN,
                ]:
                    act_group_size = -1
                elif activation_args.strategy == QuantizationStrategy.TENSOR:
                    act_group_size = 0
                else:
                    raise ValueError(
                        f"{activation_args.strategy} is not supported "
                        "in AutoRoundModifier"
                    )

            if act_data_type == "float":
                act_data_type = "fp"

        ar_quant_scheme = ARQuantizationScheme(
            bits=weight_args.num_bits,
            sym=weight_args.symmetric,
            group_size=group_size,
            data_type=data_type,
            act_bits=act_bits,
            act_group_size=act_group_size,
            act_sym=act_symmetric,
            act_dynamic=act_dynamic,
            act_data_type=act_data_type,
        )
        return ar_quant_scheme
