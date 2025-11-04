from typing import Dict, List, Optional, Tuple

import torch
from compressed_tensors.quantization import (
    QuantizationConfig,
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

__all__ = ["AutoRoundModifier"]


def _is_decoding_layer(module, name):
    return "decoderlayer" in module.__class__.__name__.lower()


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


class AutoRoundModifier(Modifier, QuantizationMixin):
    """
    Implements the AutoRound algorithm from https://arxiv.org/pdf/2309.05516. This modifier
    leverages signed gradient descent (SignSGD) and block-wise loss to optimize rounding values
    and weight clipping in a few steps.

    | Sample yaml:
    | test_stage:
    |    modifiers:
    |      AutoRoundModifier:
    |          iters: 200
    |          config_groups:
    |            group_0:
    |                targets:
    |                  - "Linear"
    |                input_activations: null
    |                output_activations: null
    |                weights:
    |                    num_bits: 4
    |                    type: "int"
    |                    symmetric: true
    |                    strategy: group
    |                    group_size: 128

    Lifecycle:
        - on_initialize
            - apply config to model
        - on_start
            - add input/output capture hooks to decoding layers
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
        will be set to the targets parameter set at the modifier level. Can also be set
        to a dictionary of the format `preset_scheme_name: targets` for example:
        `W8A8: ['Linear']` for weight and activation 8-bit.
    """

    # AutoRound modifier arguments
    iters: Optional[int] = 200

    # private variables
    _module_names: Dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _cur_layer_idx = PrivateAttr(default=0)
    _all_module_input: Dict[str, List[Tuple]] = PrivateAttr(default_factory=dict)

    def resolve_quantization_config(self) -> QuantizationConfig:
        config = super().resolve_quantization_config()
        return config

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize and run the AutoRound algorithm on the current state

        :param state: session state storing input model and calibration data
        """
        # apply config to model and prepare calibration hooks
        if QuantizationMixin.has_config(self):
            QuantizationMixin.initialize_quantization(self, state.model)

        # prepare module names
        self._module_names = {
            m: name
            for name, m in match_named_modules(state.model, self.targets, self.ignore)
        }
        # add temporary names to all modules
        for name, mod in state.model.named_modules():
            mod._tmp_name = name
        # freeze all model parameters
        for name, param in state.model.named_parameters():
            param.requires_grad_(False)
        return True

    def start_calibration(self, model: torch.nn.Module):
        """
        Register activation calibration hooks and enable quantization as we calibrate

        :param model: model to prepare for calibration
        """

        for _, module in match_named_modules(model, self.targets, self.ignore):
            # Note: No need to register observers for auto-round
            # self._initialize_observers(module)
            self._calibration_hooks |= self._initialize_hooks(module)
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
        for name, module in state.model.named_modules():
            if _is_decoding_layer(module, name):
                # register input/output capture hooks for decoding layers
                self.register_hook(
                    module, self.input_capture_hook, "forward_pre", with_kwargs=True
                )

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        if event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            self.apply_autoround(state)
            self.post_autoround_cleanup()

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            if not self.ended_:
                self.on_end(state, None)

    def _mapping_config_to_autoround(self):
        from auto_round.schemes import QuantizationScheme as ARQuantizationScheme

        resolved_config = self.resolved_config
        quant_scheme = None
        for scheme in resolved_config.config_groups.values():
            assert isinstance(
                scheme, QuantizationScheme
            ), f"Expected QuantizationScheme, got {type(scheme)}"
            quant_scheme = scheme
        weight_args = quant_scheme.weights
        # TODO: release below constraint in later PRs
        assert weight_args.strategy == QuantizationStrategy.GROUP, (
            "Only group-wise quantization is supported in AutoRoundModifier for now, "
            f"got {weight_args.strategy}"
        )
        assert quant_scheme.input_activations is None, (
            "Input activation quantization is not supported in AutoRoundModifier, "
            f"got {quant_scheme.input_activations}"
        )
        assert quant_scheme.output_activations is None, (
            "Output activation quantization is not supported in AutoRoundModifier, "
            f"got {quant_scheme.output_activations}"
        )
        ar_quant_scheme = ARQuantizationScheme(
            bits=weight_args.num_bits,
            sym=weight_args.symmetric,
            group_size=weight_args.group_size,
            data_type=weight_args.type,
            act_bits=16,
        )
        return ar_quant_scheme

    def apply_autoround(self, state):
        """Applies AutoRound quantization tuning on the current decoding layer.

        The tuning logic is below:
        for iter in range(iters):
           quant_output = forward(layer, cached_inputs)
           loss = mse_loss(quant_output, original_output)
           loss.backward()
           optimizer.step()
           if loss < best_loss:
                best_params = save_params(layer)
        For more details, please refer to the AutoRound repository:
        https://github.com/intel/auto-round/
        """

        cur_layer_idx = self._cur_layer_idx
        self._cur_layer_idx += 1
        logger.info(f">>||>> AutoRound for decoding layer index {cur_layer_idx}")
        if cur_layer_idx >= len(state.model.model.layers):
            # skip the lm_head layer
            return
        decoding_layer = state.model.model.layers[cur_layer_idx]

        wrapped_model = _wrap_decoding_layer(decoding_layer)

        with torch.enable_grad(), align_module_device(decoding_layer):
            import auto_round

            parsed_scheme = self._mapping_config_to_autoround()
            ar = auto_round.AutoRound(
                model=wrapped_model,
                tokenizer="",
                scheme=parsed_scheme,
                iters=self.iters,
                enable_quanted_input=False,
                enable_torch_compile=True,
            )
            # TODO: configure layer-wise config based on self.resolved_config
            ar.configure_layer_config()
            first_param = next(decoding_layer.parameters())
            device = first_param.device
            input_name = f"model.layers.{cur_layer_idx}"
            cur_inputs = self._all_module_input[input_name]
            decoding_layer.tuning_device = device

            ar.quantize_block(
                block=decoding_layer,
                inputs=cur_inputs,
                normalize_inputs=True,
                device=device,
            )
            # Update offload parameters and remove temporary attributes
            for _, module in decoding_layer.named_modules():
                if hasattr(module, "weight_scale") and hasattr(
                    module, "weight_zero_point"
                ):
                    # Note: The model's weight is already q-dq in-place by auto-round.
                    weight_scale = module.scale
                    del module.scale
                    del module.zp
                    # TODO: update weight as well
                    update_offload_parameter(module, "weight_scale", weight_scale)
        decoding_layer.eval()

    def post_autoround_cleanup(self):
        self._all_module_input.clear()

    def on_end(self, state: State, event: Event, **kwargs):
        """
        Finish calibrating by removing observers and calibration hooks
        """
        self.ended_ = True
        QuantizationMixin.end_calibration(self, state.model)
        self.remove_hooks()  # remove gptq hooks

    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        disable the quantization observers used by the AutoRound algorithm

        :param state: session state storing input model and calibration data
        """
        if not self.ended_:
            self.on_end(state, None)

        return True
