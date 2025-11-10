from typing import Dict, List, Optional, Tuple, Union

import torch
from auto_round import AutoRound
from auto_round.schemes import QuantizationScheme as ARQuantizationScheme
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
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    untie_if_target_shared_embedding,
)
from llmcompressor.utils.pytorch.module import get_no_split_params

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


class AutoRoundModifier(Modifier, QuantizationMixin):
    """
    Implements the AutoRound algorithm from https://aclanthology.org/2024.findings-emnlp.662.pdf.
    This modifier leverages signed gradient descent (SignSGD) optimizer and
    block-wise loss to optimize rounding values and weight clipping in a few steps.

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
    """

    sequential_targets: Union[str, List[str], None] = None
    # AutoRound modifier arguments
    iters: int = 200
    enable_torch_compile: bool = True

    # private variables
    _module_names: Dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
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
        self._module_names = {
            m: name
            for name, m in match_named_modules(state.model, self.targets, self.ignore)
        }
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
        untie_if_target_shared_embedding(model, self._module_names.values())

        for _, module in match_named_modules(model, self.targets, self.ignore):
            # Note: No need to register observers for auto-round
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

        with torch.enable_grad(), align_module_device(decoding_layer):
            ar_quant_scheme = self._mapping_config_to_autoround()
            ar = AutoRound(
                model=wrapped_model,
                tokenizer="",
                scheme=ar_quant_scheme,
                iters=self.iters,
                enable_torch_compile=self.enable_torch_compile,
            )
            # TODO: configure layer-wise config based on self.resolved_config
            ar.configure_layer_config(enable_gguf_official_mixed=False)
            ar.batch_dim = 0
            first_param = next(decoding_layer.parameters())
            device = first_param.device
            cur_inputs = self._all_module_input[decoding_layer._tmp_name]
            decoding_layer.tuning_device = device

            q_input, _ = ar.quantize_block(
                block=decoding_layer,
                inputs=cur_inputs,
                q_input=self._q_input,
                device=device,
                # Leave offload for LLMC
                auto_offload=False,
            )
            self._q_input = q_input
            # Update offload parameters and remove temporary attributes
            for _, module in decoding_layer.named_modules():
                if hasattr(module, "weight_scale") and hasattr(
                    module, "weight_zero_point"
                ):
                    # Note: The model's weight is already q-dq in-place by auto-round.
                    weight_scale = module.scale
                    del module.scale
                    # TODO: update zero_point after supporting asymmetric quantization
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
