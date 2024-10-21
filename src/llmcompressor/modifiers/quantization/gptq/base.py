import gc
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence
import contextlib
from functools import partial
from compressed_tensors.quantization import (
    QuantizationScheme,
    freeze_module_quantization,
)
from loguru import logger
from pydantic import Field, field_validator

from llmcompressor.core import State
from llmcompressor.modifiers import Modifier, ModifierFactory
from llmcompressor.modifiers.quantization.gptq.utils import (
    get_output_error,
    gptq_hook
)
from llmcompressor.modifiers.quantization.gptq.utils.gptq_quantize import quantize_weight
from llmcompressor.modifiers.quantization.gptq.utils.helpers import MetricsLogger
from llmcompressor.modifiers.quantization.quantization.base import QuantizationModifier
from llmcompressor.modifiers.utils.pytorch_helpers import run_calibration_forward
from llmcompressor.transformers.finetune.data.data_helpers import format_calibration_data
from llmcompressor.utils.fsdp.context import fix_fsdp_module_name
from llmcompressor.utils.helpers import calibration_forward_context, align_module, getattr_chain
from llmcompressor.utils.pytorch.module import (
    get_layers,
    get_no_split_params,
    qat_active,
)

from compressed_tensors.utils import (
    get_offloaded_device,
    is_module_offloaded,
    update_parameter_data,
    update_prefix_dict,
)

__all__ = ["GPTQModifier"]


class GPTQModifier(Modifier):
    """
    Modifier for applying the one-shot OBCQ algorithm to a model

    Lifecycle:
        - on_initialize
            - initialize_compression()
                - compressible_layers()
                - LayerCompressor.pre_compress()
            - apply_compression()
                - run_calibration_forward()
                - LayerCompressor.compress()
                - LayerCompressor.post_compress()
                - LayerCompressor.revert_layer_wrappers()
    | Sample yaml:
    | test_stage:
    |    obcq_modifiers:
    |      GPTQModifier:
    |          sequential_update: true
    |          dampening_frac: 0.001
    |          block_size: 128
    |          config_groups:
    |            group_0:
    |                targets:
    |                  - "Linear"
    |                input_activations: null
    |                output_activations: null
    |                weights:
    |                    num_bits: 8
    |                    type: "int"
    |                    symmetric: true
    |                    strategy: "tensor"
    |                    group_size: 128
    |                    actorder: False


    :param sequential_update: Whether or not to update weights sequentially by layer,
        True saves on GPU memory, default is True
    :param targets: list of layer names to compress during GPTQ, or '__ALL__'
        to compress every layer in the model
    :param block_size: Used to determine number of columns to compress in one pass
    :param quantize: Set to True to quantize using an existing quantization modifier,
        or pass in the configuration for a quantization modifier if one does not
        already exist in the recipe
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    :param config_groups: [Used, if a quantization modifier is not specified],
        dictionary specifying quantization schemes to apply to target
        modules. Modules not matching a scheme target will NOT be quantized.
    :param ignore: [Used, if a quantization modifier is not specified]
        optional list of module class names or submodule names to not
        quantize even if they match a target in config_groups. Defaults to empty list.
    :param disable_quantization_observer_epoch: [Used, if a quantization modifier is
        not specified] Epoch to disable updates to the module
        quantization observers. At this point, quantized weights and zero points will
        not be updated. Leave None to not disable observers during QAT. Default is None
    :param num_calibration_steps: Number of steps to run post training calibration for.
        When None, the entire calibration_dataloader is used
    :param scheme: [Used, if a quantization modifier is not specified], the quantization
        scheme to apply to the model, this is a dictionary that supports all keys from
        QuantizationScheme except targets, which will be set to the targets parameter
        set at the modifier level. Can also be set to a dictionary of the format
        `preset_scheme_name: targets` for example: `W8A8: ['Linear']` for weight 8 bit
        or a string of a preset scheme if targets is provided
        and activation 8 bit quantization on the Linear layers.
    """

    sequential_update: bool = True
    true_sequential: bool = False
    targets: Union[str, List[str], None] = None
    sequential_targets: Union[str, List[str], None] = None
    block_size: int = 128
    quantize: Union[bool, Dict] = True
    dampening_frac: Optional[float] = 0.01
    config_groups: Optional[Dict[str, QuantizationScheme]] = None
    ignore: List[str] = Field(default_factory=list)
    disable_quantization_observer_epoch: Optional[float] = None
    num_calibration_steps: Optional[int] = None
    scheme: Optional[Union[str, Dict[str, Any]]] = None

    _layer_index: int = 0
    _num_layers: int = 0
    _hooks_disabled: bool = False
    quantization_modifier_: Optional[QuantizationModifier] = None

    @field_validator("sequential_update", mode="before")
    def validate_sequential_update(cls, value: bool) -> bool:
        if not value:
            logger.warning(
                "Not using sequential_update requires allocating all hessians in "
                "GPU memory. If you are running into GPU memory issues, consider "
                "using sequential_update=True"
            )

        return value
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._layer_index = 0
        self._num_layers = 0
        self.quantization_modifier_ = None

    def on_initialize_structure(self, state: State, **kwargs):
        """
        Check the model's quantization state matches that expected by this modifier,
        adding a default quantization scheme if needed

        TODO: Depreciate and fold into `on_initialize`

        :param state: session state storing input model and calibration data
        """
        quantization_already_active = qat_active(state.model)
        if isinstance(self.quantize, bool):
            if not self.quantize and quantization_already_active:
                logger.warning(
                    "GPTQ quantization is set to False, but a "
                    "quantization modifier is already active on the model "
                    "resetting quantize to True"
                )
                self.quantize = True
            elif self.quantize and not quantization_already_active:
                logger.warning(
                    "GPTQ quantization is set to True without an "
                    "active quantization modifier."
                )
                self._build_quant_modifier()
            return  # use existing quantization modifier if there is one
        else:
            if not isinstance(self.quantize, Dict):
                raise ValueError(
                    "GPTQModifier.quantize accepts only a single "
                    "quantization modifier or a boolean. Found "
                    f"type {type(self.quantize)}"
                )
            if len(self.quantize) != 1:
                raise ValueError(
                    "GPTQModifier.quantize accepts only a single "
                    "quantization modifier or a boolean. Found "
                    f"{len(self.quantize)} modifiers"
                )
            if quantization_already_active:
                logger.warning(
                    "Attempting to initialize quantization for GPTQ "
                    "but a quantization modifier has already been applied. "
                    "The quantization configuration defined under the "
                    "GPTQ modifier will be ignored."
                )
                self.quantize = True
                return
            self._build_quant_modifier_from_dict(self.quantize)
            self.quantize = True

        if self.quantization_modifier_:
            self.quantization_modifier_.on_initialize_structure(state, **kwargs)

    def on_initialize(self, state: "State", **kwargs) -> bool:
        """
        Initialize and run the GPTQ algorithm on the current state

        :param state: session state storing input model and calibration data
        """
        if not self.initialized_structure_:
            self.on_initialize_structure(state, **kwargs)
        if self.quantization_modifier_:
            self.quantization_modifier_.initialize(state, **kwargs)
        if not self.quantize:
            raise ValueError("To use the GPTQModifier, quantization must be enabled.")

        # find layers (used for printing even if true_sequential=True)
        # if no targets are provided, default to the modules that shouldn't be
        # split by FSDP. For Transformers models this is equivalent to the
        # decoder layers (ie LlamaDecoderLayer)
        if self.sequential_targets is None:
            self.sequential_targets = get_no_split_params(state.model)
        layers = get_layers(self.sequential_targets, state.model)
        self._num_layers = len(layers)

        # add hooks to targets and layers
        # after lifecycle refactor, move this to pre_batch
        self.register_hooks(state.model, layers)

        # apply calibration and trigger hooks (hooks are self removing)
        self.calibration_forward(state.model, state.data.calib)

        # freeze quantization
        # after lifecycle refactor, move this to post_batch
        state.model.apply(freeze_module_quantization)

        return True
    
    def on_finalize(self, state: "State", **kwargs) -> bool:
        """
        disable the quantization observers used by the OBCQ algorithm

        :param state: session state storing input model and calibration data
        """
        if self.quantization_modifier_:
            self.quantization_modifier_.finalize(state, **kwargs)

        self.remove_gptq_hooks(state.model)

        return True
    
    def register_hooks(self, model: torch.nn.Module, layers: Dict[str, torch.nn.Module]):
        for name, module in model.named_modules():
            if getattr_chain(module, "quantization_scheme.weights", None) is not None:
                pre_hook = partial(self.target_pre_forward, name)
                post_hook = partial(self.target_post_forward, name)
                module._gptq_pre_hook = module.register_forward_pre_hook(pre_hook)
                module._gptq_post_hook = module.register_forward_hook(post_hook)

            if module in layers.values():
                pre_hook = partial(self.layer_pre_forward, name)
                post_hook = partial(self.layer_post_forward, name)
                module._gptq_pre_hook = module.register_forward_pre_hook(pre_hook)
                module._gptq_post_hook = module.register_forward_hook(post_hook, with_kwargs=True)

    def calibration_forward(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader):
        dataset = dataloader.dataset
        def collate_fn(batch):
            # extract input_ids and attention_mask from the batch
            input_ids = [torch.tensor(item["input_ids"]) for item in batch]
            attention_masks = [torch.tensor(item["attention_mask"]) for item in batch]
            
            # pad sequences in the batch
            padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
            padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

            return {
                "input_ids": padded_input_ids,
                "attention_mask": padded_attention_masks
            }
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        with calibration_forward_context(model):
            run_calibration_forward(model, dataloader, mask_padding=True)

    @gptq_hook
    def target_pre_forward(self, name: str, module: torch.nn.Module, args):
        if self.true_sequential:
            # compress first so output is from quantized weights
            self.quantize_module(name, module, args)
        
    @gptq_hook
    def target_post_forward(self, name: str, module: torch.nn.Module, args: torch.Tensor, _output: Any):
        if not self.true_sequential:
            # compress after so output is from unquantized weights
            self.quantize_module(name, module, args)
        
    @gptq_hook
    def layer_pre_forward(self, name: str, module: torch.nn.Module, args: Any):
        logger.info(f"\n===== Compressing layer {self._layer_index}/{self._num_layers} =====")
        
    @gptq_hook
    def layer_post_forward(self, name: str, module: torch.nn.Module, args: torch.Tensor, kwargs: Dict[str, Any], output: Any):
        if not self.true_sequential:
            # rerun with (now) quantized weights
            with self.disable_hooks():
                output = module(*args, **kwargs)

        self._layer_index += 1
        return output

    def quantize_module(self, name, module, args):
        logger.info(f"Compressing {name}...")

        inp = args[0]  # Assume that first argument is input (true for most Module types)
        quant_args = getattr_chain(module, "quantization_scheme.weights")

        # with onloaded weight
        with align_module(module), MetricsLogger(module) as metrics_logger:
            losses, quantized_weight, scale, zero_point, g_idx = quantize_weight(
                module.weight.data,
                inp,
                quant_args,
                blocksize=self.block_size,
                percdamp=self.dampening_frac,
                module_class=type(module),
            )
        
            #weight = torch.lerp(module.weight.data, quantized_weight, self.alpha)
            weight = quantized_weight
        
            if is_module_offloaded(module):
                update_prefix_dict(self.layer, "weight", weight)
            update_parameter_data(module, scale, "weight_scale")
            update_parameter_data(module, zero_point, "weight_zero_point")
            update_parameter_data(module, g_idx, "weight_g_idx")

            metrics_logger.set_losses(losses)
        
    @contextlib.contextmanager
    def disable_hooks(self):
        try:
            self._hooks_disabled = True
            yield
        finally:
            self._hooks_disabled = False

    def remove_gptq_hooks(self, module: torch.nn.Module, recurse: bool = True):
        if hasattr(module, "_gptq_pre_hook"):
            module._gptq_pre_hook.remove()
            delattr(module, "_gptq_pre_hook")

        if hasattr(module, "_gptq_post_hook"):
            module._gptq_post_hook.remove()
            delattr(module, "_gptq_post_hook")

        if recurse:
            for child_module in module.children():
                self.remove_gptq_hooks(child_module)

    def _build_quant_modifier(self):
        """
        Build a quantization modifier based on the specified config_groups,
        ignore list, and num_calibration_steps.

        :postcondition: self.quantization_modifier_ is set to the built
            quantization modifier
        """

        quantization_args_names = [
            "config_groups",
            "targets",
            "scheme",
            "num_calibration_steps",
            "ignore",
            "disable_quantization_observer_epoch",
        ]

        quant_args = {
            key: getattr(self, key)
            for key in quantization_args_names
            if getattr(self, key, False)
        }

        logger.info(f"Building quantization modifier with args: {quant_args}")
        vllm_quant_config = {"QuantizationModifier": quant_args}
        self._build_quant_modifier_from_dict(vllm_quant_config)

    def _build_quant_modifier_from_dict(self, quant_config):
        modifier_type = list(quant_config.keys())[0]
        modifier_args = quant_config[modifier_type]
        self.quantization_modifier_ = ModifierFactory.create(
            modifier_type,
            allow_registered=True,
            allow_experimental=True,
            **modifier_args,
        )
