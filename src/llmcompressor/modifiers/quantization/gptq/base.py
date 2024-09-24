from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from compressed_tensors.quantization import (
    QuantizationScheme,
    disable_quantization,
    enable_quantization,
    freeze_module_quantization,
)
from loguru import logger
from pydantic import Field
from torch.nn import Module

from llmcompressor.core.state import State
from llmcompressor.modifiers import Modifier, ModifierFactory
from llmcompressor.modifiers.quantization.gptq.utils.gptq_wrapper import GPTQWrapper
from llmcompressor.modifiers.utils.layer_compressor import LayerCompressor
from llmcompressor.modifiers.utils.pytorch_helpers import run_calibration_forward
from llmcompressor.utils.fsdp.context import fix_fsdp_module_name
from llmcompressor.utils.pytorch.module import (
    get_layers,
    get_no_split_params,
    qat_active,
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
    |          sequential_update: True
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
        True saves on GPU memory
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

    sequential_update: Optional[bool] = False
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

    model: Optional[Any] = None
    layer_compressors_: Optional[List[Any]] = None
    compressible_layers_: Optional[List] = None
    quantization_modifier_: Any = None

    def on_initialize_structure(self, state: State, **kwargs):
        """
        Check the model's quantization state matches that expected by this modifier,
        adding a default quantization scheme if needed

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
        # Checking if a GPTQ Modifier has set a Quantization Modifier - if yes, initialize it (similar to how the GPTQ Modifier was)
        if self.quantization_modifier_:
            breakpoint()
            self.quantization_modifier_.initialize(state, **kwargs) # Initialize the zero points and scales for each of the layers; remove the weight observers for W4A16 (why are they there?)
            # Will run calibration of possible - not for W4A16 - what is this calibrating? [Look into this]
        if not self.quantize:
            raise ValueError("To use the GPTQModifier, quantization must be enabled.")

        modifiable_model = state.model
        calibration_dataloader = state.data.calib
        breakpoint()

        if self.sequential_targets is None:
            # if no targets are provided, default to the modules that shouldn't be
            # split by FSDP. For Transformers models this is equivalent to the
            # decoder layers (ie LlamaDecoderLayer)
            self.sequential_targets = get_no_split_params(modifiable_model)

        self.initialize_compression(modifiable_model, calibration_dataloader) # Initialize compressors; wraps each layer with a GPTQModifier - adds hook for hessian calculation? add batch is added as a hook by the llm_compressor but actually implemented by the GPTQWrapper
        breakpoint()
        self.apply_compression(calibration_dataloader)
        state.model.apply(freeze_module_quantization)
        breakpoint()

        return True

    def on_finalize(self, state: "State", **kwargs) -> bool:
        """
        disable the quantization observers used by the OBCQ algorithm

        :param state: session state storing input model and calibration data
        """
        if self.quantization_modifier_:
            self.quantization_modifier_.finalize(state, **kwargs)

        return True

    def compressible_layers(self) -> Dict:
        """
        Retrieves the modules corresponding to a list of
        compressible layer names

        :precondition: self.model is set and is a torch.nn.Module
        :return: dictionary of modules to compress
        """
        if not isinstance(self.model, Module):
            raise ValueError(
                "`self.model` must be a torch.nn.Module to use "
                f"the {self.__class__.__qualname__} modifier but got "
                f"{type(self.model)} instead"
            )

        return get_layers(self.sequential_targets, self.model)

    def initialize_compression(
        self,
        model: Module,
        dataloader: Optional[Iterable[Tuple[List, Dict[str, Any]]]] = None,
    ):
        """
        Setup for GPTQ, initializes the model
        and other parameters, also initilializes the
        compressible layers of model, and sets the device

        :param model: model to initialize for compression
        :param dataloader: calibration data for GPTQ
        """
        self.model = model
        self.compressible_layers_ = self.compressible_layers()
        self.layer_compressors_ = []

        # Iterate through for every layer, add a LayerCompressor
        # LayerCompressor gets the layer and the GPTQWrapper
        # Will wrap each layer with the GPTQQWrapper + hook to calculate 
        # hessians - is this what calculates the hessians?
        for idx, (name, layer) in enumerate(self.compressible_layers_.items()):
            name = fix_fsdp_module_name(name)
            logger.info(f"Preparing {name} for compression")
            args = self._pruning_arguments()
            comp_cls = self._compression_class()
            compressor = LayerCompressor(comp_cls, self.model, layer, idx, name, args)
            if not self.sequential_update:
                # add all batch processing hooks before the forward pass
                compressor.pre_compress() # Why do we have to pre_compress now if not sequential update? - hessian memory
            self.layer_compressors_.append(compressor)

        breakpoint()
        if self.sequential_update:
            first_layer_compressor = self.layer_compressors_[0]
            first_layer_compressor.set_early_stop() # artificially trigger an exception after the first layer

    @torch.no_grad()
    def apply_compression(
        self, dataloader: Optional[Iterable[Tuple[List, Dict[str, Any]]]] = None
    ) -> Dict:
        """
        Run GPTQ on the loaded model, using dataloader as calibration data

        :param dataloader: calibration data for GPTQ
        """
        class_name = self.__class__.__name__.replace("PyTorch", "")
        logger.info(
            f"Running {class_name} calibration with " f"{len(dataloader)} samples..."
        )

        # Zero-points and weights were added as part of `set_module_for_calibration` call in on_ititialize
        # Where are the observers attached?
        # quantization scales and zp are already initialized but we do not
        # want to calibrate wrt to these

        # Why do we have to disabled if we already froze?
        
        self.model.apply(disable_quantization)  # prevents the forward pass for the inputs

        forward_pass_use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

        # in non-sequential mode we run calibration through the full model
        # in sequential mode we run calibration up to the first transformer target

        # This one is always called; otherwise called at the beginning only if possible
        # data is passed forward, add batch is called to update hessian calculation?
        # where are the intermediates saved?
        # in non sequential: initialize hessians up front; uses more memory
        # in sequential mode: will be stopped early/before the first transformer block; will go through the entire model 
        intermediates = run_calibration_forward(
            self.model, dataloader, mask_padding=True
        )
        # needed to update the hessians
        self.layer_compressors_[0].clear_early_stop()

        num_layers = len(self.compressible_layers_)
        for idx, layer_compressor in enumerate(self.layer_compressors_):
            logger.info(f"\n===== Compressing layer {idx+1}/{num_layers} " " =====")

            # Prune/quantize using GPTQ
            if self.sequential_update:
                # in sequential mode we run the forward pass for each transformer layer
                # one at a time, caching the intermediate outputs between layers
                layer_compressor.pre_compress() # Why do we have to do this here?
                logger.info(f"Calibrating {layer_compressor.name}...")
                intermediates = layer_compressor.calibrate_layer(intermediates)

            # Compressor calls fake quantize - use the hessians + QDQ to update the weights - how are the scales or zeros ever updated?
            # scales/zp are created by the observer and then attached to the module in `set_module_for_calibration` --> but when are they updated based on the updated calculation?
            # still in dense form until saved to disk at which point, save_pretrained_compressed is called - creates the compressor which will use the updates zp/scales and compress the model on disk
            # If this is the case, then when are the weights/scales updated? when is the moving average updated
            layer_compressor.compress()
            layer_compressor.post_compress()
            layer_compressor.revert_layer_wrappers()
            torch.cuda.empty_cache()

        self.model.config.use_cache = forward_pass_use_cache

        # re-enable quantization
        self.model.apply(enable_quantization)

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
        # When GPTQ modifier is created, it will build a quant modifier as well
        self.quantization_modifier_ = ModifierFactory.create(
            modifier_type,
            allow_registered=True,
            allow_experimental=True,
            **modifier_args,
        )
        breakpoint()

    def _pruning_arguments(self):
        """
        Gather the parameters needed for root module compression in a dict

        :return: dict of params for pruning
        """
        return {
            "blocksize": self.block_size,
            "percdamp": self.dampening_frac,
        }

    def _compression_class(self):
        """
        :return: wrapper class used for root modules of this compression class
        """
        return GPTQWrapper
