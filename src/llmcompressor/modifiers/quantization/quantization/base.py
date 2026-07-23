import torch
import torch.distributed as dist
from compressed_tensors.distributed import (
    greedy_bin_packing,
    is_distributed,
)
from compressed_tensors.quantization.utils import is_module_quantized

from llmcompressor.core import Event, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.calibration import (
    observe,
    update_qparams,
)
from llmcompressor.modifiers.quantization.quantization.mixin import QuantizationMixin
from llmcompressor.observers import ACTIVATION_OBS
from llmcompressor.utils.dist import broadcast_qparams_and_cleanup

__all__ = ["QuantizationModifier"]

_WEIGHT_Q_PARAMS = ["weight_scale", "weight_zero_point", "weight_global_scale"]


class QuantizationModifier(Modifier, QuantizationMixin):
    """
    Enables post training quantization (PTQ) and quantization aware training (QAT) for a
    given module or its submodules. After calibration (PTQ) or the start epoch (QAT),
    the specified module(s) forward pass will emulate quantized execution and the
    modifier will be enabled until training is completed.

    In DDP mode, activation observer statistics are all-reduced across ranks at
    sequential layer boundaries so all ranks share identical quantization parameters.

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
    :param kv_cache_scheme: optional QuantizationArgs, that specify the
        quantization of the kv cache. If None, kv cache is not quantized.
        When applying kv cache quantization to transformer AutoModelForCausalLM,
        the kv_cache_scheme gets converted into a QuantizationScheme that:
            - targets the `q_proj` and `k_proj` modules of the model. The outputs
              of those modules are the keys and values that might be cached
            - quantizes the outputs of the aforementioned layers, so that
              keys and values are compressed before storing them in the cache
        There is an explicit assumption that the model contains modules with
        `k_proj` and `v_proj` in their names. If this is not the case
        and kv_cache_scheme != None, the quantization of kv cache will fail
    """

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Prepare to calibrate activations and weights

        According to the quantization config, a quantization scheme is attached to each
        targeted module. The module's forward call is also overwritten to perform
        quantization to inputs, weights, and outputs.

        Then, according to the module's quantization scheme, observers and calibration
        hooks are added. These hooks are disabled until the modifier starts.
        """
        if not QuantizationMixin.has_config(self):
            raise ValueError(
                "QuantizationModifier requires that quantization fields be specified"
            )
        QuantizationMixin.initialize_quantization(self, state.model)

        return True

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        """
        Begin calibrating activations.
        """
        QuantizationMixin.start_calibration(self, state.model)

    def on_sequential_epoch_end(
        self, state: State, event: Event, modules: list[torch.nn.Module], **kwargs
    ):
        modules = [module for module in modules if is_module_quantized(module)]
        self.sync_obs_act_stats(modules)
        update_qparams(modules, ACTIVATION_OBS)
        self.validate_module_calibration(state.model, modules, ACTIVATION_OBS)

        ### Not Distributed
        if not is_distributed():
            observe(modules, "weight")
            update_qparams(modules, "weight")
            self.validate_module_calibration(state.model, modules, "weight")
            return

        ### Distributed
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        module_list, rank_to_modules, module_to_rank = greedy_bin_packing(
            modules,
            world_size,
            item_weight_fn=lambda mod: mod.weight.numel(),
        )

        observe(rank_to_modules[rank], "weight")
        update_qparams(rank_to_modules[rank], "weight")
        broadcast_qparams_and_cleanup(module_list, module_to_rank, _WEIGHT_Q_PARAMS)
        self.validate_module_calibration(state.model, rank_to_modules[rank])

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        """
        Finish calibrating by removing observers and calibration hooks
        """
        QuantizationMixin.end_calibration(self, state.model)
