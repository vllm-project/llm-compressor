import tqdm
from compressed_tensors.utils import match_named_modules

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.calibration import (
    recompute_qparams_from_observer,
    update_weight_global_scale,
    update_weight_zp_scale,
)
from llmcompressor.modifiers.quantization.quantization.mixin import QuantizationMixin
from llmcompressor.modifiers.utils import update_fused_layer_weight_global_scales
from llmcompressor.utils.distributed import (
    all_reduce_max,
    all_reduce_min,
    broadcast_module_parameter,
    build_module_to_rank_map,
    get_rank,
    is_distributed,
    partition_modules_by_weight_size,
)

__all__ = ["QuantizationModifier"]


class QuantizationModifier(Modifier, QuantizationMixin):
    """
    Enables post training quantization (PTQ) and quantization aware training (QAT) for a
    given module or its submodules. After calibration (PTQ) or the start epoch (QAT),
    the specified module(s) forward pass will emulate quantized execution and the
    modifier will be enabled until training is completed.

    In DDP mode, weight calibration is partitioned across ranks and activation
    observer statistics are all-reduced at sequential layer boundaries.

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

    def on_start(self, state: State, event: Event, **kwargs):
        """
        Begin calibrating activations and weights. Calibrate weights only once
        on start. In DDP mode, weight calibration is partitioned across ranks.
        """
        self.started_ = True
        QuantizationMixin.start_calibration(self, state.model)

        named_modules = list(
            match_named_modules(state.model, self.resolved_targets, self.ignore)
        )

        if is_distributed():
            self._calibrate_weights_distributed(state.model, named_modules)
        else:
            self._calibrate_weights_single(state.model, named_modules)

    def _calibrate_weights_single(self, model, named_modules):
        """Original single-process weight calibration."""
        # TODO: this step can be combined with update_weight_zp_scale
        # once update_fused_layer_weight_global_scales is removed
        # and not required by vLLM
        for _, module in named_modules:
            update_weight_global_scale(module)

        # NOTE: update_fused_layer_weight_global_scales operates on Attention
        # and MLP layers, not quantizable Linear layers. Rather than running
        # on targeted modules, we need to run on all modules.
        # Because this call is idempotent, setting all global_scales to the
        # min value, it is ok to run potentially multiple times for all modules
        for module in model.modules():
            update_fused_layer_weight_global_scales(module)

        for _, module in tqdm.tqdm(named_modules, desc="Calibrating weights"):
            update_weight_zp_scale(module)

    def _calibrate_weights_distributed(self, model, named_modules):
        """
        DDP-partitioned weight calibration. Each rank calibrates a subset of
        modules and broadcasts results to all ranks.
        """
        module_to_rank = build_module_to_rank_map(named_modules)
        my_modules = partition_modules_by_weight_size(named_modules)
        rank = get_rank()

        # compute global_scale for assigned modules only
        for _, module in tqdm.tqdm(
            my_modules, desc=f"[Rank {rank}] Updating global scales"
        ):
            update_weight_global_scale(module)

        # broadcast global_scales so all ranks can run the fuse step
        for _, module in named_modules:
            src_rank = module_to_rank[module]
            broadcast_module_parameter(module, "weight_global_scale", src_rank)

        # fuse global_scales (all ranks, idempotent)
        for module in model.modules():
            update_fused_layer_weight_global_scales(module)

        # compute scale/zp for assigned modules only
        for _, module in tqdm.tqdm(
            my_modules, desc=f"[Rank {rank}] Calibrating weights"
        ):
            update_weight_zp_scale(module)

        # broadcast scale/zp to all ranks
        for _, module in named_modules:
            src_rank = module_to_rank[module]
            broadcast_module_parameter(module, "weight_scale", src_rank)
            if hasattr(module, "weight_zero_point"):
                broadcast_module_parameter(module, "weight_zero_point", src_rank)

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        if event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            self._sync_activation_observers(state.model)

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            self._sync_activation_observers(state.model)
            if not self.ended_:
                self.on_end(state, None)

    def _sync_activation_observers(self, model):
        """
        All-reduce activation observer min/max values across DDP ranks,
        then recompute scale/zp from the global statistics.
        No-op if not distributed.
        """
        if not is_distributed():
            return

        for _, module in match_named_modules(model, self.resolved_targets, self.ignore):
            for base_name in ("input", "output", "q", "k", "v"):
                observer = getattr(module, f"{base_name}_observer", None)
                if observer is None:
                    continue

                # all-reduce accumulated min/max across ranks
                if (
                    hasattr(observer, "past_min_vals")
                    and observer.past_min_vals is not None
                ):
                    observer.past_min_vals = all_reduce_min(observer.past_min_vals)
                if (
                    hasattr(observer, "past_max_vals")
                    and observer.past_max_vals is not None
                ):
                    observer.past_max_vals = all_reduce_max(observer.past_max_vals)

                # all-reduce global min/max (TENSOR_GROUP strategy)
                if (
                    hasattr(observer, "past_global_min_vals")
                    and observer.past_global_min_vals is not None
                ):
                    observer.past_global_min_vals = all_reduce_min(
                        observer.past_global_min_vals
                    )
                if (
                    hasattr(observer, "past_global_max_vals")
                    and observer.past_global_max_vals is not None
                ):
                    observer.past_global_max_vals = all_reduce_max(
                        observer.past_global_max_vals
                    )

                recompute_qparams_from_observer(module, base_name)

    def on_end(self, state: State, event: Event, **kwargs):
        """
        Finish calibrating by removing observers and calibration hooks
        """
        self.ended_ = True
        QuantizationMixin.end_calibration(
            self, state.model
        )  # keep quantization enabled

    def on_finalize(self, state: State, **kwargs) -> bool:
        if not self.ended_:
            self.on_end(state, None)
