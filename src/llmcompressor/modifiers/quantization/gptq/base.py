import contextlib
from typing import Dict, List, Optional, Tuple, Union

import torch
from compressed_tensors.offload.dist_utils import as_broadcastable, is_distributed
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStrategy,
)
from compressed_tensors.quantization.quant_args import ActivationOrdering
from compressed_tensors.utils import (
    align_module_device,
    get_execution_device,
    getattr_chain,
    match_named_modules,
    update_offload_parameter,
)
from loguru import logger
from pydantic import PrivateAttr
from torch import distributed as dist

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.calibration import update_weight_global_scale
from llmcompressor.modifiers.quantization.gptq.gptq_quantize import (
    accumulate_hessian,
    make_empty_hessian,
    quantize_weight,
)
from llmcompressor.modifiers.quantization.quantization import QuantizationMixin
from llmcompressor.modifiers.utils import update_fused_layer_weight_global_scales
from llmcompressor.sentinel import Sentinel
from llmcompressor.utils import greedy_bin_packing, wait_for_comms
from llmcompressor.utils.metric_logging import CompressionLogger

__all__ = ["GPTQModifier"]

_GPTQ_Q_PARAMS = ["weight", "weight_scale", "weight_zero_point", "weight_g_idx"]


class GPTQModifier(Modifier, QuantizationMixin):
    """
    Implements the GPTQ algorithm from https://arxiv.org/abs/2210.17323. This modifier
    uses activations to calibrate a hessian matrix, which is then used to determine
    optimal quantization values and orderings for the model weights.

    Sample yaml:

    ```yaml
    test_stage:
      obcq_modifiers:
        GPTQModifier:
          block_size: 128
          dampening_frac: 0.001
          offload_hessians: False
          actorder: static
          config_groups:
            group_0:
              targets:
                - "Linear"
              input_activations: null
              output_activations: null
              weights:
                num_bits: 8
                type: "int"
                symmetric: true
                strategy: group
                group_size: 128
    ```

    Lifecycle:

    - on_initialize
        - apply config to model
    - on_start
        - add activation calibration hooks
        - add gptq weight calibration hooks
    - on_sequential_epoch_end
        - quantize_weight
    - on_finalize
        - remove_hooks()
        - model.apply(freeze_module_quantization)

    :param sequential_targets: list of layer names to compress during GPTQ, or
        '__ALL__' to compress every layer in the model
    :param block_size: Used to determine number of columns to compress in one pass
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    :param actorder: order in which weight columns are quantized. Defaults to "static"
        activation ordering, which achieves best accuracy recovery with no runtime cost.
        For more information, see https://github.com/vllm-project/vllm/pull/8135
    :param offload_hessians: Set to True for decreased memory usage but increased
        runtime.

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

    # gptq modifier arguments
    sequential_targets: Union[str, List[str], None] = None
    block_size: int = 128
    dampening_frac: Optional[float] = 0.01
    # TODO: this does not serialize / will be incorrectly written
    actorder: Optional[Union[ActivationOrdering, Sentinel]] = Sentinel("static")
    offload_hessians: bool = False

    # private variables
    _module_names: Dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _hessians: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)
    _num_samples: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(
        default_factory=dict
    )

    def resolve_quantization_config(self) -> QuantizationConfig:
        config = super().resolve_quantization_config()

        def resolve_actorder(existing):
            # sentinel default only overrides if existing is None
            if self.actorder == Sentinel("static"):
                return ActivationOrdering.STATIC if existing is None else existing

            # user-provided value always attempts to override
            if existing is None or self.actorder == existing:
                return self.actorder

            # if existing provided and conflicts
            raise ValueError(
                "Cannot resolve activation ordering when both "
                "`GPTQModifier.actorder` and `QuantizationScheme.actorder` "
                f"are provided and differ ({self.actorder}, {existing}). "
                "Either unset `GPTQModifier.actorder` or "
                "remove `actorder` from config groups."
            )

        for scheme in config.config_groups.values():
            assert isinstance(scheme, QuantizationScheme)
            if (
                getattr_chain(scheme, "weights.strategy", None)
                == QuantizationStrategy.GROUP
            ):
                scheme.weights.actorder = resolve_actorder(scheme.weights.actorder)
        return config

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize and run the GPTQ algorithm on the current state

        :param state: session state storing input model and calibration data
        """
        # apply config to model and prepare calibration hooks
        if QuantizationMixin.has_config(self):
            QuantizationMixin.initialize_quantization(self, state.model)

        # prepare module names
        self._module_names = {
            m: name
            for name, m in match_named_modules(
                state.model, self.resolved_targets, self.ignore
            )
        }

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

        # register quantization calibration hooks
        # assume quantization has been initialized by this modifier or one before it
        QuantizationMixin.start_calibration(self, state.model)

        # register gptq hooks
        added_hook = False

        named_modules = list(
            match_named_modules(state.model, self.resolved_targets, self.ignore)
        )

        for _, module in named_modules:
            if getattr_chain(module, "quantization_scheme.weights", None) is not None:
                # HACK: previously, embeddings were not quantized because they were not
                # accessible by the layer compressor. For now, we manually ignore it,
                # but in the FUTURE this should be ignored by the user
                if not isinstance(module, torch.nn.Embedding):
                    self.register_hook(module, self.calibrate_module, "forward")
                    added_hook = True

        # Optionally generate global scales if using TENSOR_GROUP quantization
        for _, module in named_modules:
            update_weight_global_scale(module)

        for module in state.model.modules():
            update_fused_layer_weight_global_scales(module)

        if not added_hook:
            raise ValueError(
                "GPTQModifier requires a weight quantization config be specified by "
                "this modifier or a modifier preceding it"
            )

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        if event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            self.compress_modules()

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            self.compress_modules()

            if not self.ended_:
                self.on_end(state, None)

    def calibrate_module(
        self,
        module: torch.nn.Module,
        args: Tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ):
        """
        Calibration hook used to accumulate the hessian of the input to the module

        :param module: module being calibrated
        :param args: inputs to the module, the first element of which is the
            canonical input
        :param _output: uncompressed module output, unused
        """
        # Assume that first argument is the input
        inp = args[0]

        # Initialize hessian if not present
        if module not in self._num_samples:
            init_device = (
                "cpu" if self.offload_hessians else get_execution_device(module)
            )
            self._hessians[module] = make_empty_hessian(module, device=init_device)
            self._num_samples[module] = torch.zeros(
                tuple(), device=get_execution_device(module)
            )

        # Accumulate hessian with input with optional offloading
        with self._maybe_onload_hessian(module):
            self._hessians[module], self._num_samples[module] = accumulate_hessian(
                inp,
                module,
                self._hessians[module],
                self._num_samples[module],
            )

    def compress_modules(self):
        """
        Quantize modules which have been calibrated
        """
        ### Not Distributed
        if not is_distributed():
            self.compress_module_list(list(self._num_samples.keys()))
            return

        ### Distributed
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Assign modules to ranks
        module_list, rank_to_modules, module_to_rank = greedy_bin_packing(
            list(self._hessians.keys()),
            world_size,
            item_weight_fn=lambda mod: self._hessians[mod].shape[0],
        )

        # send hessians to assigned ranks
        self._reduce_hessian_to_target_rank(module_list, module_to_rank)

        self.compress_module_list(rank_to_modules[rank])

        # broadcast compressed modules to each rank
        self._broadcast_quantized_params(module_list, module_to_rank)

    def compress_module_list(self, module_list):
        for module in module_list:
            name = self._module_names[module]
            num_samples = self._num_samples[module]
            quant_args = getattr_chain(module, "quantization_scheme.weights")

            logger.info(f"Quantizing {name} using {num_samples} samples")
            with (
                torch.no_grad(),
                align_module_device(module),
                self._maybe_onload_hessian(module),
                CompressionLogger(module) as comp_logger,
            ):
                loss, q_param_dict = quantize_weight(
                    module=module,
                    quant_args=quant_args,
                    hessian=self._hessians.pop(module) / self._num_samples.pop(module),
                    blocksize=self.block_size,
                    percdamp=self.dampening_frac,
                )
                comp_logger.set_loss(loss)

            for attr, val in q_param_dict.items():
                update_offload_parameter(module, attr, val)

    def _reduce_hessian_to_target_rank(self, module_list, module_to_rank):
        rank = dist.get_rank()
        pending_comms = []
        for module in module_list:
            target_rank = module_to_rank[module]
            with self._maybe_onload_hessian(module):
                pending_comms.append(
                    dist.reduce(
                        self._hessians[module],
                        op=dist.ReduceOp.SUM,
                        dst=target_rank,
                        async_op=True,
                    )
                )
                pending_comms.append(
                    dist.reduce(
                        self._num_samples[module],
                        op=dist.ReduceOp.SUM,
                        dst=target_rank,
                        async_op=True,
                    )
                )
                if rank != target_rank:
                    self._hessians.pop(module, None)
                    self._num_samples.pop(module, None)
        wait_for_comms(pending_comms)

    def _broadcast_quantized_params(self, module_list, module_to_rank):
        pending_comms = []
        for module in module_list:
            src_rank = module_to_rank[module]

            # Get parameters from module
            for attr in _GPTQ_Q_PARAMS:
                if getattr(module, attr, None) is not None:
                    pending_comms.append(
                        dist.broadcast(
                            as_broadcastable(getattr(module, attr)),
                            src=src_rank,
                            async_op=True,
                        )
                    )
        wait_for_comms(pending_comms)

    def on_end(self, state: State, event: Event, **kwargs):
        """
        Finish calibrating by removing observers and calibration hooks
        """
        self.ended_ = True
        QuantizationMixin.end_calibration(self, state.model)
        self.remove_hooks()  # remove gptq hooks

    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        disable the quantization observers used by the OBCQ algorithm

        :param state: session state storing input model and calibration data
        """
        if not self.ended_:
            self.on_end(state, None)

        if len(self._num_samples) > 0:
            raise ValueError(f"Failed to compress {len(self._num_samples)} modules")

        self._hessians = dict()
        self._num_samples = dict()

        return True

    @contextlib.contextmanager
    def _maybe_onload_hessian(self, module: torch.nn.Module):
        if self.offload_hessians:
            device = get_execution_device(module)
            self._hessians[module] = self._hessians[module].to(device=device)

        yield

        if self.offload_hessians:
            if module in self._hessians:  # may have been deleted in context
                self._hessians[module] = self._hessians[module].to(device="cpu")
