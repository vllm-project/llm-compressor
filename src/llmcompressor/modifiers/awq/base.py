import inspect
from typing import Literal

import torch
from compressed_tensors.quantization import (
    QuantizationStrategy,
    disable_quantization,
)
from compressed_tensors.utils import (
    align_modules,
    getattr_chain,
    match_modules_set,
    match_named_modules,
)
from loguru import logger
from pydantic import ConfigDict, PrivateAttr, field_validator
from torch.nn import Module
from tqdm import tqdm

from llmcompressor.core import Event, EventType, State, active_session
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.awq.mappings import (
    AWQMapping,
    ResolvedMapping,
    get_layer_mappings_from_architecture,
)
from llmcompressor.modifiers.awq.helpers import (
    accumulate_mean,
    apply_scale_to_module,
    compute_scale_losses,
    extract_masked_activations,
    flatten_balance_layers,
    get_lowest_common_ancestor_with_avoid,
    get_mapping_skip_reason,
    resolve_activation_hook_target,
    select_best_scales_from_losses,
    should_skip_smoothing_for_outputs,
    validate_and_get_smooth_layer,
)
from llmcompressor.modifiers.quantization.calibration import (
    update_weight_global_scale,
    update_weight_zp_scale,
)
from llmcompressor.modifiers.quantization.quantization import QuantizationMixin
from llmcompressor.modifiers.utils import update_fused_layer_weight_global_scales
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.modifiers.utils.pytorch_helpers import is_moe_model
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.sentinel import Sentinel
from llmcompressor.utils.helpers import calibration_forward_context
from llmcompressor.utils.pytorch.module import (
    get_module_to_name_dict,
)

__all__ = ["AWQModifier"]


class AWQModifier(Modifier, QuantizationMixin):
    """
    Implements the AWQ (Activation-Weighted Quantization) algorithm,
    as described in https://arxiv.org/pdf/2306.00978. The algorithm
    significantly reduces quantization error by protecting only 1%
    of the most salient weight channels.

    Instead of relying on raw weight values, AWQ identifies important channels by
    analyzing activation patterns, focusing on the channels in the weight tensor that
    are most responsive to the input. To reduce quantization error, it scales these
    channels in a way that preserves the model's original behavior, using scaling
    factors computed offline from activation statistics.

    Because this modifier manipulates the weights of the model, it can only be used in
    in one-shot and not during training. Activation ranges are determined by running a
    small set of calibration data through the model.

    example recipe:
    ```yaml
    AWQModifier:
      mappings:
        - smooth_layer: "re:.*self_attn_layer_norm"
          balance_layers: ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"]
        - smooth_layer: "re:.*final_layer_norm"
          balance_layers: ["re:.*fc1"]
        # activation_hook_target specifies which submodule of the parent to hook
        # for activation caching.
        # This change is only useful for MoE models with parallel transformer blocks,
        # and one should use the default value (None) in most cases.
      ignore: ["lm_head"]
      config_groups:
        group_0:
          targets:
            - "Linear"
          input_activations: null
          output_activations: null
          weights:
            num_bits: 4
            type: int
            symmetric: false
            strategy: group
            group_size: 128
    ```

    Lifecycle:

    - on_initialize
        - resolve mappings
        - capture kwargs needed for forward passes into modules
    - on_start
        - set up activation cache hooks to capture input activations
            to balance layers
    - on sequential epoch end
        - apply smoothing to each smoothing layer
            - consume cached activations across all batches
                - clear cached activations as they are used
            - find best smoothing scale for each smoothing layer via grid search
            - apply best scales to model weights
            - raise error if any unused activations remain
    - on_end
        - re-run logic of sequential epoch end (in case of basic pipeline)
        - set scales and zero points
        - remove activation hooks
    - on_finalize
        - clear resolved mappings and captured activations

    :param sequential_targets: list of module names to compress in
        the same calibration pass
    :param mappings: list activation layers to smooth, and which layers to
        scale the output such that activations are smoothed.
        Each entry of the mapping list should be a list itself, in which the first
        entry is a list of layers who share the same input activation (the one to be
        to smoothed) and the second entry is the layer whose output is scaled to
        achieve the smoothing.
        If regex is used, it matches layers with the largest overlap in module name.
        Each mapping may also include an ``activation_hook_target``: a dotted
        attribute path relative to the parent module (lowest common ancestor)
        specifying which submodule to hook for activation caching. This is useful
        for parallel transformer blocks where the default (hooking
        ``balance_layers[0]``) would capture the wrong activations.
    :param ignore: list of layers to ignore during quantization (not smoothed).
        It should match the name of layers whose outputs are scaled to achieve
        smoothing (the second entry of the mappings list).
    :param offload_device: offload cached args to this device, which reduces memory
        requirements but requires more time to move data between cpu and execution
        device. Defaults to None, so cached args are not offloaded. Consider setting
        to torch.device("cpu") if you are encountering OOM errors
    :param duo_scaling: whether to use duo scaling, which uses both input activations
        and weights to determine the scaling factor. Defaults to True
        If True, both activations and weights are used.
        If False, only activations are used.
        If "both", half the grid search is performed with duo_scaling=False and the
        other half is performed with duo_scaling=True.
    :param n_grid: when performing the best scales grid search for each mapping,
        this specifies how many grid points should be used. To decrease the runtime,
        at the possible cost of slightly worse scales, this can be decreased.
        Defaults to 20
    """

    # Allow arbitrary types because AWQMapping has fields of type torch.nn.Module
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    # User-provided vars (in addition to QuantizationMixin args)
    sequential_targets: str | list[str] | None = None
    mappings: list[AWQMapping] | None = None
    offload_device: torch.device | None | Sentinel = Sentinel("not_provided")
    duo_scaling: bool | Literal["both"] = True
    n_grid: int = 20

    # Private vars set during initialization, cleared during finalization
    _resolved_mappings: list[ResolvedMapping] = PrivateAttr(default_factory=list)
    # Cache list of forward input args for each parent module, one dict for each batch
    _parent_args_cache: dict[Module, IntermediatesCache] = PrivateAttr(
        default_factory=dict
    )
    # Dict[smooth layer name, (activation means, activation counts)]
    _smooth_activation_means: dict[str, tuple[torch.FloatTensor, int]] = PrivateAttr(
        default_factory=dict
    )
    # List to store error metrics for each layer
    _error_metrics: list[dict] = PrivateAttr(default_factory=list)

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize AWQ on the given state
        Initialize quantization, resolve mappings, cache module kwargs

        :param state: state to run AWQ on
        :return: True on a successful run, False otherwise
        """

        # apply config to model and prepare calibration hooks
        if QuantizationMixin.has_config(self):
            QuantizationMixin.initialize_quantization(self, state.model)

        # Validate that duo_scaling is only used with per-channel quantization
        if self.duo_scaling is not False:
            for _, module in match_named_modules(
                state.model, self.resolved_targets, self.ignore
            ):
                if (
                    hasattr(module, "quantization_scheme")
                    and hasattr(module.quantization_scheme, "weights")
                    and module.quantization_scheme.weights.strategy
                    == QuantizationStrategy.TENSOR
                ):
                    raise ValueError(
                        "duo_scaling is only supported with per-channel quantization "
                        "strategies (group or channel), but found TENSOR strategy. "
                        "Please set duo_scaling=False or use a per-channel "
                        "quantization strategy."
                    )

        if self.mappings is None:
            logger.info("No AWQModifier.mappings provided, inferring from model...")
            self.mappings = get_layer_mappings_from_architecture(
                architecture=state.model.__class__.__name__
            )

        # Set default offload_device
        if self.offload_device == Sentinel("not_provided"):
            # Check if we have a MoE model
            if is_moe_model(state.model):
                self.offload_device = torch.device("cpu")
                logger.info(
                    "MoE model detected: setting offload_device to 'cpu' by default "
                    "to reduce memory usage. You can override this by explicitly "
                    "setting offload_device in your recipe."
                )
            else:
                # For non-MoE models, convert sentinel to None
                # (no offloading by default)
                self.offload_device = None

        self._set_resolved_mappings(state.model)

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

        # Check for unsupported token masking with MoE up_proj -> down_proj mappings
        if state.loss_masks is not None and self._has_moe_up_down_proj_mapping():
            raise ValueError(
                "Token masking (use_loss_mask=True) is not supported with "
                "up_proj -> down_proj mappings in MoE models. The MoE routing "
                "mechanism dispatches tokens to different experts, and the loss mask "
                "cannot be properly aligned with this dispatch. Please either "
                "disable token masking or exclude the up_proj -> down_proj mapping "
                "for MoE layers from the AWQ configuration."
            )

        # register quantization calibration hooks
        # assume quantization has been initialized by this modifier or one before it
        QuantizationMixin.start_calibration(self, state.model)
        # AWQ performs forward passes during _apply_smoothing
        # before any scales or zero points are updated
        # Quantization must be disabled, otherwise NaNs will
        # appear in quantized forward method
        state.model.apply(disable_quantization)

        self._setup_activation_cache_hooks()

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        elif event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            # Run smoothing in case of sequential pipeline
            self._apply_smoothing(state.model)

        elif event.type_ == EventType.CALIBRATION_EPOCH_END:
            # Run smoothing in case of basic pipeline
            self._apply_smoothing(state.model)

            if not self.ended_:
                self.on_end(state, None)

    def on_end(self, state: State, event: Event, **kwargs):
        """
        Finish calibrating by setting scales and zero-points,
         removing observers and calibration hooks
        """
        self._assert_all_activations_consumed()

        self.ended_ = True

        named_modules = list(
            match_named_modules(state.model, self.resolved_targets, self.ignore)
        )

        # For TENSOR_GROUP (nvfp4), calculate global scales after smoothing
        for _, module in tqdm(named_modules, desc="Updating global scales"):
            update_weight_global_scale(module)

        # For TENSOR_GROUP (nvfp4), fuse global scales for attention and MLP layers
        # This is a requirement for vLLM inference.
        for module in tqdm(state.model.modules(), desc="Fusing global scales"):
            update_fused_layer_weight_global_scales(module)

        # Calculate scales and zero points using the fused global scales
        for _, module in tqdm(named_modules, desc="Calibrating weights"):
            update_weight_zp_scale(module)

        QuantizationMixin.end_calibration(self, state.model)

        # remove activation hooks
        self.remove_hooks()

    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        Clean up by clearing the activations and mapping data

        :param state: unused
        :return: True
        """
        if not self.ended_:
            self.on_end(state, None)

        self._log_error_metrics()

        self._parent_args_cache.clear()
        self._smooth_activation_means.clear()
        self._resolved_mappings.clear()
        self._error_metrics.clear()

        return True

    def _set_resolved_mappings(self, model: Module) -> None:
        """
        Transforms the list of activations to smooth and their corresponding weights
        into ResolvedMapping objects, resolving regular expressions.
        Result is stored in _resolved_mappings.

        For each activation in the mapping list, we find the corresponding weight to
        balance by searching for the longest substring. For instance, if our balance
        weight is ".*re:.*q_proj" and the activation is "re:.*self_attn_layer_norm" we
        would match model.layer.0.p_proj to model.layer.0.self_attn_layer_norm and
        repeat for model.layer.1 and so on
        """
        module_to_name = get_module_to_name_dict(model)
        targeted_names = self._get_targeted_names(model)
        resolved_mappings: list[ResolvedMapping] = []

        for mapping in self.mappings:
            resolved_mappings.extend(
                self._resolve_mapping_instances(
                    model=model,
                    mapping=mapping,
                    module_to_name=module_to_name,
                    targeted_names=targeted_names,
                )
            )

        self._resolved_mappings = resolved_mappings

    def _get_targeted_names(self, model: Module) -> set[str]:
        # Get names of modules targeted for quantization (excludes ignored).
        return {
            name
            for name, _ in match_named_modules(model, self.resolved_targets, self.ignore)
        }

    def _resolve_mapping_instances(
        self,
        model: Module,
        mapping: AWQMapping,
        module_to_name: dict[Module, str],
        targeted_names: set[str],
    ) -> list[ResolvedMapping]:
        resolved_mappings: list[ResolvedMapping] = []

        # We deliberately don't use the ignore list when matching mappings so that we
        # can handle layers that need smoothing but not quantization. We only skip if
        # no layers in the mapping are targeted for quantization.
        for smooth_layers, *nested_balance_layers in match_modules_set(
            model, (mapping.smooth_layer, *mapping.balance_layers)
        ):
            smooth_layer, smooth_name = validate_and_get_smooth_layer(
                smooth_layers=smooth_layers,
                module_to_name=module_to_name,
                mapping=mapping,
            )
            balance_layers, balance_names = flatten_balance_layers(
                nested_balance_layers, module_to_name
            )
            skip_message = get_mapping_skip_reason(
                smooth_layer=smooth_layer,
                smooth_name=smooth_name,
                balance_layers=balance_layers,
                balance_names=balance_names,
                targeted_names=targeted_names,
            )
            if skip_message:
                logger.warning(
                    f"skipping AWQ for {smooth_name} for mapping {mapping}{skip_message}"
                )
                continue

            ancestor_name, ancestor = get_lowest_common_ancestor_with_avoid(
                balance_names, model, torch.nn.ModuleList
            )
            activation_hook_target = resolve_activation_hook_target(
                mapping=mapping,
                ancestor=ancestor,
                ancestor_name=ancestor_name,
            )

            resolved_mappings.append(
                ResolvedMapping(
                    smooth_name,
                    smooth_layer,
                    balance_layers,
                    balance_names=balance_names,
                    parent=ancestor,
                    parent_name=ancestor_name,
                    activation_hook_target=activation_hook_target,
                )
            )

        return resolved_mappings

    def _setup_activation_cache_hooks(self) -> None:
        """
        Attach a forward hook to each activation we want to smooth. This allows us to
        calculate the dynamic range during calibration
        """

        for mapping in self._resolved_mappings:
            self._register_parent_cache_hook(mapping.parent)
            self._register_smooth_activation_hook(mapping)

    def _register_parent_cache_hook(self, parent: Module) -> None:
        # Parent kwargs needed for future forward passes. The same parent may appear
        # multiple times in resolved mappings, so only register once.
        if parent in self._parent_args_cache:
            return
        self._parent_args_cache[parent] = IntermediatesCache(None, self.offload_device)
        self.register_hook(
            parent,
            self._cache_parent_kwargs_hook,
            "forward_pre",
            with_kwargs=True,
        )

    def _register_smooth_activation_hook(self, mapping: ResolvedMapping) -> None:
        # Input activations to balance layers are needed for loss. Storing inputs to
        # the first balance layer is usually sufficient, unless a custom hook target
        # is configured for parallel transformer blocks.
        layer_to_hook = mapping.activation_hook_target or mapping.balance_layers[0]
        self.register_hook(
            layer_to_hook,
            self._create_cache_smooth_activations_hook_fn(mapping.smooth_name),
            "forward",
        )

    def _cache_parent_kwargs_hook(
        self,
        module: Module,
        args: tuple[torch.Tensor, ...],
        kwargs,
    ) -> None:
        values = inspect.signature(module.forward).bind(*args, **kwargs)
        self._parent_args_cache[module].append(values.arguments)

    def _create_cache_smooth_activations_hook_fn(self, smooth_name: str):
        def cache_smooth_activations_hook(
            _module: Module,
            args: tuple[torch.Tensor, ...],
            _output: torch.Tensor,
        ):
            masked_activations = extract_masked_activations(args[0].abs().detach())
            act_mean, count = accumulate_mean(
                masked_activations,
                self._smooth_activation_means.get(smooth_name, None),
            )
            self._smooth_activation_means[smooth_name] = (act_mean.cpu(), count)

        return cache_smooth_activations_hook

    @torch.no_grad()
    def _apply_smoothing(self, model: Module) -> None:
        """
        Calculate the best scaling factors for each layer to smooth activations and
        apply the scaling factors to the weights of the next layer to offset the
        smoothing

        :param model: model to apply smoothing to
        """
        # NOTE: When using SequentialPipeline, not all the mappings
        # will have cached activations in the segment being updated
        mappings_to_smooth = self._get_mappings_with_cached_activations()
        for mapping in tqdm(mappings_to_smooth, desc="Smoothing"):
            self._apply_smoothing_to_mapping(model, mapping)

        for v in self._parent_args_cache.values():
            v.batch_intermediates.clear()
        self._assert_all_activations_consumed()

    def _get_mappings_with_cached_activations(self) -> list[ResolvedMapping]:
        # When using SequentialPipeline, not all mappings will have cached
        # activations in the segment being updated.
        return [
            mapping
            for mapping in self._resolved_mappings
            if mapping.smooth_name in self._smooth_activation_means
        ]

    @torch.no_grad()
    def _apply_smoothing_to_mapping(
        self,
        model: Module,
        mapping: ResolvedMapping,
    ) -> None:
        with (
            align_modules([mapping.parent, mapping.smooth_layer, *mapping.balance_layers]),
            calibration_forward_context(model),
            HooksMixin.disable_hooks(),
        ):
            fp16_outputs = self._run_samples(mapping.parent)
            if should_skip_smoothing_for_outputs(mapping, fp16_outputs):
                del self._smooth_activation_means[mapping.smooth_name]
                return

            orig_layer_weights = {
                balance_layer: balance_layer.weight.clone()
                for balance_layer in mapping.balance_layers
            }
            best_scales = self._compute_best_scale(mapping, fp16_outputs, orig_layer_weights)
            self._apply_scales_to_mapping(mapping, best_scales, orig_layer_weights)

            # Remove caches needed to smooth this mapping.
            del self._smooth_activation_means[mapping.smooth_name]
            del orig_layer_weights

    @torch.no_grad()
    def _apply_scales_to_mapping(
        self,
        mapping: ResolvedMapping,
        best_scales: torch.Tensor,
        orig_layer_weights: dict[Module, torch.Tensor],
    ) -> None:
        for layer in mapping.balance_layers:
            apply_scale_to_module(
                module=layer,
                scales=best_scales,
                balance_layers=mapping.balance_layers,
                smooth_layer=mapping.smooth_layer,
                orig_layer_weights=orig_layer_weights,
            )
        apply_scale_to_module(
            module=mapping.smooth_layer,
            scales=best_scales,
            balance_layers=mapping.balance_layers,
            smooth_layer=mapping.smooth_layer,
            orig_layer_weights=orig_layer_weights,
        )

    @torch.no_grad()
    def _run_samples(self, module: Module) -> list[torch.Tensor]:
        outputs = [
            module(**batch_kwargs) for batch_kwargs in self._parent_args_cache[module]
        ]
        return [
            # If tuple, assume that first argument is the input
            output[0] if isinstance(output, tuple) else output
            for output in outputs
        ]

    def _compute_best_scale(
        self,
        mapping: ResolvedMapping,
        fp16_outputs: list[torch.Tensor],
        orig_layer_weights: dict[torch.nn.Module, torch.Tensor],
    ) -> torch.Tensor:
        """
        Select best scales for a given mapping in a grid search
        Best scales are those that minimize MSE loss of quantized weight
            outputs compared to fp16_outputs

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | _pseudo_quantize_tensor(W * s)
        X: inputs from calib dataset    | X
        W: original weights in FP16     | layer
        s: per channel scaling factor   | s^-1 * X

        :param mapping: best scales will be found for the ResolvedMapping.
        :param fp16_outputs: output of mapping.parent in unquantized case,
            one tensor for each batch.
        :return: tensor of best scales, one for each channel
        """
        loss_history = compute_scale_losses(
            mapping=mapping,
            fp16_outputs=fp16_outputs,
            orig_layer_weights=orig_layer_weights,
            get_grid_search_means=self._get_grid_search_means,
            get_grid_configuration=self._get_grid_configuration,
            run_samples=self._run_samples,
            compute_loss=self._compute_loss,
        )
        best_scales, best_ratio, best_error, initial_error = (
            select_best_scales_from_losses(loss_history)
        )

        err_reduction = best_error / initial_error if initial_error > 0 else 1.0
        logger.debug(
            f"AWQ grid search for {mapping.smooth_name}: "
            f"initial error = {initial_error:.3e}, "
            f"best error = {best_error:.3e}, "
            f"error reduction rate (best/initial) = {err_reduction * 100:.3f}%"
        )

        # Store error metrics for this layer
        self._error_metrics.append(
            {
                "layer_name": mapping.smooth_name,
                "parent_name": mapping.parent_name,
                "initial_error": initial_error,
                "best_error": best_error,
                "reduction": err_reduction,
            }
        )

        assert (
            torch.isnan(best_scales).sum() == 0
        ), f"Nan found in scales: {best_scales}"

        return best_scales.detach().cpu()

    def _get_grid_search_means(
        self, mapping: ResolvedMapping, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x_mean = self._smooth_activation_means[mapping.smooth_name][0].to(device)
        w_mean = None
        if self.duo_scaling:
            w_mean = self._compute_layer_means(mapping.balance_layers).to(device)
        return x_mean, w_mean

    def _get_grid_configuration(self) -> tuple[int, list[bool]]:
        # If duo_scaling is "both", perform half the grid search with duo scaling
        # disabled and half with duo scaling enabled.
        if self.duo_scaling == "both":
            return int(self.n_grid / 2), [False, True]
        return self.n_grid, [self.duo_scaling]

    @torch.no_grad()
    def _compute_loss(
        self,
        fp16_outputs: list[torch.Tensor],
        int_w_outputs: list[torch.Tensor],
    ) -> float:
        session = active_session()
        loss_masks = session.state.loss_masks if session.state else None

        loss = 0.0
        num_elements = 0

        # Compute the MSE loss for each batch
        for batch_idx, (fp16_batch, int_w_batch) in enumerate(
            zip(fp16_outputs, int_w_outputs)
        ):
            loss_mask = loss_masks[batch_idx] if loss_masks else None

            if loss_mask is not None:
                token_mask = loss_mask.to(fp16_batch.device) == 1  # (batch, seq)
                fp16_masked = fp16_batch[token_mask]  # (num_masked_tokens, hidden)
                int_w_masked = int_w_batch.to(fp16_batch.device)[token_mask]
                loss += torch.nn.functional.mse_loss(
                    fp16_masked, int_w_masked, reduction="sum"
                )
                num_elements += fp16_masked.numel()
            else:
                loss += torch.nn.functional.mse_loss(
                    fp16_batch, int_w_batch.to(fp16_batch.device), reduction="sum"
                )
                num_elements += fp16_batch.numel()

        # Normalize the loss by the total number of elements
        return (loss / num_elements).item()

    def _log_error_metrics(self):
        """
        Log the error metrics (initial error, best error, reduction).
        """

        # Prepare data for saving
        metrics_data = {
            "quantization_config": {
                "duo_scaling": self.duo_scaling,
                "n_grid": self.n_grid,
            },
            "total_layers": len(self._error_metrics),
            "metrics": self._error_metrics,
        }

        # Save to disk
        logger.debug(f"AWQ per-mapping error metrics: {metrics_data}")

        # Also print summary statistics
        reductions = [m["reduction"] for m in self._error_metrics]
        avg_reduction = sum(reductions) / len(reductions)
        min_reduction = min(reductions)
        max_reduction = max(reductions)
        sorted_reductions = sorted(reductions)
        median_reduction = sorted_reductions[len(sorted_reductions) // 2]
        logger.debug(
            f"Error reduction statistics: "
            f"avg={avg_reduction:.4f}, median={median_reduction:.4f}, "
            f"min={min_reduction:.4f}, max={max_reduction:.4f}"
        )

    def _assert_all_activations_consumed(self):
        """
        Confirm all activations have been consumed
        If not, something has gone wrong
        """
        if len(self._smooth_activation_means) != 0:
            raise RuntimeError("Some cached activations were not used")

    def _has_moe_up_down_proj_mapping(self) -> bool:
        """
        Check if any resolved mapping is an up_proj -> down_proj mapping
        where the balance layers are MoE experts (indicated by '.experts.'
        in the name).

        Token masking is not supported for such mappings because the MoE
        routing mechanism dispatches tokens to different experts, and the
        loss mask cannot be properly aligned with this dispatch.
        """
        for mapping in self._resolved_mappings:
            # Check if this is an up_proj -> down_proj mapping
            if mapping.smooth_name.endswith("up_proj"):
                for balance_name in mapping.balance_names:
                    if (
                        balance_name.endswith("down_proj")
                        and ".experts." in balance_name
                    ):
                        return True
        return False

    @staticmethod
    def _compute_layer_means(layers: list[Module]) -> torch.Tensor:
        """
        Compute per-channel/group/block/tensor mean of normalised weights
        for all passed in layers taking into account the quantization_scheme.

        To minimize memory requirements, layers are reduced to a running total
            of sums and counts when calculating mean
        """
        # to calculate mean without having to carry full population
        weight_total_count = 0
        weight_total_sum = 0

        for layer in layers:
            if not hasattr(layer, "weight"):
                logger.warning(
                    "Unable to find weight param for targeted"
                    f" layer {type(layer)}, skipping"
                )
                continue
            weight = layer.weight.clone()
            orig_shape = weight.shape

            q_args = getattr_chain(layer, "quantization_scheme.weights", None)
            if not q_args:
                logger.warning(
                    "Unable to find quantization scheme for "
                    f"targeted layer {type(layer)}, skipping"
                )
                continue

            match q_args.strategy:
                # chunk size is the size of the size of the
                # set of elements that get quantized together
                case QuantizationStrategy.TENSOR:
                    chunk_size = weight.numel()
                case QuantizationStrategy.CHANNEL:
                    chunk_size = weight.size(1)
                case QuantizationStrategy.GROUP | QuantizationStrategy.TENSOR_GROUP:
                    chunk_size = q_args.group_size
                case QuantizationStrategy.BLOCK:
                    block_height, block_width = q_args.block_structure
                    weight = (  # (row, col) = (num_H*block_H, num_W*block_W)
                        weight.unflatten(0, (-1, block_height))
                        .unflatten(-1, (-1, block_width))
                        .transpose(1, 2)  # ↳ (num_H, num_W, block_H, block_W)
                    )
                    intermediate_shape = weight.shape
                    chunk_size = block_height * block_width

            # need to get to shape (num_chunks x chunk_size)
            weight = weight.reshape(-1, chunk_size)
            # normalize
            weight.abs_()
            weight.div_(weight.amax(dim=1, keepdim=True) + 1e-6)
            # Reshape back to original dimensions
            if q_args.strategy == QuantizationStrategy.BLOCK:
                weight = weight.view(intermediate_shape).transpose(1, 2)

            # back to (rows, cols)
            weight = weight.reshape(orig_shape)
            # Gets the average rescaled magnitude for each output channel
            weight_total_count += weight.size(0)
            weight_sum = weight.sum(0, dtype=torch.float64)
            weight_total_sum += weight_sum

        return weight_total_sum / weight_total_count

    @field_validator("duo_scaling")
    @classmethod
    def validate_duo_scaling(cls, v):
        """Validate that duo_scaling is either True, False, or 'both' (lowercase)"""
        if v not in (True, False, "both"):
            raise ValueError(f"duo_scaling must be True, False, or 'both', got {v!r}")
        return v
