import inspect
from typing import Dict, List, Optional, Tuple, Union

import torch
from compressed_tensors.quantization import disable_quantization
from compressed_tensors.utils import (
    align_modules,
    get_execution_device,
    match_named_modules,
    update_offload_parameter,
)
from loguru import logger
from pydantic import ConfigDict, PrivateAttr, model_validator
from torch.nn import Module
from tqdm import tqdm

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.awq.mappings import (
    AWQMapping,
    ResolvedMapping,
    get_layer_mappings_from_architecture,
)
from llmcompressor.modifiers.quantization.calibration import update_weight_zp_scale
from llmcompressor.modifiers.quantization.quantization import QuantizationMixin
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.utils.fsdp.helpers import get_fsdp_parent
from llmcompressor.utils.helpers import calibration_forward_context
from llmcompressor.utils.pytorch.module import get_layer_by_name

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
      ]
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
                - find best smoothing scale for each smoothing layer
                - apply to model weights
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
    :param ignore: list of layers to ignore, even if they match a regex in mappings.
        It should match the name of layers whose outputs are scaled to achieve
        smoothing (the second entry of the mappings list).
    :param offload_device: offload cached args to this device, which reduces memory
        requirements but requires more time to move data between cpu and execution
        device. Defaults to None, so cached args are not offloaded. Consider setting
        to torch.device("cpu") if you are encountering OOM errors
    :param duo_scaling: whether to use duo scaling, which uses both input activations
        and weights to determine the scaling factor
    """

    # Allow arbitrary types because AWQMapping has fields of type torch.nn.Module
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    # User-provided vars (in addition to QuantizationMixin args)
    sequential_targets: Union[str, List[str], None] = None
    mappings: Optional[List[AWQMapping]] = None
    offload_device: Optional[torch.device] = None
    duo_scaling: bool = True

    # Private vars set during validation
    _num_bits: Optional[int] = PrivateAttr(default=None)
    _symmetric: Optional[bool] = PrivateAttr(default=None)
    _group_size: Optional[int] = PrivateAttr(default=None)

    # Private vars set during initialization, cleared during finalization
    _resolved_mappings: List[ResolvedMapping] = PrivateAttr(default_factory=list)
    # Cache list of forward input args for each parent module, one dict for each batch
    _parent_args_cache: Dict[Module, IntermediatesCache] = PrivateAttr(
        default_factory=dict
    )
    # Dict[smooth layer name, (activation means, activation counts)]
    _smooth_activation_means: Dict[str, Tuple[torch.FloatTensor, int]] = PrivateAttr(
        default_factory=dict
    )

    # NOTE: different name chosen to avoid collision with
    # QuantizationMixin.validate_model_after, which must be called first
    @model_validator(mode="after")
    def validate_awq_after(model: "AWQModifier") -> "AWQModifier":
        """
        Confirm only one configuration for group_size, symmetric, and num_bits,
        as AWQ algorithm depends on it
        Confirm no activation quantization, as AWQ only works with WNA16
        """
        config = model.resolve_quantization_config()

        num_bits_set = set(
            group.weights.num_bits
            for group in config.config_groups.values()
            if group.weights is not None
        )
        assert (
            len(num_bits_set) == 1
        ), "In AWQ, all config groups must use the same configuration for num_bits"

        model._num_bits = next(iter(num_bits_set))

        symmetric_set = set(
            group.weights.symmetric
            for group in config.config_groups.values()
            if group.weights is not None
        )
        assert (
            len(symmetric_set) == 1
        ), "In AWQ, all config groups must use the same configuration for symmetric"

        model._symmetric = next(iter(symmetric_set))

        group_size_set = set(
            group.weights.group_size
            for group in config.config_groups.values()
            if group.weights is not None
        )
        assert (
            len(group_size_set) == 1
        ), "In AWQ, all config groups must use the same configuration for group_size"

        model._group_size = next(iter(group_size_set))

        in_num_bits_set = set(
            group.input_activations.num_bits
            for group in config.config_groups.values()
            if group.input_activations is not None
        )
        assert len(in_num_bits_set) == 0 or in_num_bits_set == {16}, (
            "AWQ activations must be 16-bit precision, "
            f"input activations {in_num_bits_set} not allowed"
        )

        out_num_bits_set = set(
            group.output_activations.num_bits
            for group in config.config_groups.values()
            if group.output_activations is not None
        )
        assert len(out_num_bits_set) == 0 or out_num_bits_set == {16}, (
            "AWQ activations must be 16-bit precision, "
            f"output activations {out_num_bits_set} not allowed"
        )

        return model

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

        if self.mappings is None:
            logger.info("No AWQModifier.mappings provided, inferring from model...")
            self.mappings = get_layer_mappings_from_architecture(
                architecture=state.model.__class__.__name__
            )

        self._set_resolved_mappings(state.model)

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

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

        for _, module in tqdm(
            match_named_modules(state.model, self.targets, self.ignore),
            desc="Calibrating weights",
        ):
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

        self._parent_args_cache.clear()
        self._smooth_activation_means.clear()
        self._resolved_mappings.clear()

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
        resolved_mappings: list[ResolvedMapping] = []
        for mapping_idx, mapping in enumerate(self.mappings):
            num_skipped_mappings = 0

            for smooth_name, smooth_layer in (
                pbar := tqdm(
                    match_named_modules(model, [mapping.smooth_layer], self.ignore)
                )
            ):
                pbar.set_description(
                    f"Resolving mapping {mapping_idx+1}/{len(self.mappings)}"
                    f" ({num_skipped_mappings} skipped)"
                )

                smooth_parent_name = ".".join(smooth_name.split(".")[:-1])
                smooth_parent = get_layer_by_name(smooth_parent_name, model)

                balance_layers, balance_names = [], []
                for balance_regex in mapping.balance_layers:
                    # find the submodules that match the activation layer
                    for balance_suffix, balance_layer in match_named_modules(
                        smooth_parent, [balance_regex], self.ignore
                    ):
                        balance_name = f"{smooth_parent_name}.{balance_suffix}"

                        # exclude v_proj->o_proj mappings whose shapes are incompatible
                        # https://github.com/mit-han-lab/llm-awq/pull/67#issuecomment-1681632777
                        if (
                            isinstance(smooth_layer, torch.nn.Linear)
                            and isinstance(balance_layer, torch.nn.Linear)
                            and balance_name.endswith(".o_proj")
                            and (
                                (
                                    smooth_name.endswith(".v_proj")
                                    and smooth_layer.out_features
                                    != balance_layer.in_features
                                )
                                or (
                                    smooth_name.endswith(".qkv_proj")
                                    and smooth_layer.out_features
                                    != 3 * balance_layer.in_features
                                )
                            )
                        ):
                            num_skipped_mappings += 1
                            continue

                        balance_layers.append(balance_layer)
                        balance_names.append(balance_name)

                if len(balance_layers) == 0:
                    continue

                elif len(balance_layers) == 1:
                    # for single balance layer, parent is the balance layer
                    parent_name, parent = balance_name, balance_layer
                else:
                    # for multiple balance layers, find lowest common parent
                    parent_name, parent = get_lowest_common_parent(balance_names, model)

                resolved_mappings.append(
                    ResolvedMapping(
                        smooth_name,
                        smooth_layer,
                        balance_layers,
                        balance_names=balance_names,
                        parent=parent,
                        parent_name=parent_name,
                    )
                )
        self._resolved_mappings = resolved_mappings
        return

    def _setup_activation_cache_hooks(self) -> None:
        """
        Attach a forward hook to each activation we want to smooth. This allows us to
        calculate the dynamic range during calibration
        """

        def cache_parent_kwargs_hook(
            module: torch.nn.Module,
            args: Tuple[torch.Tensor, ...],
            kwargs,
        ):
            values = inspect.signature(module.forward).bind(*args, **kwargs)
            self._parent_args_cache[module].append(values.arguments)

        def create_cache_smooth_activations_hook_fn(smooth_name):
            def cache_smooth_activations_hook(
                _module: torch.nn.Module,
                args: Tuple[torch.Tensor, ...],
                _output: torch.Tensor,
            ):
                self._smooth_activation_means[smooth_name] = _accumulate_mean(
                    # Assume that first argument is the input
                    args[0].cpu().abs().detach().squeeze(),
                    self._smooth_activation_means.get(smooth_name, None),
                )

            return cache_smooth_activations_hook

        for mapping in self._resolved_mappings:
            # parent kwargs needed for future forward passes
            # same parent may appear multiple times in resolved mappings
            if mapping.parent not in self._parent_args_cache:
                self._parent_args_cache[mapping.parent] = IntermediatesCache(
                    None,
                    self.offload_device,
                )
                self.register_hook(
                    mapping.parent,
                    cache_parent_kwargs_hook,
                    "forward_pre",
                    with_kwargs=True,
                )

            # input activations to balance layers needed for loss function
            # storing inputs to first balance layer is sufficient
            # other balance layers get the same input
            self.register_hook(
                mapping.balance_layers[0],
                create_cache_smooth_activations_hook_fn(mapping.smooth_name),
                "forward",
            )

    @torch.no_grad()
    def _apply_smoothing(self, model: Module) -> None:
        """
        Calculate the best scaling factors for each layer to smooth activations and
        apply the scaling factors to the weights of the next layer to offset the
        smoothing

        :param model: model to apply smoothing to
        """
        # NOTE: When using SequentialPipeline, not all the mappings
        # will have cached activations in the segment being udpated
        mappings_to_smooth = [
            mapping
            for mapping in self._resolved_mappings
            if mapping.smooth_name in self._smooth_activation_means
        ]
        for mapping in tqdm(mappings_to_smooth, desc="Smoothing"):
            smooth_layer = mapping.smooth_layer
            balance_layers = mapping.balance_layers
            parent_module = mapping.parent

            with align_modules(
                [parent_module, smooth_layer, *balance_layers]
            ), calibration_forward_context(model), HooksMixin.disable_hooks():
                # [STEP 1]: Compute per-channel mean of normalised weights
                # All layer weights are concatted together
                weight = torch.cat([bl.weight for bl in balance_layers], dim=0)
                org_shape = weight.shape
                # The weights are reshaped to be organised by quantization group
                weight = weight.view(-1, self._group_size)
                # Calculates the relative magnitude of the weights within
                # each of the quantization groups, and rescales each group
                # individually so that each group has weights on a 0-1 scale.
                weight.abs_()
                weight.div_(weight.amax(dim=1, keepdim=True) + 1e-6)
                # Resizes the rescaled weight matrix back up to its original dimensions
                weight = weight.view(org_shape)
                # Gets the average rescaled magnitude for each output channel
                w_mean = weight.mean(0)
                del weight

                # [STEP 3]: Compute output of module
                # could cache from hook, rather than recomputing here
                fp16_outputs = self._run_samples(parent_module)
                if len(fp16_outputs) == 0 or all(f.numel() == 0 for f in fp16_outputs):
                    logger.info(
                        f"Skipping smooth_layer {mapping.smooth_name}, no activations "
                        "found to scale. This can occasionally occur in MoE models "
                        "when certain experts are not activated by calibration samples."
                    )
                    del self._smooth_activation_means[mapping.smooth_name]
                    continue
                if not all(
                    [fp16_output.isfinite().all() for fp16_output in fp16_outputs]
                ):
                    logger.warning(
                        f"Skipping smooth_layer {mapping.smooth_name}, NaN or inf "
                        "outputs found during forward pass of the parent module "
                        f"{mapping.parent_name}. The model is either generating NaN "
                        "output with provided calibration data set, or the mappings "
                        "are incorrectly set and modifying the model in undesired "
                        "ways. If you encounter this consistently, raise an issue at "
                        "https://github.com/vllm-project/llm-compressor/issues"
                    )
                    del self._smooth_activation_means[mapping.smooth_name]
                    continue

                x_mean = self._smooth_activation_means[mapping.smooth_name][0]

                # [STEP 4]: Compute loss
                best_scales = self._compute_best_scale(
                    x_mean, w_mean, parent_module, balance_layers, fp16_outputs
                )

                @torch.no_grad()
                def _smooth(module):
                    scales = best_scales.to(module.weight.device)
                    if module in balance_layers:
                        update_offload_parameter(
                            module,
                            "weight",
                            module.weight.mul_(scales.view(1, -1)),
                        )
                    elif module == smooth_layer:
                        if module.weight.ndim == 1:
                            update_offload_parameter(
                                module,
                                "weight",
                                module.weight.div_(scales),
                            )
                        else:
                            # NOTE: edge case when smooth layer number of out_features
                            # is not equal to balance layer number of in_features
                            # e.g. when fused qkv_proj is used to smooth o_proj
                            # in this case, default to scaling the last output features
                            # because the desired smooth layer is v_proj
                            # https://github.com/casper-hansen/AutoAWQ/blob/main/awq/quantize/scale.py#L123
                            weight = module.weight
                            weight[-scales.size(0) :].div_(scales.view(-1, 1))
                            update_offload_parameter(module, "weight", weight)
                        if hasattr(module, "bias") and module.bias is not None:
                            update_offload_parameter(
                                module,
                                "bias",
                                module.bias.div_(scales),
                            )

                parent = get_fsdp_parent(mapping.smooth_name, model)
                if parent is not None:
                    parent.apply(_smooth)
                else:
                    # if we're not running with FSDP we can apply smoothing directly
                    for layer in balance_layers:
                        _smooth(layer)
                    _smooth(smooth_layer)

                # remove caches needed to smooth this mapping
                del self._smooth_activation_means[mapping.smooth_name]

        for v in self._parent_args_cache.values():
            v.batch_intermediates.clear()
        self._assert_all_activations_consumed()

    def _run_samples(self, module: Module) -> List[torch.Tensor]:
        outputs = [
            module(**batch_kwargs) for batch_kwargs in self._parent_args_cache[module]
        ]
        return [
            # If Tuple, assume that first argument is the input
            output[0] if isinstance(output, Tuple) else output
            for output in outputs
        ]

    def _compute_best_scale(
        self,
        x_mean: torch.Tensor,
        w_mean: torch.Tensor,
        parent_module: torch.nn.Module,
        linears2scale: List[torch.nn.Linear],
        fp16_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | _pseudo_quantize_tensor(W * s)
        X: inputs from calib dataset    | X
        W: original weights in FP16     | layer
        s: per channel scaling factor   | s^-1 * X
        """
        n_grid = 20
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")

        org_sd = {
            k: v.cpu()
            for k, v in parent_module.state_dict().items()
            if v.device != torch.device("meta")
        }

        device = get_execution_device(parent_module)
        x_mean = x_mean.view(-1).to(device)
        w_mean = w_mean.view(-1).to(device)

        for ratio in range(n_grid):
            # create new scales
            ratio = ratio / n_grid

            # NOTE: s^-1 * x is fused here, according to paper
            if self.duo_scaling:
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(
                    min=1e-4
                )
            else:
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            _scalesview = scales.view(1, -1).to(device)

            # avoid scaling values that overflow
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1

            # Q(W * s)
            for linear in linears2scale:
                linear.weight.mul_(_scalesview)
                update_offload_parameter(
                    linear,
                    "weight",
                    _pseudo_quantize_tensor(
                        w=linear.weight.data,
                        symmetric=self._symmetric,
                        bit_width=self._num_bits,
                        group_size=self._group_size,
                    )[0]
                    / _scalesview,
                )

            # W * X
            int_w_outputs = self._run_samples(parent_module)

            # compute mean squared error (L2 norm)
            loss = self._compute_loss(fp16_outputs, int_w_outputs, device)

            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()

            parent_module.load_state_dict(org_sd, strict=False)

        if best_ratio == -1:
            logger.debug(history)
            raise Exception(
                "No finite loss was found in best scalesgrid search. This typically "
                "means NaN values are appearing in the forward pass of the parent "
                "module. If you encounter this error, raise an issue at "
                "https://github.com/vllm-project/llm-compressor/issues"
            )

        assert (
            torch.isnan(best_scales).sum() == 0
        ), f"Nan found in scales: {best_scales}"

        return best_scales.detach().cpu()

    @torch.no_grad()
    def _compute_loss(
        self,
        fp16_outputs: List[torch.Tensor],
        int_w_outputs: List[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        loss = 0.0
        num_elements = 0

        # Compute the MSE loss for each batch
        for fp16_batch, int_w_batch in zip(fp16_outputs, int_w_outputs):
            batch_loss = (
                (fp16_batch.to(device) - int_w_batch.to(device))
                .view(-1)
                .float()
                .pow(2)
                .sum()
                .item()
            )
            loss += batch_loss
            num_elements += fp16_batch.numel()

        # Normalize the loss by the total number of elements
        loss /= num_elements

        return loss

    def _assert_all_activations_consumed(self):
        """
        Confirm all activations have been consumed
        If not, something has gone wrong
        """
        if len(self._smooth_activation_means) != 0:
            raise RuntimeError("Some cached activations were not used")


def _pseudo_quantize_tensor(
    w: torch.Tensor, symmetric: bool = False, bit_width: int = 8, group_size: int = -1
):
    org_w_shape = w.shape
    if group_size > 0:
        assert org_w_shape[-1] % group_size == 0, (
            f"org_w_shape ({org_w_shape[-1]}) must be a multiple "
            + f"of group_size ({group_size})!"
        )
        w = w.reshape(-1, group_size)
    assert w.dim() == 2
    assert torch.isnan(w).sum() == 0

    # zero point quantization
    if not symmetric:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**bit_width - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
        zeros = (zeros - 2 ** (bit_width - 1)).view(org_w_shape[0], -1)
    else:
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (bit_width - 1) - 1
        min_int = -(2 ** (bit_width - 1))
        scales = max_val / max_int
        zeros = None
        w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    scales = scales.view(org_w_shape[0], -1)
    w = w.reshape(org_w_shape)

    return w, scales, zeros


def _accumulate_mean(
    inp: torch.Tensor,
    prev_mean_and_count: Optional[Tuple[torch.FloatTensor, int]],
) -> Tuple[torch.FloatTensor, int]:
    sum_added = inp.sum(dim=0)
    num_added = inp.size(0)
    if prev_mean_and_count is None:
        return sum_added, num_added

    prev_mean, prev_count = prev_mean_and_count

    prev_sum = prev_mean * prev_count
    new_count = prev_count + num_added

    return (prev_sum + sum_added) / new_count, new_count


def get_lowest_common_parent(names: List[str], module: Module) -> Tuple[str, Module]:
    """
    Given a list of names, returns the lowest-scope common parent.

    NOTE: function excludes parents of type ModuleList, which don't play
    nicely with hooks because their forward method is never directly
    called for MoE models. See Qwen3MoeSparseMoeBlock for example, experts
    are selected based on router output and their forward method is called.
    https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L233

    Returns name of parent and pointer to parent module

    Implementation is a small alteration of os.path.commonprefix
    https://docs.python.org/3/library/os.path.html#os.path.commonprefix
    """
    s1 = min(names)
    s2 = max(names)
    parent_name = ""
    for i, c in enumerate(s1):
        if c != s2[i]:
            parent_name = s1[:i].rstrip(".")
            break

    while True:
        if parent_name == "":
            return "", module
        parent = get_layer_by_name(parent_name, module)
        if not isinstance(parent, torch.nn.ModuleList):
            return parent_name, parent
        parent_name = ".".join(parent_name.split(".")[:-1])
