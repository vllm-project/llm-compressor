import inspect
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from compressed_tensors.quantization import disable_quantization
from compressed_tensors.utils import (
    align_module_device,
    get_execution_device,
    update_offload_parameter,
)
from loguru import logger
from pydantic import ConfigDict, PrivateAttr, model_validator
from torch.nn import Module
from torch.utils.hooks import RemovableHandle
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
from llmcompressor.utils.fsdp.helpers import get_fsdp_parent
from llmcompressor.utils.helpers import calibration_forward_context
from llmcompressor.utils.pytorch.module import (
    get_layers,
    get_matching_layer,
    get_parent_by_name,
)

__all__ = ["AWQModifier"]


# TODO (Brian INFERENG-531) Add support for offloaded models
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
    :param group_size: number of weights to group together for scaling
    :param max_chunk_memory: maximum memory to use for each chunk of input activations
    :param bits: number of bits to quantize the weights to
    :param symmetric: whether to use symmetric quantization
    :param duo_scaling: whether to use duo scaling, which uses both input activations
        and weights to determine the scaling factor
    """

    # Allow arbitrary types because AWQMapping has fields of type torch.nn.Module
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    # User-provided vars (in addition to QuantizationMixin args)
    sequential_targets: Union[str, List[str], None] = None
    mappings: Optional[List[AWQMapping]] = None
    max_chunk_memory: int = 1024 * 1024 * 1024
    duo_scaling: bool = True

    # Private vars set during validation
    _num_bits: Optional[int] = PrivateAttr(default=None)
    _symmetric: Optional[bool] = PrivateAttr(default=None)
    _group_size: Optional[int] = PrivateAttr(default=None)

    # Private vars set during initialization, cleared during finalization
    _resolved_mappings: List[ResolvedMapping] = PrivateAttr(default_factory=list)
    _activations: Dict[str, List[torch.Tensor]] = PrivateAttr(default_factory=dict)
    _activation_hooks: Set[RemovableHandle] = PrivateAttr(default_factory=set)
    _module_kwargs: Dict = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def validate_model_after(model: "AWQModifier") -> "AWQModifier":
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

        self._set_module_kwargs(state.model, state.data.calib)

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

        # register quantization calibration hooks
        # assume quantization has been initialized by this modifier or one before it
        QuantizationMixin.start_calibration(self, state.model)
        # Unlike qmod, do not quantize as we calibrate
        # This choice does not seem to have a meaningful impact on accuracy
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

        modules = list(state.model.modules())
        for module in tqdm(modules, desc="Calibrating weights"):
            update_weight_zp_scale(module)

        QuantizationMixin.end_calibration(self, state.model)

        # remove activation hooks
        self.remove_hooks(self._activation_hooks)
        self._activation_hooks.clear()

    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        Clean up by clearing the activations and mapping data

        :param state: unused
        :return: True
        """
        if not self.ended_:
            self.on_end(state, None)

        self._activations.clear()
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
        num_skipped_oproj_mappings = 0
        for mapping in self.mappings:
            to_smooth_layers = get_layers(mapping.smooth_layer, model)
            for layer_name, smooth_layer in to_smooth_layers.items():
                # always exclude `.weight_observer`, only want `.weight`
                if layer_name not in self.ignore and not layer_name.endswith(
                    "_observer"
                ):
                    balance_layers, balance_names = [], []
                    for balance_suffix in mapping.balance_layers:
                        # find the submodule that matches the activation layer
                        balance_name, balance_layer = get_matching_layer(
                            balance_suffix, layer_name, model
                        )
                        if not balance_layer:
                            continue

                        # exclude v_proj->o_proj mappings whose shapes are incompatible
                        # https://github.com/mit-han-lab/llm-awq/pull/67#issuecomment-1681632777
                        if (
                            isinstance(smooth_layer, torch.nn.Linear)
                            and isinstance(balance_layer, torch.nn.Linear)
                            and ".o_proj" in balance_name
                            and (
                                (
                                    ".v_proj" in layer_name
                                    and smooth_layer.out_features
                                    != balance_layer.in_features
                                )
                                or (
                                    ".qkv_proj" in layer_name
                                    and smooth_layer.out_features
                                    != 3 * balance_layer.in_features
                                )
                            )
                        ):
                            num_skipped_oproj_mappings += 1
                            continue

                        balance_layers.append(balance_layer)
                        balance_names.append(balance_name)

                    if len(balance_layers) == 0:
                        continue

                    # each mapping can contain multiple layers to balance, but only
                    # one layer to smooth
                    if len(balance_layers) == 1:
                        # for single balance layer, parent is the balance layer
                        parent_name, parent = balance_name, balance_layer
                    else:
                        # for multiple balance layers,
                        # parent of any balance layer is the parent
                        parent_name, parent = get_parent_by_name(
                            layer_name=balance_name, model=model
                        )
                    resolved_mappings.append(
                        ResolvedMapping(
                            layer_name,
                            smooth_layer,
                            balance_layers,
                            balance_names=balance_names,
                            parent=parent,
                            parent_name=parent_name,
                        )
                    )
        if num_skipped_oproj_mappings > 0:
            logger.info(
                f"Excluded {num_skipped_oproj_mappings} from resolved "
                "mappings due to shape mismatch"
            )
        self._resolved_mappings = resolved_mappings
        return

    def _setup_activation_cache_hooks(self) -> None:
        """
        Attach a forward hook to each activation we want to smooth. This allows us to
        calculate the dynamic range during calibration
        """

        def create_cache_activation_hook(smooth_layer_name):
            def cache_activation_hook_fn(
                _module: torch.nn.Module,
                args: Tuple[torch.Tensor, ...],
                _output: torch.Tensor,
            ):
                # Assume that first argument is the input
                inp = args[0].cpu().detach()

                if smooth_layer_name in self._activations:
                    self._activations[smooth_layer_name].append(inp)
                else:
                    self._activations[smooth_layer_name] = [inp]

            return cache_activation_hook_fn

        for mapping in self._resolved_mappings:
            # storing inputs to first balance layer is sufficient
            # other balance layers get the same input
            layer = mapping.balance_layers[0]
            hook = self.register_hook(
                layer, create_cache_activation_hook(mapping.smooth_name), "forward"
            )
            self._activation_hooks.add(hook)

    @torch.no_grad()
    def _apply_smoothing(self, model: Module) -> None:
        """
        Calculate the best scaling factors for each layer to smooth activations and
        apply the scaling factors to the weights of the next layer to offset the
        smoothing

        :param model: model to apply smoothing to
        """
        for mapping in tqdm(self._resolved_mappings, desc="Smoothing"):
            # NOTE: When using SequentialPipeline, not all the mappings
            # will have cached activations in the segment being udpated
            if mapping.smooth_name not in self._activations:
                continue

            activations = torch.cat(self._activations[mapping.smooth_name], dim=0)
            del self._activations[mapping.smooth_name]

            smooth_layer = mapping.smooth_layer
            balance_layers = mapping.balance_layers
            module2inspect = mapping.parent

            # [STEP 1]: Compute per-channel mean of normalised weights
            # All layer weights are concatted together
            weight = torch.cat([bl.weight for bl in balance_layers], dim=0)
            org_shape = weight.shape
            # The weights are reshaped to be organised by quantization group
            weight = weight.view(-1, self._group_size)
            # Calculates the relative magnitude of the weights within
            # each of the quantization groups, and rescales each group
            # individually so that each group has weights on a 0-1 scale.
            w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
            # Resizes the rescaled weight matrix back up to its original dimensions
            w_scale = w_scale.view(org_shape)
            # Gets the average rescaled magnitude for each output channel
            w_mean = w_scale.mean(0)

            # [STEP 2]: Compute per-channel mean of the input activation with chunking
            # move inp to cpu to avoid memory leak
            inp = activations.to(weight.device)
            inp_flat = activations.cpu().abs().view(-1, inp.shape[-1])
            num_elements = inp_flat.size(0)
            num_channels = inp_flat.size(1)
            element_size_bytes = inp_flat.element_size() * 2  # multiplied by 2 for FP32

            # Calculate chunk size dynamically based on max_chunk_memory
            chunk_size = int(
                self.max_chunk_memory // (element_size_bytes * num_channels)
            )
            chunk_size = min(chunk_size, num_elements)

            # Use float32 for sum calculation
            x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp.device)

            for i in range(0, num_elements, chunk_size):
                end = min(i + chunk_size, num_elements)
                chunk_sum = inp_flat[i:end].to(torch.float32).sum(dim=0)
                x_sum += chunk_sum.to(inp.device)

            x_mean = (x_sum / num_elements).to(inp.dtype)

            with calibration_forward_context(model), HooksMixin.disable_hooks():
                # [STEP 3]: Compute output of module
                fp16_output = self._forward_input_with_kwargs(
                    module=module2inspect,
                    inputs=inp,
                    input_kwargs=_sanitize_kwargs(self._module_kwargs, module2inspect),
                )
                fp16_output = fp16_output.clip(
                    torch.finfo(fp16_output.dtype).min,
                    torch.finfo(fp16_output.dtype).max,
                )

                # [STEP 4]: Compute loss
                best_scales = self._compute_best_scale(
                    inp, w_mean, x_mean, module2inspect, balance_layers, fp16_output
                )

            @torch.no_grad()
            def smooth(module):
                with align_module_device(module):
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
                parent.apply(smooth)
            else:
                # if we're not running with FSDP we can apply smoothing directly
                for layer in balance_layers:
                    smooth(layer)
                smooth(smooth_layer)

        self._assert_all_activations_consumed()

    def _compute_best_scale(
        self,
        x: torch.Tensor,
        w_mean: torch.Tensor,
        x_mean: torch.Tensor,
        module2inspect: torch.nn.Module,
        linears2scale: List[torch.nn.Linear],
        fp16_output: torch.Tensor,
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

        org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}

        device = x.device
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
                with align_module_device(linear):
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
            int_w_output = self._forward_input_with_kwargs(
                module=module2inspect, inputs=x, input_kwargs=self._module_kwargs
            )
            int_w_output = int_w_output.clip(
                torch.finfo(int_w_output.dtype).min,
                torch.finfo(int_w_output.dtype).max,
            )

            # compute mean squared error (L2 norm)
            loss = self._compute_loss(fp16_output, int_w_output, device)

            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()
            module2inspect.load_state_dict(org_sd)

        if best_ratio == -1:
            logger.debug(history)
            raise Exception

        assert (
            torch.isnan(best_scales).sum() == 0
        ), f"Nan found in scales: {best_scales}"

        return best_scales.detach().cpu()

    @torch.no_grad()
    def _compute_loss(
        self,
        fp16_output: torch.Tensor,
        int_w_output: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        loss = 0.0
        fp16_output_flat = fp16_output.view(-1)
        int_w_output_flat = int_w_output.view(-1)
        num_elements = fp16_output_flat.size(0)
        element_size_bytes = fp16_output.element_size()

        # Calculate chunk size dynamically based on max_chunk_memory
        # Divide the max_chunk_memory by twice the element size
        chunk_size = self.max_chunk_memory // (element_size_bytes * 2)
        chunk_size = min(chunk_size, num_elements)

        # Split the computation into chunks
        fp16_chunks = torch.split(fp16_output_flat, chunk_size)
        int_w_chunks = torch.split(int_w_output_flat, chunk_size)

        # Compute the MSE loss for each chunk
        for fp16_chunk, int_w_chunk in zip(fp16_chunks, int_w_chunks):
            chunk_loss = (
                (fp16_chunk.to(device) - int_w_chunk.to(device))
                .float()
                .pow(2)
                .sum()
                .item()
            )
            loss += chunk_loss

        # Normalize the loss by the total number of elements
        loss /= num_elements

        return loss

    def _assert_all_activations_consumed(self):
        """
        Confirm all activations have been consumed
        If not, something has gone wrong
        """
        if len(self._activations) > 0:
            raise RuntimeError("Some cached activations were not used")

    def _set_module_kwargs(self, model, dataloader) -> None:
        _, modules = next(iter(get_layers("re:.*layers", model).items()))

        samples = [batch["input_ids"] for batch in dataloader]

        samples = torch.cat(samples, dim=0)

        inps = []
        layer_kwargs = {}

        best_device = "cuda"
        modules[0] = modules[0].to(best_device)

        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
        class Catcher(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                # assume first input to forward is hidden states
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        # patch layer 0 to catch input and kwargs
        modules[0] = Catcher(modules[0])
        try:
            with calibration_forward_context(model):
                model(samples.to(next(model.parameters()).device))
        except ValueError:  # work with early exit
            pass
        modules[0] = modules[0].module  # restore

        # Update the layer kwargs with `prepare_inputs_for_generation` method
        # that takes care of everything to avoid unexpected errors.
        layer_kwargs = model.prepare_inputs_for_generation(samples, **layer_kwargs)
        # Pop the input_ids as they are not needed at all.
        layer_kwargs.pop("input_ids")

        del samples
        inps = inps[0]

        if layer_kwargs.get("attention_mask") is not None:
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to(
                best_device
            )

        self._module_kwargs = layer_kwargs

    def _forward_input_with_kwargs(
        self,
        module: Module,
        inputs: torch.Tensor,
        input_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Forward pass with input arguments

        :param module: module to run forward pass on
        :param inputs: input tensor to pass to the module
        :param input_kwargs: additional arguments to pass to the module
        :return: the first output tensor from the forward pass
        """
        kwargs = input_kwargs or self._module_kwargs
        kwargs = _sanitize_kwargs(kwargs, module)

        inputs = inputs.to(get_execution_device(module))

        return module(inputs, **kwargs)[0]


def _sanitize_kwargs(input_kwargs: Dict[str, Any], module: Module) -> Dict[str, Any]:
    """
    Sanitize input keyword arguments to match the module's forward method signature,
    excluding `use_cache` which is not desired to be passed into module.

    Args:
        inputs_kwargs (`dict`):
            The input dictionary to pass to the model layer
        module (`torch.nn.Module`):
            Target module to quantize.
    """

    params = inspect.signature(module.forward).parameters

    # Filter out any kwargs not in module.forward signature
    sanitized_kwargs = {k: v for k, v in input_kwargs.items() if k in params}

    # Edge Case: forward pass has optional dependencies that don't default to None.
    # This is the case for `LlamaAttention.forward` which has input
    #  `attention_mask: Optional[torch.Tensor],` (with no `= None` default)
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L246
    for k, v in params.items():
        if (
            k not in sanitized_kwargs
            and v.default is inspect.Parameter.empty
            and str(v.annotation).startswith("typing.Optional")
        ):
            sanitized_kwargs[k] = None

    # Exclude `use_cache` entirely
    sanitized_kwargs.pop("use_cache", None)

    return sanitized_kwargs


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
