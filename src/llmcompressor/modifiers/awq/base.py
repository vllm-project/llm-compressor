from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from loguru import logger
from pydantic import ConfigDict
from torch import Tensor
from torch.nn import Module
from tqdm import tqdm

from llmcompressor.core import Event, State
from llmcompressor.modifiers import Modifier
from llmcompressor.pytorch.utils.helpers import tensor_forward_with_input_args
from llmcompressor.utils.fsdp.helpers import get_fsdp_parent
from llmcompressor.utils.pytorch import (
    apply_clip_list,
    get_layers,
    get_layers_in_module,
    get_matching_layer,
    get_parent_by_name,
    mse_loss_with_chunking,
    pseudo_quantize_tensor,
    reclaim_memory,
    set_layer,
)

MappingsType = List[List[Union[List[str], str]]]
DEFAULT_AWQ_MAPPINGS: MappingsType = [
    [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
    [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
    [["re:.*down_proj"], "re:.*up_proj"],
    [["re:.*o_proj"], "re:.*v_proj"],
]


@dataclass
class AWQLayer:
    """
    A dataclass to store the name and layer of a layer in the model

    :param name: name of the layer
    :param layer: layer itself
    """

    name: str
    layer: Module


@dataclass
class AWQMapping:
    """
    A dataclass to store the layers to smooth, balance, and inspect
    for AWQ targets

    :param smooth_layer: AWQLayer to smooth
    :param balance_layers: list of AWQLayer(s) to balance
    :param inspect_layer: AWQLayer to inspect outputs of
        during quantization error calculation
    """

    smooth_layer: AWQLayer
    balance_layers: List[AWQLayer]
    inspect_layer: AWQLayer


class AWQModifier(Modifier):
    """
    Implements the AWQ (Activation-Weighted Quantization) algorithm,
    as described in https://arxiv.org/pdf/2306.00978. The algorithm
    significantly reduces quantization error by protecting 1%
    of the most salient weight channels.

    Instead of focusing on the weight values directly, AWQ identifies
    salient channels based on the activation distribution.

    To further minimize quantization error, the algorithm scales up these
    salient channels using an equivalent transformation. The scaling factor
    is determined offline by collecting activation statistics.

    Because this modifier manipulates the weights of the model, it can only be used in
    in one-shot and not during training. Activation values are determined by running a
    small set of calibration data through the model.

    example recipe:
    ```yaml
    AWQModifier:
    bits: 4
    mappings: [
        [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*self_attn_layer_norm"],
        [["re:.*fc1"], "re:.*final_layer_norm"]
    ]
    ignore: ["model.decoder.final_layer_norm"]
    ```
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
    :param bits: number of bits to quantize the weights to
    :param group_size: number of weights to group together for scaling
    :param symmetric: whether to use symmetric quantization
    :param duo_scaling: whether to use duo scaling, which uses both input activations
        and weights to determine the scaling factor
    :param apply_clip: whether to apply clipping to the weights
    :param grid_size: the grid search size for the scaling factor
    :param max_chunk_memory: maximum memory to use for each chunk of input activations

    """

    mappings: MappingsType = DEFAULT_AWQ_MAPPINGS
    ignore: Optional[List[str]] = None
    bits: int = 4
    group_size: int = 128
    symmetric: bool = True
    duo_scaling: bool = True
    apply_clip: bool = True
    grid_size: int = 20
    max_chunk_memory: int = 1024 * 1024 * 1024

    layer_inputs_: Union[List[Tensor], Tensor, None] = None
    module_kwargs_: Optional[Dict[str, Any]] = None
    layer_dict_: Dict[str, Module] = None
    resolved_mappings_: List[AWQMapping] = None
    hooks_: Optional[List] = None
    input_activations_: Optional[Dict] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def on_initialize_structure(self, state: State, **kwargs):
        # nothing needed for this modifier
        pass

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize and run AWQ on the given state

        :param state: state to run AWQ on
        :return: True on a successful run, False otherwise
        """
        self._validate_and_initialize(state)
        self._apply_awq_to_layers()
        return True

    def on_start(self, state: State, event: Event, **kwargs):
        pass

    def on_update(self, state: State, event: Event, **kwargs):
        pass

    def on_end(self, state: State, event: Event, **kwargs):
        pass

    def on_event(self, state: State, event: Event, **kwargs):
        pass

    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        Clean up by removing the hooks and freeing up resources

        :param state: unused
        :return: True
        """
        reclaim_memory(self.layer_inputs_)
        reclaim_memory(self.module_kwargs_)
        reclaim_memory(self.layer_dict_)
        reclaim_memory(self.resolved_mappings_)
        reclaim_memory(self.hooks_)
        reclaim_memory(self.input_activations_)
        return True

    def _validate_and_initialize(self, state: State):
        self._raise_error_if_not_oneshot()
        self.ignore = self.ignore or []
        self.layer_dict_, self.module_kwargs_, self.layer_inputs_ = (
            self._initialize_awq(state=state)
        )

    def _initialize_awq(
        self, state
    ) -> Tuple[Dict[str, Module], Dict[str, Any], Tensor]:
        """
        Initialize AWQ on the given state, returning the
        layer dictionary, module kwargs, and inputs to the
        first layer

        :param state: state to run AWQ on
        :return: A tuple containing the layer dictionary, module kwargs, and inputs to
            the first layer
        """
        model = state.model
        layer_dict = get_layers_in_module(module=model)
        calibration_dataloader = state.data.calib
        first_layer_name, first_layer = next(iter(layer_dict.items()))
        samples = []
        for sample in calibration_dataloader:
            if isinstance(sample, dict):
                sample = sample["input_ids"]
            samples.append(sample)
        samples = torch.cat(samples, dim=0)

        first_layer_inputs = []
        layer_kwargs = {}

        class Catcher(Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                # first input is hidden states
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                first_layer_inputs.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit

        logger.debug(f"Saving inputs to first layer {first_layer_name}")

        # Wrap first layer
        set_layer(first_layer_name, Catcher(first_layer), model)
        samples = samples.to(next(model.parameters()).device)

        with suppress(ValueError):
            model(samples)

        # restore original layer
        set_layer(first_layer_name, first_layer, model)

        layer_kwargs = model.prepare_inputs_for_generation(samples, **layer_kwargs)

        # input_ids not needed as kwargs
        layer_kwargs.pop("input_ids", None)

        del samples
        first_layer_inputs = first_layer_inputs[0]
        reclaim_memory()
        return layer_dict, layer_kwargs, first_layer_inputs

    def _apply_awq_to_layers(self):
        """
        Apply AWQ to the layers in the layer dictionary
        """
        logger.info("Applying AWQ...")
        for layer_name, layer in tqdm(
            self.layer_dict_.items(), desc="AWQ", leave=False
        ):
            self._apply_awq_to_layer(layer=layer, layer_name=layer_name)

    def _apply_awq_to_layer(self, layer: Module, layer_name: str):
        """
        Apply AWQ to the given layer

        :param layer: layer to apply AWQ to
        :param layer_name: name of the layer
        """
        self.resolved_mappings_ = self._resolve_mappings(module=layer)
        self._collect_activations(module=layer)
        for mapping in tqdm(self.resolved_mappings_, leave=False, desc=f"{layer_name}"):
            self._apply_awq_to_mapping(module=layer, mapping=mapping)

    def _resolve_mappings(self, module: Module) -> List:
        """
        Transforms the list of activations to smooth and their corresponding weights
        into AWQMapping objects, resolving regular expressions.

        For each activation in the mapping list, we find the corresponding weight to
        balance by searching for the longest substring. For instance, if our balance
        weight is ".*re:.*q_proj" and the activation is "re:.*self_attn_layer_norm" we
        would match model.layer.0.p_proj to model.layer.0.self_attn_layer_norm and
        repeat for model.layer.1 and so on
        """
        resolved_mappings = []
        for to_balance, to_smooth in self.mappings:
            to_smooth_layers = get_layers(to_smooth, module)
            for layer_name, smooth_layer in to_smooth_layers.items():
                if layer_name not in self.ignore:
                    balance_layers = []
                    for balance_suffix in to_balance:
                        # find the submodule that matches the activation layer
                        balance_name, balance_layer = get_matching_layer(
                            balance_suffix, layer_name, module
                        )
                        if balance_layer:
                            awq_balance_layer = AWQLayer(
                                name=balance_name, layer=balance_layer
                            )
                            balance_layers.append(awq_balance_layer)

                    # each mapping can contain multiple layers to balance, but only
                    # one layer to smooth

                    if len(balance_layers) == 1:
                        # for single balance layer, parent is the balance layer
                        inspect_layer = balance_layers[0]
                    else:
                        name_, layer_ = get_parent_by_name(layer_name, module)
                        inspect_layer = AWQLayer(name=name_, layer=layer_)

                    mapping = AWQMapping(
                        smooth_layer=AWQLayer(name=layer_name, layer=smooth_layer),
                        balance_layers=balance_layers,
                        inspect_layer=inspect_layer,
                    )
                    resolved_mappings.append(mapping)
        return resolved_mappings

    def _collect_activations(self, module: Module):
        self.hooks_ = []
        self.input_activations_ = defaultdict(list)

        self._setup_scale_hooks()
        outputs = self._module_forward_with_kwargs(
            module=module, inputs=self.layer_inputs_, **self.module_kwargs_
        )
        for hook in self.hooks_:
            hook.remove()

        # for next layer
        self.layer_inputs_ = outputs
        self.input_activations_ = {
            k: torch.cat(v, dim=0) for k, v in self.input_activations_.items()
        }

    def _setup_scale_hooks(self):
        """
        Attach a forward hook to each activation we want to smooth. This allows us to
        calculate the dynamic range during calibration
        """

        def create_hook_fn(layer_name):
            def hook_fn(module, inp, out):
                inp = inp[0]
                inp.cpu().detach()
                self.input_activations_[layer_name].append(inp)

            return hook_fn

        for mapping in self.resolved_mappings_:
            for awq_layer in mapping.balance_layers:
                name = awq_layer.name
                layer = awq_layer.layer
                self.hooks_.append(layer.register_forward_hook(create_hook_fn(name)))

    def _apply_awq_to_mapping(self, module: Module, mapping: AWQMapping):
        inputs = self.input_activations_[mapping.balance_layers[0].name]
        self._search_and_apply_smoothing_scales(
            module=module, mapping=mapping, inputs=inputs
        )
        if self.apply_clip:
            self._search_and_apply_clipping(
                mapping=mapping, module=module, inputs=inputs
            )

    def _search_and_apply_smoothing_scales(
        self, module: Module, mapping: AWQMapping, inputs: Tensor
    ):
        smoothing_scales: Tensor = self._search_smoothing_scales(
            balance_layers=mapping.balance_layers,
            inputs=inputs,
            inspect_layer=mapping.inspect_layer,
            kwargs=self.module_kwargs_,
        )
        self._apply_smoothing(
            mapping=mapping, module=module, smoothing_scales=smoothing_scales
        )

    @torch.no_grad()
    def _search_smoothing_scales(
        self,
        balance_layers: List[AWQLayer],
        inputs: Tensor,
        inspect_layer: AWQLayer,
        **kwargs,
    ) -> Tensor:
        """
        Computes the per-channel scaling factors for the given balance layers

        :param balance_layers: list of layers to balance (upscale)
        :param inputs: inputs to the balance layers
        :param inspect_layer: layer to inspect outputs of
            for quantization error calculation
        :param kwargs: additional kwargs to pass to the inspect layer
        :return: Per channel scaling factors which minimise the
            quantization error
        """
        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        weight = torch.cat(
            [awq_layer.layer.weight for awq_layer in balance_layers], dim=0
        )
        # per-channel mean of concatted normalised weights
        # from all balance layers
        weights_mean = self._per_channel_mean_weights(weight)

        # per-channel mean of the input activation with chunking
        inputs = inputs.to(next(inspect_layer.layer.parameters()).device)
        activation_means = self._per_channel_mean_activations(inputs)

        # output of module
        with torch.no_grad():
            fp16_output = self._module_forward_with_kwargs(
                module=inspect_layer.layer, inputs=inputs, **self.module_kwargs_
            )

        # grid search for the best scaling factor
        smoothing_scales = self._grid_search_scales(
            inputs=inputs,
            weights_mean=weights_mean,
            activation_mean=activation_means,
            inspect_layer=inspect_layer,
            balance_layers=balance_layers,
            fp16_output=fp16_output,
            kwargs=self.module_kwargs_,
        )

        return smoothing_scales

    @torch.no_grad()
    def _apply_smoothing(
        self, mapping: AWQMapping, module: Module, smoothing_scales: Tensor
    ):
        """
        Applies the given smoothing scales to the layers in the mapping

        :post-condition: The weights of the balance layers are
            scaled up (multiplied) by the per-channel scaling factors
            and the weights of the smooth layer are scaled down (divided)
            by the per-channel scaling factors
        :param mapping: AWQMapping to apply smoothing to
        :param module: module within which smoothing is applied
        :param smoothing_scales: per-channel scaling factors
        """
        balance_layers = mapping.balance_layers
        smooth_layer = mapping.smooth_layer

        def smooth(module):
            if any(module == awq_layer.layer for awq_layer in balance_layers):
                module.weight.mul_(
                    smoothing_scales.view(1, -1).to(module.weight.device)
                )
            elif module == smooth_layer.layer:
                if module.weight.ndim == 1:
                    module.weight.div_(smoothing_scales.to(module.weight.device))
                else:
                    module.weight.div_(
                        smoothing_scales.view(-1, 1).to(module.weight.device)
                    )
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.div_(smoothing_scales.to(module.bias.device))

        parent = get_fsdp_parent(layer_name=smooth_layer.name, model=module)
        if parent is not None:
            parent.apply(smooth)
        else:
            for awq_layer in balance_layers:
                smooth(awq_layer.layer)
            smooth(smooth_layer.layer)

    def _search_and_apply_clipping(
        self, mapping: AWQMapping, module: Module, inputs: Tensor
    ):
        clip_list: List[Tuple[str, Tensor]] = self._search_clip_list(
            balance_layers=mapping.balance_layers,
            input_feat=inputs,
        )
        apply_clip_list(module=module, clip_list=clip_list)

    @torch.no_grad()
    def _search_clip_list(self, balance_layers: List[AWQLayer], input_feat: Tensor):
        clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        for awq_layer in balance_layers:
            name = awq_layer.name
            layer = awq_layer.layer

            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in avoid_clipping]):
                continue

            max_val = self._grid_search_clipping_factor(
                weight=layer.weight,
                input_feat=input_feat,
            )
            clip_list.append((name, max_val))

        return clip_list

    @torch.no_grad()
    def _grid_search_clipping_factor(
        self,
        weight: Tensor,
        input_feat: Tensor,
        max_shrink=0.5,
        n_sample_token=512,
    ):
        assert weight.dim() == 2
        org_w_shape = weight.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = self.group_size if self.group_size > 0 else org_w_shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

        # Compute input feature step size (minimum 1)
        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size]

        weight = weight.reshape(org_w_shape[0], 1, -1, group_size)

        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64  # prevent OOM
        assert org_w_shape[0] % oc_batch_size == 0
        w_all = weight
        best_max_val_all = []

        for i_b in range(org_w_shape[0] // oc_batch_size):
            weight = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            org_max_val = weight.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(weight.device)
            org_out = (input_feat * weight).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * self.grid_size)):
                max_val = org_max_val * (1 - i_s / self.grid_size)
                min_val = -max_val
                cur_w = torch.clamp(weight, min_val, max_val)
                q_w = pseudo_quantize_tensor(
                    weights=cur_w,
                    symmetric=self.symmetric,
                    group_size=group_size,
                    bit_width=self.bits,
                )[0]
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        reclaim_memory(input_feat)
        reclaim_memory(org_out)

        return best_max_val.squeeze(1)

    @torch.no_grad()
    def _module_forward_with_kwargs(self, module: Module, inputs: Tensor, **kwargs):
        """
        Forward pass for a module with the given inputs and kwargs

        :param module: module to forward pass through
        :param inputs: inputs to the module
        :param kwargs: additional kwargs to pass to the module
        :return: output of the forward pass
        """
        return tensor_forward_with_input_args(module=module, inputs=inputs, **kwargs)[0]

    def _raise_error_if_not_oneshot(self):
        if not (self.end is None or self.end == -1):
            raise ValueError(
                f"{self.__class__.__name__} can only be applied during one-shot. "
                f" Expected end to be None or -1, got {self.end}"
            )
        if self.start and self.start != -1:
            raise ValueError(
                f"{self.__class__.__name__} can only be applied during one-shot. "
                f"Expected start to be None or -1, got {self.end}"
            )

    def _per_channel_mean_weights(self, weight):
        org_shape = weight.shape
        # The weights are reshaped to be organised
        # by quantization group
        weight = weight.view(-1, self.group_size)

        # Calculates the relative magnitude of the weights
        # within each of the quantization groups,
        # and rescales each group individually such that
        # each group has weights on a 0-1 scale.
        w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)

        # Resizes the rescaled weight matrix back up to its original dimensions
        w_scale = w_scale.view(org_shape)

        # Gets the average rescaled magnitude for each output channel
        w_mean = w_scale.mean(0)
        reclaim_memory(weight)
        return w_mean

    def _per_channel_mean_activations(self, inputs):
        inp_flat = inputs.cpu().abs().view(-1, inputs.shape[-1])
        num_elements = inp_flat.size(0)
        num_channels = inp_flat.size(1)
        element_size_bytes = inp_flat.element_size() * 2  # multiplied by 2 for FP32

        # Calculate chunk size dynamically based on max_chunk_memory
        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))
        chunk_size = min(chunk_size, num_elements)

        # Use float32 for sum calculation
        x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inputs.device)

        for i in range(0, num_elements, chunk_size):
            end = min(i + chunk_size, num_elements)
            chunk_sum = inp_flat[i:end].to(torch.float32).sum(dim=0)
            x_sum += chunk_sum.to(inputs.device)

        x_mean = (x_sum / num_elements).to(inputs.dtype)
        reclaim_memory(x_sum)
        return x_mean

    def _grid_search_scales(
        self,
        inputs: Tensor,
        weights_mean: Tensor,
        activation_mean: Tensor,
        inspect_layer: AWQLayer,
        balance_layers: List[AWQLayer],
        fp16_output: Tensor,
        kwargs: Dict = {},
    ):
        """
        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||

        Notations:
            Q: weight quantization function | pseudo_quantize_tensor(W * s)
            X: inputs from calib dataset    | X
            W: original weights in FP16     | layer
            s: per channel scaling factor   | s^-1 * X

        :param inputs: inputs to the balance layers
        :param weights_mean: per channel mean of the weights
        :param activation_mean: per channel mean of the inputs
        :param inspect_layer: layer to inspect outputs of
            for quantization error calculation
        :param balance_layers: list of layers to balance (upscale)
        :param fp16_output: output of the inspect layer in FP16
        :param kwargs: additional kwargs to pass to the inspect layer
        :return: Per channel scaling factors which minimise the
            quantization error
        """

        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")

        org_sd = {k: v.cpu() for k, v in inspect_layer.layer.state_dict().items()}

        device = inputs.device
        activation_mean = activation_mean.view(-1).to(device)
        weights_mean = weights_mean.view(-1).to(device)

        for ratio in range(self.grid_size):
            ratio = ratio / self.grid_size

            if self.duo_scaling:
                candidate_scales = (
                    activation_mean.pow(ratio) / (weights_mean.pow(1 - ratio) + 1e-4)
                ).clamp(min=1e-4)
            else:
                candidate_scales = activation_mean.pow(ratio).clamp(min=1e-4).view(-1)
            candidate_scales = (
                candidate_scales
                / (candidate_scales.max() * candidate_scales.min()).sqrt()
            )
            scales_view = candidate_scales.view(1, -1).to(device)

            # avoid overflow
            candidate_scales[torch.isinf(candidate_scales)] = 1
            candidate_scales[torch.isnan(candidate_scales)] = 1

            # Q(W * s) * s^-1
            for awq_layer in balance_layers:
                linear = awq_layer.layer
                linear.weight.mul_(scales_view)
                linear.weight.data = (
                    self._pseudo_quantize_tensor(linear.weight.data)[0] / scales_view
                )

            # Q(W * s) * s^-1 * X
            int_w_output = self._module_forward_with_kwargs(
                module=inspect_layer.layer, inputs=inputs, **kwargs
            )

            quant_error = self._mse_loss_with_chunking(
                fp16_output, int_w_output, device
            )

            history.append(quant_error)
            if quant_error < best_error:
                best_error = quant_error
                best_ratio = ratio
                best_scales = candidate_scales.clone()
            inspect_layer.layer.load_state_dict(org_sd)

        if best_ratio == -1:
            logger.debug(history)
            raise Exception("Best ratio could not be found")

        assert torch.isnan(best_scales).sum() == 0, best_scales

        return best_scales.detach().cpu()

    def _mse_loss_with_chunking(
        self,
        fp16_output: Tensor,
        int_w_output: Tensor,
        device: torch.device,
    ) -> float:
        return mse_loss_with_chunking(
            tensor_a=fp16_output,
            tensor_b=int_w_output,
            device=device,
            max_chunk_memory=self.max_chunk_memory,
        )

    def _pseudo_quantize_tensor(
        self, weights: Tensor
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        return pseudo_quantize_tensor(
            weights=weights,
            symmetric=self.symmetric,
            bit_width=self.bits,
            group_size=self.group_size,
        )
