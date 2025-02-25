from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from compressed_tensors.utils import align_module_device, update_offload_parameter
from loguru import logger
from pydantic import ConfigDict
from torch.nn import Module
from tqdm import tqdm

from llmcompressor.core import State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.utils.pytorch_helpers import run_calibration_forward
from llmcompressor.pytorch.utils import (
    pseudo_quantize_tensor,
    tensor_forward_with_input_args,
)
from llmcompressor.utils.fsdp.helpers import get_fsdp_parent
from llmcompressor.utils.helpers import calibration_forward_context
from llmcompressor.utils.pytorch.module import (
    get_layer,
    get_layers,
    get_matching_layer,
    get_parent_by_name,
)

__all__ = ["AWQScale", "AWQMapping", "AWQModifier"]


@dataclass
class AWQScale:
    """
    Dataclass for storing the input activations of a layer to be smoothed
    """

    inps: Union[List[torch.Tensor], torch.Tensor]


@dataclass
class AWQMapping:
    """
    Dataclass storing config of activation mappings to smooth
    The output activations of smooth_layer are input activations
    into the balance_layers

    `AWQMapping`s are resolved into `ResolvedMapping`s, which
    retain pointers to the actual `torch.nn.Module`s and additional
    metadata at runtime
    """

    smooth_layer: str
    balance_layers: list[str]


DEFAULT_AWQ_MAPPINGS: list[AWQMapping] = [
    AWQMapping(
        "re:.*input_layernorm",
        ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
    ),
    AWQMapping(
        "re:.*post_attention_layernorm",
        ["re:.*gate_proj", "re:.*up_proj"],
    ),
    AWQMapping(
        "re:.*up_proj",
        ["re:.*down_proj"],
    ),
    # TODO this generally results in higher perplexity for llama 2 7B on wikitext
    # AWQMapping("re:.*v_proj", ["re:.*o_proj"]),
]


@dataclass
class ResolvedMapping:
    """
    Dataclass for storing the resolved mappings between an activation layer
    and the following weights that must be balanced during smoothing

    :param smooth_name: name of the activation layer
    :param smooth_layer: PyTorch module storing the activation layer
    :param balance_layers: list of PyTorch modules that smooth_layer feeds into, must be
        balanced to offset the smoothing of smooth_layer
    :param balance_names: optional list of names of the balance_layers
    :param parent: parent module of the balance_layers
    :param parent_name: name of the parent module
    """

    smooth_name: str
    smooth_layer: Module
    balance_layers: List[Module]
    balance_names: Optional[List[str]] = None
    parent: Optional[Module] = None
    parent_name: Optional[str] = None


class AWQModifier(Modifier):
    """
    Implements the AWQ (Activation-Weighted Quantization) algorithm,
    as described in https://arxiv.org/pdf/2306.00978. The algorithm
    significantly reduces quantization error by protecting only 1%
    of the most salient weight channels.

    Instead of focusing on the weight values directly, AWQ identifies
    salient channels based on the activation distribution.
    To further minimize quantization error, the algorithm scales up these
    salient channels using an equivalent transformation. The scaling factor
    is determined offline by collecting activation statistics

    Because this modifier manipulates the weights of the model, it can only be used in
    in one-shot and not during training. Activation ranges are determined by running a
    small set of calibration data through the model.

    example recipe:
    ```yaml
    AWQModifier:
      bits: 4
      mappings:
        - smooth_layer: "re:.*self_attn_layer_norm"
          balance_layers: ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"]
        - smooth_layer: "re:.*final_layer_norm"
          balance_layers: ["re:.*fc1"]
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
    :param num_calibration_steps: number of samples to use for calibration, or None to
        use the whole dataset
    :param calibration_function: optional function to use for the forward pass, or None
        to use the default tensor_module_forward
    :param group_size: number of weights to group together for scaling
    :param max_chunk_memory: maximum memory to use for each chunk of input activations
    :param bits: number of bits to quantize the weights to
    :param symmetric: whether to use symmetric quantization
    :param duo_scaling: whether to use duo scaling, which uses both input activations
        and weights to determine the scaling factor
    :param apply_clip: whether to apply clipping to the weights after scaling
    """

    # Allow arbitrary types because AWQMapping has fields of type torch.nn.Module
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    mappings: List[AWQMapping] = DEFAULT_AWQ_MAPPINGS
    ignore: Optional[List[str]] = None
    num_calibration_steps: Optional[int] = None
    calibration_function: Optional[Callable] = None
    group_size: int = 128
    max_chunk_memory: int = 1024 * 1024 * 1024
    bits: int = 4
    symmetric: bool = True
    duo_scaling: bool = True
    apply_clip: bool = True

    resolved_mappings_: Optional[List[ResolvedMapping]] = None
    scales_: Optional[Dict] = None
    module_kwargs_: Optional[Dict] = None

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize and run AWQ on the given state

        :param state: state to run AWQ on
        :return: True on a successful run, False otherwise
        """
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

        self.ignore = [] if not self.ignore else self.ignore
        self.resolved_mappings_ = self._get_resolved_mappings(state.model)
        self.scales_ = {}

        calibration_dataloader = state.data.calib

        with calibration_forward_context(state.model):
            self._set_module_kwargs(state.model, calibration_dataloader)
            self._setup_scale_hooks()
            self._calibrate(state.model, calibration_dataloader)
            self._concat_collected_activations()
            self._apply_smoothing(state.model)

        return True

    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        Clean up by clearing the scale and mapping data

        :param state: unused
        :return: True
        """
        if self.scales_ is not None:
            self.scales_.clear()
        if self.resolved_mappings_ is not None:
            self.resolved_mappings_.clear()

        return True

    def _get_resolved_mappings(self, model: Module) -> List[ResolvedMapping]:
        """
        Transforms the list of activations to smooth and their corresponding weights
        into ResolvedMapping objects, resolving regular expressions.

        For each activation in the mapping list, we find the corresponding weight to
        balance by searching for the longest substring. For instance, if our balance
        weight is ".*re:.*q_proj" and the activation is "re:.*self_attn_layer_norm" we
        would match model.layer.0.p_proj to model.layer.0.self_attn_layer_norm and
        repeat for model.layer.1 and so on
        """
        resolved_mappings: list[ResolvedMapping] = []
        for mapping in self.mappings:
            to_smooth_layers = get_layers(mapping.smooth_layer, model)
            for layer_name, smooth_layer in to_smooth_layers.items():
                if layer_name not in self.ignore:
                    balance_layers, balance_names = [], []
                    for balance_suffix in mapping.balance_layers:
                        # find the submodule that matches the activation layer
                        balance_name, balance_layer = get_matching_layer(
                            balance_suffix, layer_name, model
                        )
                        if balance_layer:
                            balance_layers.append(balance_layer)
                            balance_names.append(balance_name)

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
        return resolved_mappings

    def _setup_scale_hooks(self):
        """
        Attach a forward hook to each activation we want to smooth. This allows us to
        calculate the dynamic range during calibration
        """

        def create_hook_fn(layer_name):
            def hook_fn(module, inp, out):
                inp = inp[0].cpu().detach()

                if layer_name in self.scales_:
                    self.scales_[layer_name].inps.append(inp)
                else:
                    self.scales_[layer_name] = AWQScale(inps=[inp])

            return hook_fn

        for mapping in self.resolved_mappings_:
            name = mapping.smooth_name
            # storing inps to first balance layer
            # is enough, as other balance layers
            # get the same input
            layer = mapping.balance_layers[0]
            self.register_hook(layer, create_hook_fn(name), "forward")

    @torch.no_grad()
    def _calibrate(self, model: Module, calibration_dataloader: List):
        """
        Catch the output dynamic ranges of each layer that will be smoothed by running
        forward passes with calibration_dataloader
        """
        class_name = self.__class__.__name__.replace("PyTorch", "")
        logger.info(
            f"Running {class_name} calibration with "
            f"{len(calibration_dataloader)} samples..."
        )
        if not calibration_dataloader:
            raise ValueError(
                "Calibration data loader not set, must populate the calib_data field of"
                " CompressionSession to run the AWQ modifier"
            )

        # with calibration_forward_context(model):
        run_calibration_forward(
            model,
            calibration_dataloader,
            self.num_calibration_steps,
            self.calibration_function,
        )

        # remove the hooks now that we are done calibrating
        self.remove_hooks()

    def _concat_collected_activations(self):
        """
        Concatenate the collected activation values from each forward pass into a single
        tensor for each layer

        :postcondition: each layer in self.scales_ will have a single tensor containing
            all the activation values seen during calibration
        """
        for mapping in self.resolved_mappings_:
            name = mapping.smooth_name
            self.scales_[name].inps = torch.cat(self.scales_[name].inps, dim=0)

        torch.cuda.empty_cache()

    @torch.no_grad()
    def _apply_smoothing(self, model: Module):
        """
        Calculate the best scaling factors for each layer to smooth activations and
        apply the scaling factors to the weights of the next layer to offset the
        smoothing

        :param model: model to apply smoothing to
        """
        logger.info("Smoothing activation scales...")
        for mapping in tqdm(self.resolved_mappings_):
            smooth_layer = mapping.smooth_layer
            balance_layers = mapping.balance_layers
            balance_names = mapping.balance_names

            activations = self.scales_[mapping.smooth_name].inps

            module2inspect = mapping.parent

            # [STEP 1]: Compute per-channel mean of normalised weights
            # All layer weights are concatted together
            weight = torch.cat([bl.weight for bl in balance_layers], dim=0)
            org_shape = weight.shape
            # The weights are reshaped to be organised by quantization group
            weight = weight.view(-1, self.group_size)
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

            # [STEP 3]: Compute output of module
            fp16_output = self._forward_input_with_kwargs(
                module=module2inspect, inputs=inp, input_kwargs=self.module_kwargs_
            )

            # [STEP 4]: Compute loss
            best_scales = self._compute_best_scale(
                inp, w_mean, x_mean, module2inspect, balance_layers, fp16_output
            )

            scales = best_scales

            @torch.no_grad()
            def smooth(module):
                # TODO calls to module._hf_hook.pre_forward(module) and
                # module._hf_hook.post_forward(module, None) appear a couple places
                # in SmoothQuantModifier, do we need them anywhere else?
                with align_module_device(module):
                    if module in balance_layers:
                        module.weight.mul_(scales.view(1, -1).to(module.weight.device))
                    elif module == smooth_layer:
                        if module.weight.ndim == 1:
                            module.weight.div_(scales.to(module.weight.device))
                        else:
                            module.weight.div_(
                                scales.view(-1, 1).to(module.weight.device)
                            )
                        if hasattr(module, "bias") and module.bias is not None:
                            module.bias.div_(scales.to(module.bias.device))

            parent = get_fsdp_parent(mapping.smooth_name, model)
            if parent is not None:
                parent.apply(smooth)
            else:
                # if we're not running with FSDP we can apply smoothing directly
                for layer in balance_layers:
                    smooth(layer)
                smooth(smooth_layer)

            if self.apply_clip:
                clip_list = self._search_best_clip(
                    balance_layers=balance_layers,
                    balance_names=balance_names,
                    input_feat=inp,
                )

                _apply_clip(model, clip_list)

        # clear out allocated smoothing scales
        torch.cuda.empty_cache()

    def _compute_best_scale(
        self,
        x: torch.Tensor,
        w_mean: torch.Tensor,
        x_mean: torch.Tensor,
        module2inspect: torch.nn.Module,
        linears2scale: List[torch.nn.Linear],
        fp16_output: torch.Tensor,
    ):
        """
        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | pseudo_quantize_tensor(W * s)
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
            scales_view = scales.view(1, -1).to(device)

            # avoid scaling values that overflow
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1

            # Q(W * s)
            for fc in linears2scale:
                with align_module_device(fc):
                    fc.weight.mul_(scales_view)
                    update_offload_parameter(
                        fc,
                        "weight",
                        pseudo_quantize_tensor(
                            w=fc.weight.data,
                            symmetric=self.symmetric,
                            bit_width=self.bits,
                            group_size=self.group_size,
                        )[0]
                        / scales_view,
                    )

            # W * X
            int_w_output = self._forward_input_with_kwargs(
                module=module2inspect, inputs=x, input_kwargs=self.module_kwargs_
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
    ):
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

        # Compute the loss for each chunk
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

    def _set_module_kwargs(self, model, dataloader) -> None:
        _, modules = next(iter(get_layers("re:.*layers", model).items()))

        samples = [batch["input_ids"] for batch in dataloader]

        samples = torch.cat(samples, dim=0)

        inps = []
        layer_kwargs = {}

        best_device = "cuda"
        modules[0] = modules[0].to(best_device)
        # self.awq_model.move_embed(self.model, best_device)

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
            model(samples.to(next(model.parameters()).device))
        except ValueError:  # work with early exit
            pass
        modules[0] = modules[0].module  # restore

        # Update the layer kwargs with `prepare_inputs_for_generation` method
        # that takes care of everything to avoid unexpected errors.
        layer_kwargs |= model.prepare_inputs_for_generation(samples, **layer_kwargs)
        # Pop the input_ids as they are not needed at all.
        layer_kwargs.pop("input_ids")

        del samples
        inps = inps[0]

        torch.cuda.empty_cache()

        if layer_kwargs.get("attention_mask") is not None:
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to(
                best_device
            )

        self.module_kwargs_ = layer_kwargs

    def _forward_input_with_kwargs(
        self,
        module: Module,
        inputs: torch.Tensor,
        input_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Forward pass with input arguments

        :param module: module to run forward pass on
        :param inputs: input tensor to pass to the module
        :param input_kwargs: additional arguments to pass to the module
        :return: the first output tensor from the forward pass
        """
        kwargs = input_kwargs or self.module_kwargs_ or {}
        return tensor_forward_with_input_args(
            module=module,
            inputs=inputs,
            input_kwargs=kwargs,
        )[0]

    @torch.no_grad()
    def _search_best_clip(self, balance_layers, balance_names, input_feat):
        clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        for name, layer in zip(balance_names, balance_layers):
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in avoid_clipping]):
                continue

            max_val = self._compute_best_clip(layer.weight, input_feat)
            clip_list.append((name, max_val))

        return clip_list

    @torch.no_grad()
    def _compute_best_clip(
        self,
        w: torch.Tensor,
        input_feat: torch.Tensor,
        n_grid=20,
        max_shrink=0.5,
        n_sample_token=512,
    ):
        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = self.group_size if self.group_size > 0 else org_w_shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

        # Compute input feature step size (minimum 1)
        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size]

        w = w.reshape(org_w_shape[0], 1, -1, group_size)

        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64  # prevent OOM
        assert org_w_shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(org_w_shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = pseudo_quantize_tensor(
                    w=cur_w,
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

        return best_max_val.squeeze(1)


@torch.no_grad()
def _apply_clip(module, clip_list: Tuple[str, torch.Tensor]):
    """
    Apply clipping to the weights of the given module

    :post-condition: the weights of the module are clipped to the given maximum values
    :param module: module to apply clipping to
    :param clip_list: list of tuples containing the name of the layer and the maximum
        value to clip the weights to
    """
    for name, max_val in clip_list:
        _, layer = get_layer(target=name, module=module)
        assert isinstance(layer, torch.nn.Linear)
        with align_module_device(layer):
            max_val = max_val.to(layer.weight.device)
            org_shape = layer.weight.shape
            layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
            layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
            layer.weight.data = layer.weight.data.reshape(org_shape)
