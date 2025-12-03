import inspect
from itertools import product
from typing import Literal

import torch
from compressed_tensors.quantization import disable_quantization
from compressed_tensors.utils import (
    align_modules,
    get_execution_device,
    match_named_modules,
    update_offload_parameter,
)
from loguru import logger
from pydantic import ConfigDict, PrivateAttr, Field
from torch.nn import Module
from tqdm import tqdm
import tensorly as tl
from tensorly.decomposition import tensor_train

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.calibration import update_weight_zp_scale
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.utils.fsdp.helpers import get_fsdp_parent
from llmcompressor.utils.helpers import calibration_forward_context
from llmcompressor.utils.pytorch.module import get_layer_by_name

tl.set_backend("pytorch")

__all__ = ["TensorNetworkModifier"]


class TensorNetworkModifier(Modifier):
    """
    This experimental module converts all targeted linear layers to
    a tensor network factorization known as a Matrix Product Operator
    (MPO), similar to https://arxiv.org/abs/2305.06058. The paper claims
    that neural networks can be compressed using tensor networks with
    exponentially fewer variational parameters, without loss to accuracy.

    Reference: Section 4.3 of
      https://tensorly.org/dev/user_guide/tensor_decomposition.html

    Because this modifier manipulates the weights and structure of the model,
    it can only be used in one-shot and requires a calibration dataset.

    example recipe:
    ```yaml
    TensorizeModifier:
      ignore: ["lm_head"]
      targets: ["Linear"]
    ```

    Lifecycle:
        - on_initialize
            - nothing
        - on_start
            - set up activation cache hooks to capture input activations
                to target layers
        - on sequential epoch end
            - apply tensorization to each target layer
                - consume cached activations across all batches
                    - clear cached activations as they are used
                - apply tensor-network factorization
                - train against calibration dataset
        - on_end
            - re-run logic of sequential epoch end (in case of basic pipeline)
            - remove activation hooks
        - on_finalize
            - clear captured activations

    :param ignore: list of layers to ignore, even if they match a regex in mappings.
        It should match the name of layers whose outputs are scaled to achieve
        smoothing (the second entry of the mappings list).
    :param targets: list of layer names to quantize if a scheme is provided. If unset,
        will default to ["Linear"] (i.e. all Linear layers will be targeted).
    :param offload_device: offload cached args to this device, which reduces memory
        requirements but requires more time to move data between cpu and execution
        device. Defaults to None, so cached args are not offloaded. Consider setting
        to torch.device("cpu") if you are encountering OOM errors
    :param num_blocks: Number of blocks to break target linear matrix into. Every
        target layer must have row and column size evenly divisible by n_blocks.
        Value must be an integer squared (e.g. 1, 4, 9, 16, ...)
        Defaults to 1.
    :param num_cores: Number of cores (also known as sites) in each resultant MPO.
        Defaults to 3.
    """

    # Allow arbitrary types because AWQMapping has fields of type torch.nn.Module
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    # User-provided vars (in addition to QuantizationMixin args)
    targets: str | list[str] = Field(default_factory=lambda: ["Linear"])
    ignore: list[str] = Field(default_factory=list)
    offload_device: torch.device | None = None
    num_blocks: int = 1
    num_cores: int = 3

    # Cache list of forward input args for each parent module, one dict for each batch
    _target_args_cache: dict[tuple[str, Module], IntermediatesCache] = PrivateAttr(
        default_factory=dict
    )

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        :param state: state to run TensorNetworkModifier on
        :return: True on a successful run, False otherwise
        """

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

        # register caching hooks
        self._setup_activation_cache_hooks()

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        elif event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            # Run smoothing in case of sequential pipeline
            self._tensorize(state.model)

        elif event.type_ == EventType.CALIBRATION_EPOCH_END:
            # Run smoothing in case of basic pipeline
            self._tensorize(state.model)

            if not self.ended_:
                self.on_end(state, None)

    def on_end(self, state: State, event: Event, **kwargs):
        """
        Finish calibrating by setting scales and zero-points,
         removing observers and calibration hooks
        """
        self._assert_all_activations_consumed()

        self.ended_ = True

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

        self._target_args_cache.clear()

        return True

    def _setup_activation_cache_hooks(self, model: Module) -> None:
        """
        Attach a forward hook to each target layer we want to tensorize
        """

        def cache_target_kwargs_hook(
            module: torch.nn.Module,
            args: tuple[torch.Tensor, ...],
            kwargs,
        ):
            values = inspect.signature(module.forward).bind(*args, **kwargs)
            self._target_args_cache[module].append(values.arguments)

        for name, module in match_named_modules(model, self.targets, self.ignore):
            self._target_args_cache[(name, module)] = IntermediatesCache(
                None,
                self.offload_device,
            )
            self.register_hook(
                module,
                cache_target_kwargs_hook,
                "forward_pre",
                with_kwargs=True,
            )

    @torch.no_grad()
    def _tensorize(self, model: Module) -> None:
        """
        Tensorize any linear layer in model for which an entry in the cache
        exists. Tensor Network decomposition is applied, and the layer is
        retrained to match the original linear layer's mapping for the given
        calibration dataset

        :param model: model to apply smoothing to
        """
        # NOTE: When using SequentialPipeline, not all the targeted layers
        # will have cached activations in the segment being udpated
        for name, module in tqdm(self._target_args_cache.keys(), desc="Tensorizing"):
            with (
                align_modules([module]),
                calibration_forward_context(model),
                HooksMixin.disable_hooks(),
            ):
                # [STEP 1]: Compute output of module
                fp16_outputs = self._run_samples(module)
                if len(fp16_outputs) == 0 or all(f.numel() == 0 for f in fp16_outputs):
                    logger.info(
                        f"Skipping layer {name}, no activations "
                        "found to scale. This can occasionally occur in MoE models "
                        "when certain experts are not activated by calibration samples."
                    )
                    del self._target_args_cache[(name, module)]
                    continue
                if not all(
                    [fp16_output.isfinite().all() for fp16_output in fp16_outputs]
                ):
                    logger.warning(
                        f"Skipping layer {name}, NaN or inf "
                        "outputs found during forward pass of the module. "
                        "The model is either generating NaN output with provided "
                        "calibration data set, or the targets are incorrectly set "
                        "and modifying the model in undesired ways. "
                        "If you encounter this consistently, raise an issue at "
                        "https://github.com/vllm-project/llm-compressor/issues"
                    )
                    del self._target_args_cache[(name, module)]
                    continue

                # [STEP 2]: Tensorize
                factors = tensor_train(
                    module.weight.data,
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

    def _run_samples(self, module: Module) -> list[torch.Tensor]:
        outputs = [
            module(**batch_kwargs) for batch_kwargs in self._parent_args_cache[module]
        ]
        return [
            # If Tuple, assume that first argument is the input
            output[0] if isinstance(output, tuple) else output
            for output in outputs
        ]

    def _compute_best_scale(
        self,
        x_mean: torch.Tensor,
        w_mean: torch.Tensor,
        parent_module: torch.nn.Module,
        linears2scale: list[torch.nn.Linear],
        fp16_outputs: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | _pseudo_quantize_tensor(W * s)
        X: inputs from calib dataset    | X
        W: original weights in FP16     | layer
        s: per channel scaling factor   | s^-1 * X
        """
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

        match self.duo_scaling:
            # if self.duo_scaling is "both", perform half the grid search with
            # duo_scaling off and half with duo_scaling on
            case "both":
                n_grid = int(self.n_grid / 2)
                duo_scalings = [False, True]
            case _:
                n_grid = self.n_grid
                duo_scalings = [self.duo_scaling]
        for grid_idx, use_duo_scaling in product(range(n_grid), duo_scalings):
            # create new scales
            ratio = grid_idx / n_grid

            # NOTE: s^-1 * x is fused here, according to paper
            if use_duo_scaling:
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
        fp16_outputs: list[torch.Tensor],
        int_w_outputs: list[torch.Tensor],
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


def get_lowest_common_parent(names: list[str], module: Module) -> tuple[str, Module]:
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
