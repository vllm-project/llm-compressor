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
from torch import nn
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
from llmcompressor.modifiers.experimental.tensorized_linear import TensorizedLinear

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
    _target_args_cache: dict[tuple[str, nn.Linear], IntermediatesCache] = PrivateAttr(
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

    def _setup_activation_cache_hooks(self, model: nn.Module) -> None:
        """
        Attach a forward hook to each targeted Linear layer we want to tensorize
        """

        def cache_target_kwargs_hook(
            module: nn.Module,
            args: tuple[torch.Tensor, ...],
            kwargs,
        ):
            values = inspect.signature(module.forward).bind(*args, **kwargs)
            self._target_args_cache[module].append(values.arguments)

        for name, module in match_named_modules(model, self.targets, self.ignore):
            if not isinstance(module, nn.Linear):
                continue
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
    def _tensorize(self, model: nn.Module) -> None:
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
                dense_outputs = self._run_samples(module)
                if len(dense_outputs) == 0 or all(
                    f.numel() == 0 for f in dense_outputs
                ):
                    logger.info(
                        f"Skipping layer {name}, no activations "
                        "found to scale. This can occasionally occur in MoE models "
                        "when certain experts are not activated by calibration samples."
                    )
                    del self._target_args_cache[(name, module)]
                    continue
                if not all(
                    [dense_output.isfinite().all() for dense_output in dense_outputs]
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
                tensorized_linear = TensorizedLinear.from_linear(module, rank=2)

                # [STEP 3]: Retrain
                # TODO train tensorize_linear against dense_outputs w/ MSELoss

                # [STEP 4]: Apply to module
                # TODO use TensorizedLinear with einsum string
                @torch.no_grad()
                def _apply_tensorized(module: nn.Linear):
                    update_offload_parameter(
                        module,
                        "weight",
                        tensorized_linear.to_matrix(),
                    )

                _apply_tensorized(module)

                # remove caches needed to smooth this mapping
                del self._target_args_cache[(name, module)]

        self._assert_all_activations_consumed()

    def _run_samples(self, module: nn.Module) -> list[torch.Tensor]:
        outputs = [
            module(**batch_kwargs) for batch_kwargs in self._parent_args_cache[module]
        ]
        return [
            # If Tuple, assume that first argument is the input
            output[0] if isinstance(output, tuple) else output
            for output in outputs
        ]

    @torch.no_grad()
    def _compute_loss(
        self,
        dense_outputs: list[torch.Tensor],
        tensorized_outputs: list[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        loss = 0.0
        num_elements = 0

        # Compute the MSE loss for each batch
        for dense_batch, tensorized_batch in zip(dense_outputs, tensorized_outputs):
            loss += torch.nn.functional.mse_loss(
                dense_batch, tensorized_batch.to(dense_batch.device)
            ).item()
            num_elements += dense_batch.numel()

        # Normalize the loss by the total number of elements
        loss /= num_elements

        return loss

    def _assert_all_activations_consumed(self):
        """
        Confirm all activations have been consumed
        If not, something has gone wrong
        """
        if len(self._target_args_cache) != 0:
            raise RuntimeError("Some cached activations were not used")
