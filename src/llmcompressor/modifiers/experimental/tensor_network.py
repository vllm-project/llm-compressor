import inspect
from functools import reduce

import tensorly as tl
import torch
from compressed_tensors.utils import (
    align_modules,
    match_named_modules,
    update_offload_parameter,
)
from loguru import logger
from pydantic import ConfigDict, Field, PrivateAttr
from torch import nn
from tqdm import tqdm

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.experimental.tensorized_linear import (
    TensorizedLinear,
)
from llmcompressor.modifiers.experimental.block_tensorized_linear import (
    BlockTensorizedLinear,
)
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.utils.helpers import calibration_forward_context

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
    :param num_blocks: Optional block size to be used for BlockTensorizedLinear.
        If set, the Linear layer will be broken into square matrices of this size.
        If unset, TensorizedLinear will be used.
    :param num_cores: Number of cores (also known as sites) in each resultant MPO.
        Defaults to 3.
    :param rank: determines the number of parameters. The tensorized layer will
        have a total number of parameters equal to linear.weight.numel() * rank
        Should be in range (0.0, 1.0]. A value of 1.0 means no compression.
    :param batch_size: batch_size to use when updating gradient, which is often
        more memory intensive than forward passes. torch.einsum can be particularly
        memory intensive. If unset, defaults to batch_size of dataset passed.
    """

    # Allow arbitrary types because AWQMapping has fields of type torch.nn.Module
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    # User-provided vars (in addition to QuantizationMixin args)
    targets: str | list[str] = Field(default_factory=lambda: ["Linear"])
    ignore: list[str] = Field(default_factory=list)
    offload_device: torch.device | None = None
    block_size: int | None = None
    num_cores: int = 3
    rank: float = 0.5
    batch_size: int | None = None
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
        self._setup_target_args_cache_hooks(state.model)

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

    def _setup_target_args_cache_hooks(self, model: nn.Module) -> None:
        """
        Attach a forward hook to each targeted Linear layer we want to tensorize
        """

        def create_cache_target_inputs_outputs_hook(name: str):
            def cache_target_inputs_outputs_hook(
                module: nn.Module,
                args: tuple[torch.Tensor, ...],
                output: torch.Tensor,
            ):
                assert len(args) == 1, "linear layer can only have one input"
                if (name, module) not in self._target_args_cache:
                    self._target_args_cache[(name, module)] = IntermediatesCache(
                        None,
                        self.offload_device,
                    )
                self._target_args_cache[(name, module)].append(
                    {
                        "input": args[0].detach().clone(),
                        "output": output.detach().clone(),
                    }
                )

            return cache_target_inputs_outputs_hook

        for name, module in match_named_modules(model, self.targets, self.ignore):
            if not isinstance(module, nn.Linear):
                continue

            self.register_hook(
                module, create_cache_target_inputs_outputs_hook(name), "forward"
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

        for name, module in tqdm(
            # cast to list to cache in memory, as dict is modified during iteration
            list(self._target_args_cache.keys()),
            desc="Tensorizing",
        ):
            with (
                align_modules([module]),
                calibration_forward_context(model),
                HooksMixin.disable_hooks(),
            ):
                tensorized_linear = self._get_trained_tensorized_layer(name, module)

                # Replace linear layer with its tensorized_linear approximation
                parent = get_parent_of_model_by_name(model, name)
                leaf_name = name.split(".")[-1]
                setattr(parent, leaf_name, tensorized_linear)

        self._assert_all_activations_consumed()

    def _get_trained_tensorized_layer(
        self, name: str, linear: torch.nn.Linear
    ) -> TensorizedLinear | BlockTensorizedLinear:
        # create tensorized layer
        tensorized_linear = (
            TensorizedLinear.from_linear(
                linear, num_cores=self.num_cores, rank=self.rank
            )
            if self.block_size is None
            else BlockTensorizedLinear.from_linear(
                linear, self.block_size, num_cores=self.num_cores, rank=self.rank
            )
        ).to(linear.device)

        # Calculate parameter counts for logging
        original_params = sum(p.numel() for p in linear.parameters())
        tensorized_params = tensorized_linear.num_params
        compression_ratio = (
            original_params / tensorized_params if tensorized_params > 0 else 0
        )

        logger.info(
            f"Layer {name} - Original params: {original_params:.2e}, "
            f"Tensorized params: {tensorized_params:.2e}, "
            f"Compression ratio: {compression_ratio:.2f}x"
        )

        # cache all training data on device
        batch_inputs_and_outputs = self.get_batch_inputs_outputs(
            self._target_args_cache[(name, linear)], self.batch_size
        )
        del self._target_args_cache[(name, linear)]

        tensorized_linear = tensorized_linear.to(batch_inputs_and_outputs[0][0].device)

        # re-enable grad for training (parent _tensorize has @torch.no_grad())
        with torch.enable_grad():
            # Setup optimizer and loss function
            optimizer = torch.optim.Adam(tensorized_linear.parameters(), lr=1e-5)
            criterion = nn.MSELoss()

            # Training loop with early stopping
            num_epochs = 100
            patience = 10  # Stop if no improvement for this many epochs
            min_frac_delta = (
                1e-2  # Minimum improvement over previous loss to be considered progress
            )
            min_loss_threshold = 1e-6  # Stop if loss is already this good
            loss_history = []

            best_loss = float("inf")
            epochs_without_improvement = 0

            for epoch in (pbar := tqdm(range(num_epochs))):
                total_loss = 0.0
                num_batches = 0

                for batch_input, dense_output in batch_inputs_and_outputs:
                    # Zero gradients
                    optimizer.zero_grad()

                    # Forward pass through tensorized layer
                    tensorized_batch_output = tensorized_linear(batch_input)

                    # Compute loss against original dense output
                    # TODO try (1-F.cosine_similarity(..)) as criterion instead
                    loss = criterion(tensorized_batch_output, dense_output)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                loss = total_loss / num_batches if num_batches > 0 else 0.0

                # Early stopping check
                if len(loss_history) > 0:
                    if loss < best_loss:
                        best_loss = loss
                    if loss < loss_history[-1] * (1.0 - min_frac_delta):
                        # Significant improvement
                        epochs_without_improvement = 0
                else:
                    # No significant improvement
                    epochs_without_improvement += 1

                pbar.set_description(
                    f"Layer {name} - Epoch {epoch}/{num_epochs}, "
                    f"Avg Loss: {avg_loss:.2e}, Best Loss: {best_loss:.2e}"
                )

                loss_history.append(avg_loss)
                # Check early stopping conditions
                if avg_loss < min_loss_threshold:
                    logger.info(
                        f"Layer {name} - Early stopping: loss {avg_loss:.2e} "
                        f"below threshold {min_loss_threshold:.2e} at epoch {epoch}"
                    )
                    break

                if epochs_without_improvement >= patience:
                    logger.info(
                        f"Layer {name} - Early stopping: no improvement for "
                        f"{patience} epochs at epoch {epoch}. Best loss: {best_loss:.2e}"
                    )
                    break

        return tensorized_linear

    @staticmethod
    @staticmethod
    def get_batch_inputs_outputs(
        intermediates: IntermediatesCache, batch_size: int | None = None
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Fully onloads a batched dataset from IntermediatesCache,
        if necessary by re-batching to the necessary batch_size.

        This implementation avoids torch.cat to save memory - it processes
        cached batches incrementally.

        Args:
            intermediates: Cache containing input/output pairs
            batch_size: Optional target batch size for re-batching. If None,
                returns data in original batch sizes.

        Returns:
            List of tuples of (input_batches, output_batches) with appropriate
            batch_size.
        """
        # If no re-batching needed, just collect original batches
        if batch_size is None:
            input_and_output_batches = []
            for batch in intermediates.iter():
                batch_input, batch_output = batch.values()
                input_and_output_batches.append((batch_input, batch_output))
            return input_and_output_batches

        # Re-batch efficiently by appending chunks instead of individual samples
        input_and_output_batches = []

        # Accumulator for partial batches (stores chunks, not individual samples)
        current_inputs = []
        current_outputs = []
        current_count = 0

        for batch in intermediates.iter():
            batch_input, batch_output = batch.values()
            num_samples = batch_input.shape[0]
            start_idx = 0

            while start_idx < num_samples:
                # How many samples do we need to fill current batch?
                samples_needed = batch_size - current_count
                samples_available = num_samples - start_idx

                # Take as many as we can (min of needed and available)
                chunk_size = min(samples_needed, samples_available)
                end_idx = start_idx + chunk_size

                # Append chunk (not individual samples!)
                current_inputs.append(batch_input[start_idx:end_idx])
                current_outputs.append(batch_output[start_idx:end_idx])
                current_count += chunk_size

                # If we filled a batch, emit it and reset
                if current_count >= batch_size:
                    input_and_output_batches.append(
                        (
                            torch.cat(current_inputs, dim=0),
                            torch.cat(current_outputs, dim=0),
                        )
                    )
                    current_inputs = []
                    current_outputs = []
                    current_count = 0

                start_idx = end_idx

        # Don't forget the last partial batch if it exists
        if len(current_inputs) > 0:
            input_and_output_batches.append(
                (torch.cat(current_inputs, dim=0)), torch.cat(current_outputs, dim=0)
            )

        return input_and_output_batches

    def _assert_all_activations_consumed(self):
        """
        Confirm all activations have been consumed
        If not, something has gone wrong
        """
        if len(self._target_args_cache) != 0:
            raise RuntimeError("Some cached activations were not used")


def get_parent_of_model_by_name(
    model: torch.nn.Module, module_name: str
) -> torch.nn.Module:
    """
    Retrieves the parent module of a nested module specified by its full name.


    :param model: The root of the module tree
    :param module_name: The dot-separated name of the nested module, for example
        'model.layers.0.self_attn.q_proj'

    :return: The parent module, or None if the module is the root or not found.
    """
    if "." not in module_name:
        # The module is a direct child of the root, so the root is its parent
        return model

    # Split the name into parts and get the name of the parent
    parts = module_name.split(".")
    parent_name = ".".join(parts[:-1])

    # try:
    # Use reduce and getattr to traverse to the parent module
    parent_module = reduce(getattr, parent_name.split("."), model)
    # Verify that the retrieved object is indeed an nn.Module
    if isinstance(parent_module, nn.Module):
        return parent_module
    else:
        return None
    # except AttributeError:
    #     # If any part of the path is invalid, the parent is not found
    #     return None
