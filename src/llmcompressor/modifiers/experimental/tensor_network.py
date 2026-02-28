import inspect
from functools import reduce

import tensorly as tl
import torch
import torch.nn.functional as F
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

        # cast to list to cache in memory, as dict is modified during iteration
        for name, module in list(self._target_args_cache.keys()):
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

                del self._target_args_cache[(name, module)]

        self._assert_all_activations_consumed()

    def _get_trained_tensorized_layer(
        self,
        name: str,
        linear: torch.nn.Linear,
        target_sqnr=30.0,  # Reduce if target sqnr met
        rank_reduction_factor=None,  # 0.05,  # Reduce rank by 5% each iteration (num_params ~ rank**2)
        energy_threshold=0.98,  # Preserve 99.99% of energy to keep more parameters
    ) -> TensorizedLinear | BlockTensorizedLinear:
        """
        Create a tensorized equivalent of the input Linear matrix with adaptive rank pruning.

        Strategy:
        1. Start with rank=1.0 (full rank, lossless reconstruction)
        2. Train until sqnr >= threshold target_sqnr
        3. If achieved, re-decompose with reduced rank
        4. Retrain and check if threshold can still be maintained
        5. Repeat until threshold can no longer be met
        6. Return the maximally compressed layer that meets threshold

        This adaptively finds the minimal rank needed for acceptable reconstruction.
        """

        # Start with full rank for lossless reconstruction
        best_tensorized_linear = tensorized_linear = (
            TensorizedLinear.from_linear(linear, num_cores=self.num_cores, rank=2.0)
            if self.block_size is None
            else BlockTensorizedLinear.from_linear(
                linear, self.block_size, num_cores=self.num_cores, rank=2.0
            )
        ).to(linear.weight.device)

        total_num_epochs = 0
        while True:
            # Train this rank level
            final_sqnr, total_num_epochs = self._train_tensorized_layer(
                tensorized_linear,
                name,
                linear,
                target_sqnr=target_sqnr,
                total_num_epochs=total_num_epochs,
            )

            if final_sqnr >= target_sqnr:
                best_tensorized_linear = tensorized_linear

                # Compute input covariance matrix for data-aware truncation (V-SVD)
                input_cov_sqrt = None
                try:
                    # Accumulate covariance matrix across batches to handle variable-sized inputs
                    cov_accumulator = None
                    total_samples = 0
                    for batch in self._target_args_cache[(name, linear)].iter():
                        batch_input = batch["input"]
                        # Flatten batch dimensions: (..., in_features) -> (batch_size, in_features)
                        batch_flat = batch_input.reshape(-1, batch_input.shape[-1]).to(
                            torch.float64
                        )
                        # Accumulate X^T X
                        if cov_accumulator is None:
                            cov_accumulator = batch_flat.T @ batch_flat
                        else:
                            cov_accumulator += batch_flat.T @ batch_flat
                        total_samples += batch_flat.shape[0]

                    if cov_accumulator is not None and total_samples > 0:
                        # Compute covariance: Σ_X = X^T X / n
                        input_cov = cov_accumulator / total_samples
                        # Compute matrix square root: √Σ_X
                        eigenvalues, eigenvectors = torch.linalg.eigh(input_cov)
                        eigenvalues = torch.clamp(eigenvalues, min=0.0)
                        input_cov_sqrt = (
                            eigenvectors
                            @ torch.diag(torch.sqrt(eigenvalues))
                            @ eigenvectors.T
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to compute input covariance for {name}: {e}"
                    )
                    input_cov_sqrt = None

                # Preserve learned params -- truncate current layer instead of recreating
                # Use energy-preserving truncation (rank_reduction_factor=None) to maintain 99% of energy
                # Use data-aware truncation (V-SVD) with input covariance
                tensorized_linear = tensorized_linear.truncate_ranks(
                    rank_reduction_factor=None,
                    energy_threshold=energy_threshold,
                    input_cov_sqrt=input_cov_sqrt,
                )

            else:
                break

        # Return best achieved
        return best_tensorized_linear

    # re-enable grad for training (parent _tensorize has @torch.no_grad())
    @torch.enable_grad()
    def _train_tensorized_layer(
        self,
        tensorized_linear: TensorizedLinear | BlockTensorizedLinear,
        name: str,
        linear: torch.nn.Linear,
        target_sqnr=30.0,
        total_num_epochs=0,
    ) -> tuple[float, int]:
        """
        Train a tensorized layer and return final average cosine similarity.

        Early exits if target_sqnr achieved.

        Returns:
            Average sqnr across all batches after training
            Accumulated total_num_epochs
        """
        original_params = sum(p.numel() for p in linear.parameters())
        tensorized_params = tensorized_linear.num_params
        compression_pct = (
            f"{compression_pct:.1%}"
            if (compression_pct := (tensorized_params / original_params)) < 100.99
            else "100 %"
        )

        # SGD with momentum uses ~2x memory vs SGD
        # Adam uses ~3x memory vs SGD
        # optimizer = torch.optim.SGD(
        #     tensorized_linear.parameters(),
        #     lr=1e-4,  # Higher LR than Adam since SGD needs larger steps
        #     momentum=0.9,  # Helps convergence with cosine similarity loss
        # )
        optimizer = torch.optim.Adam(
            tensorized_linear.parameters(),
            lr=1e-4,
        )

        # Training loop with early stopping
        max_total_num_epochs = 100
        # Stop if no improvement for this many epochs
        patience = 3
        # Minimum improvement over previous loss to be considered progress
        min_frac_delta = 1e-2
        loss_history = []

        best_loss = float("inf")
        epochs_without_improvement = 0

        pbar = tqdm(
            total=max_total_num_epochs,
            initial=total_num_epochs,
            desc=f"{name} | ",
        )
        for epoch in range(max_total_num_epochs - total_num_epochs):
            total_loss = 0.0
            num_batches = 0
            cosine_similarities = []
            sqnrs = []
            mses = []
            matching_signs = []
            sign_norms = []

            # for batch_input, dense_output in self.get_batch_inputs_outputs(
            #     self._target_args_cache[(name, linear)], self.batch_size
            # ):
            for batch in self._target_args_cache[(name, linear)].iter():
                batch_input, dense_output = batch.values()

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass through tensorized layer
                tensorized_batch_output = tensorized_linear(batch_input)

                # Compute hybrid loss: MSE + (1 - cosine_similarity)
                # MSE forces the model to get the physical scale right
                # Cosine similarity encourages vectors to align in the same direction
                mse = F.mse_loss(tensorized_batch_output, dense_output)
                loss = mse  # * mse_weight + cosine_loss * 1e-3

                loss.backward()
                optimizer.step()

                cosine_similarities.append(
                    F.cosine_similarity(
                        tensorized_batch_output.detach(), dense_output.detach(), dim=-1
                    ).mean()
                )
                sqnrs.append(
                    compute_sqnr(
                        dense_output.detach(), tensorized_batch_output.detach()
                    )
                )
                mses.append(mse.detach().item())
                matching_signs.append(
                    count_matching_signs(
                        dense_output.detach(), tensorized_batch_output.detach()
                    )
                )
                sign_norms.append(
                    compute_sign_norm(
                        dense_output.detach(), tensorized_batch_output.detach()
                    )
                )
                total_loss += loss.item()
                num_batches += 1

            epoch_loss = total_loss / num_batches

            # Calculate average cosine similarity for this epoch
            final_avg_sqnr = sum(sqnrs) / len(sqnrs) if sqnrs else 0.0

            # Early stopping check
            if len(loss_history) > 0:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                if epoch_loss < (loss_history[-1] * (1.0 - min_frac_delta)):
                    # Significant improvement
                    epochs_without_improvement = 0
                else:
                    # No significant improvement
                    epochs_without_improvement += 1
            loss_history.append(epoch_loss)

            pbar.set_description(
                f"{name} | # params: "
                f"{tensorized_params:.1e} ({compression_pct}) | "
                f"mse: {sum(mses)/len(mses):.2e} | "
                f"sqnr: {final_avg_sqnr:.2e} | "
                f"signs: {sum(matching_signs)/len(matching_signs):.3f} | "
                f"sign norm: {sum(sign_norms)/len(sign_norms):.3f} | "
                f"cos(): {sum(cosine_similarities)/len(cosine_similarities):.3f}"
            )
            pbar.update(1)
            total_num_epochs += 1

            # Check early stopping conditions
            if (
                epoch > 2 and final_avg_sqnr > target_sqnr
            ) or epochs_without_improvement >= patience:
                break

        # Return final average sqnr
        return final_avg_sqnr, total_num_epochs

    @staticmethod
    def get_batch_inputs_outputs(
        intermediates: IntermediatesCache, batch_size: int | None = None
    ):
        """
        Generator that yields batches from IntermediatesCache one at a time.
        This is memory-efficient as each batch can be garbage collected before
        the next one is loaded.

        Args:
            intermediates: Cache containing input/output pairs
            batch_size: Optional target batch size for re-batching. If None,
                yields data in original batch sizes.

        Yields:
            Tuples of (input_batch, output_batch) with appropriate batch_size
        """
        # If no re-batching needed, just yield original batches
        if batch_size is None:
            for batch in intermediates.iter():
                batch_input, batch_output = batch.values()
                yield (batch_input, batch_output)
            return

        # Re-batch efficiently by appending chunks instead of individual samples
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

                # If we filled a batch, yield it and reset
                if current_count == batch_size:
                    batch_inputs_concat = torch.cat(current_inputs, dim=0)
                    batch_outputs_concat = torch.cat(current_outputs, dim=0)
                    yield (batch_inputs_concat, batch_outputs_concat)

                    # Explicitly delete to encourage garbage collection
                    del batch_inputs_concat, batch_outputs_concat
                    current_inputs = []
                    current_outputs = []
                    current_count = 0

                start_idx = end_idx

        # Don't forget the last partial batch if it exists
        if current_count > 0:
            yield (torch.cat(current_inputs, dim=0), torch.cat(current_outputs, dim=0))

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


# https://github.com/pytorch/pytorch/blob/v2.10.0/torch/ao/ns/fx/utils.py#L470
# def compute_sqnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     """
#     Computes the SQNR between `x` and `y`.

#     Args:
#         x: Tensor or tuple of tensors
#         y: Tensor or tuple of tensors

#     Return:
#         float or tuple of floats
#     """
#     Ps = torch.norm(x)
#     Pn = torch.norm(x - y)
#     return 20 * torch.log10(Ps / Pn)


def compute_sqnr(original_output: torch.Tensor, tensorized_output: torch.Tensor):
    """
    Calculates SQNR in dB using activation tensors directly.

    Args:
        original_output (torch.Tensor): The output from the original nn.Linear layer.
        tensorized_output (torch.Tensor): The output from your custom MPO layer.
    """
    # Ensure we are working with floats for precision
    y_true = original_output.detach().float()
    y_pred = tensorized_output.detach().float()

    # Signal Power: The variance of the original activations
    # Represents the 'strength' of the information signal
    signal_power = torch.var(y_true)

    # Noise Power: The Mean Squared Error of the approximation
    mse_noise = torch.mean((y_true - y_pred) ** 2)

    # SQNR calculation (10 * log10 of the power ratio)
    # 1e-10 added to denominator to prevent division by zero or log(0)
    sqnr_linear = signal_power / (mse_noise + 1e-10)
    sqnr_db = 10 * torch.log10(sqnr_linear)

    return sqnr_db.item()


def count_matching_signs(a: torch.Tensor, b: torch.Tensor):
    # Get signs: 1 for positive, -1 for negative, 0 for zero
    sign_a = torch.sign(a)
    sign_b = torch.sign(b)

    # Compare and sum the boolean matches
    match_pct = (sign_a == sign_b).sum() / sign_a.numel()

    return match_pct.item()


def compute_sign_norm(a: torch.Tensor, b: torch.Tensor):
    # Identify indices where signs don't match
    mismatch_mask = torch.sign(a) != torch.sign(b)

    # Calculate how much energy those specific indices hold
    mismatch_magnitude = torch.norm(a[mismatch_mask])
    total_magnitude = torch.norm(a)

    return mismatch_magnitude / total_magnitude
