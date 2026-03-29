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
from llmcompressor.modifiers.experimental.adtn_linear import (
    ADTNLinear,
    ColumnSparseLinear,
    StackedColumnSparseLinear,
)
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
    # Method: "tensorized", "adtn", "stacked_lowrank", "column_sparse", or "stacked_column_sparse"
    method: str = "stacked_column_sparse"
    # Cache list of forward input args for each parent module, one dict for each batch
    _target_args_cache: dict[tuple[str, nn.Linear], IntermediatesCache] = PrivateAttr(
        default_factory=dict
    )

    # only reduce if this signal-to-noise ratio is reached (in dB)
    # Gemini:
    #    > 40 dB	Pristine	Identical to FP16/BF16.
    # 30 - 40 dB	Safe	Logic and grammar remain intact.
    # 20 - 30 dB	Risky	Noticeable drop in benchmarks; "stuttering" outputs.
    target_snr_db: float = (40.0,)
    # When truncating, presereve this percentage of the total energy
    energy_threshold: float = 0.975  # 97.5%

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
                if self.method == "stacked_lowrank":
                    tensorized_linear = self._get_stacked_lowrank_linear(name, module)
                elif self.method == "adtn":
                    tensorized_linear = self._get_adtn_linear(name, module)
                elif self.method == "column_sparse":
                    tensorized_linear = self._get_column_sparse_linear(name, module)
                elif self.method == "stacked_column_sparse":
                    tensorized_linear = self._get_stacked_column_sparse_linear(name, module)
                else:  # tensorized
                    tensorized_linear = self._get_trained_tensorized_layer(name, module)

                # Replace linear layer with its tensorized_linear approximation
                parent = get_parent_of_model_by_name(model, name)
                leaf_name = name.split(".")[-1]
                setattr(parent, leaf_name, tensorized_linear)

                del self._target_args_cache[(name, module)]

        self._assert_all_activations_consumed()

    def _collect_input_activations(
        self, name: str, linear: torch.nn.Linear
    ) -> torch.Tensor:
        """
        Collect and concatenate all input activations from the cache for a given layer.

        Args:
            name: Name of the layer
            linear: Linear layer module

        Returns:
            Tensor of shape (num_samples, in_features) containing all input activations
        """
        input_batches = []
        for batch in self._target_args_cache[(name, linear)].iter():
            batch_input = batch["input"]
            # Flatten all batch dimensions: (..., in_features) -> (batch_total, in_features)
            batch_flat = batch_input.reshape(-1, batch_input.shape[-1])
            input_batches.append(batch_flat)

        # Concatenate all batches
        all_inputs = torch.cat(input_batches, dim=0)
        return all_inputs

    def _collect_output_activations(
        self, name: str, linear: torch.nn.Linear
    ) -> torch.Tensor:
        """
        Collect and concatenate all output activations from the cache for a given layer.

        Args:
            name: Name of the layer
            linear: Linear layer module

        Returns:
            Tensor of shape (num_samples, out_features) containing all output activations
        """
        output_batches = []
        for batch in self._target_args_cache[(name, linear)].iter():
            batch_output = batch["output"]
            # Flatten all batch dimensions: (..., out_features) -> (batch_total, out_features)
            batch_flat = batch_output.reshape(-1, batch_output.shape[-1])
            output_batches.append(batch_flat)

        # Concatenate all batches
        all_outputs = torch.cat(output_batches, dim=0)
        return all_outputs

    def _get_adtn_linear(
        self,
        name: str,
        linear: torch.nn.Linear,
        group_size: int = 64,
        max_sublayers=10,
    ) -> ADTNLinear:
        """
        Create an ADTN equivalent of the input Linear matrix that matches the
        target inputs/outputs

        Strategy:
        1. Determine which inputs are most correlated with one another
        2. Permute inputs so the most correlated inputs are grouped together
        3. Create ADTN sublayer which mimics the original linear operation as much as
            possible. The ADTN sublayer performs smaller matrix multiplications each
            group of correlated inputs, i.e. one matrix operation on the first group of
            inputs, of size group_size (defaults to 64), another matrix operation on
            the second group of inputs, and so on. Rather than being trained, each
            matrix is calculated analytically with the ordinary least squares solution,
            (i.e. with `torch.linalg.lstsq(X, Y).solution` for input and output
            activations X and Y).
        4. Add the sublayer to ADTNLinear, printing out the signal-to-noise ratio
            achieved and number of parameters in ADTNLinear.
        5. Repeat Steps 1-4 to add additional sublayers to ADTNLinear until the signal-
            to-noise ratio is higher than target_snr_threshold_db (defaults to 40 dB).
        6. Returns ADTNLinear
        """
        from llmcompressor.modifiers.experimental.adtn_linear import ADTNSublayer

        # Collect all input and output activations
        input_activations = self._collect_input_activations(name, linear)
        output_activations = self._collect_output_activations(name, linear)

        # Initialize ADTN with empty sublayers
        adtn = ADTNLinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            sublayers=[],
            dtype=linear.weight.dtype,
        )

        for sublayer_idx in tqdm(range(max_sublayers), desc=name):
            # Step 1: Compute global residual (what ADTN hasn't explained yet)
            with torch.no_grad():
                if sublayer_idx == 0:
                    global_residual = output_activations.clone()
                else:
                    current_approx = adtn(input_activations)
                    global_residual = output_activations - current_approx

            # Step 2: Determine permutation for this sublayer
            # Each sublayer needs a DIFFERENT permutation so that connections missed by
            # the block-diagonal structure in one sublayer can be captured by another
            if sublayer_idx == 0:
                # First sublayer: use spectral reordering based on input correlation
                input_perm = ADTNLinear._spectral_reordering(
                    input_activations, group_size
                )
            else:
                # Subsequent sublayers: rotate permutation to create different block structure
                # Shift by group_size/2 to maximize coverage of new connections
                shift = (sublayer_idx * (group_size // 2)) % linear.in_features
                base_perm = ADTNLinear._spectral_reordering(
                    input_activations, group_size
                )
                input_perm = torch.roll(base_perm, shifts=shift)

            # Permute the inputs
            input_permuted = input_activations[:, input_perm]

            # Step 3: Create ADTN sublayer with concatenation architecture
            # Each group fits to a SLICE of the global residual (out_features/num_groups)
            num_groups = (linear.in_features + group_size - 1) // group_size
            group_out_features = linear.out_features // num_groups
            group_linears = []

            for group_idx in range(num_groups):
                # Input slice for this group
                in_start_idx = group_idx * group_size
                in_end_idx = min((group_idx + 1) * group_size, linear.in_features)
                actual_group_size = in_end_idx - in_start_idx

                # Output slice for this group
                out_start_idx = group_idx * group_out_features
                out_end_idx = min(
                    (group_idx + 1) * group_out_features, linear.out_features
                )
                actual_out_features = out_end_idx - out_start_idx

                # Extract group inputs
                X_group = input_permuted[:, in_start_idx:in_end_idx]

                # Fit to the corresponding OUTPUT SLICE of the global residual
                Y_group_slice = global_residual[:, out_start_idx:out_end_idx]

                # Solve OLS: X @ W = Y  =>  W = lstsq(X, Y)
                solution = torch.linalg.lstsq(
                    X_group.float(), Y_group_slice.float()
                ).solution

                # Create linear layer with OLS solution (outputs slice, not full output)
                group_linear = nn.Linear(
                    actual_group_size, actual_out_features, bias=False
                )
                group_linear.weight.data = solution.T.to(linear.weight.dtype)
                group_linears.append(group_linear)

            # Create sublayer
            sublayer = ADTNSublayer(
                in_features=linear.in_features,
                out_features=linear.out_features,
                linears=group_linears,
                input_perm=input_perm,
            )

            # Step 4: Add sublayer and compute SNR
            adtn.append_sublayer(sublayer)

            # Compute current approximation by applying full ADTN to original inputs
            with torch.no_grad():
                current_approx = adtn(input_activations)

            # Compute SNR
            current_snr = compute_snr(output_activations, current_approx)

            # Print progress
            logger.info(
                f"{name} | Sublayer {sublayer_idx}: "
                f"SNR = {current_snr:.2f} dB, "
                f"Compression = {(adtn.num_params/linear.weight.numel()):.1%}"
            )

            # Step 5: Check if target SNR achieved
            if current_snr >= self.target_snr_db:
                break

        # Step 6: Return ADTNLinear
        return adtn.to(linear.weight.device)

    def _get_stacked_lowrank_linear(
        self,
        name: str,
        linear: torch.nn.Linear,
        target_snr_threshold_db: float = 40.0,
        max_layers: int = 10,
        rank_ratio: float = 0.3,
    ):
        """
        Create a StackedLowRankLinear by iteratively adding low-rank layers.

        Each layer is a low-rank factorization (U @ V) that fits the residual
        from previous layers. This achieves parameter reduction while maintaining
        cross-feature interactions (unlike block-diagonal).

        Args:
            name: Layer name
            linear: Original linear layer
            target_snr_threshold_db: Target SNR in dB (default: 40)
            max_layers: Maximum number of low-rank layers to stack
            rank_ratio: Rank as fraction of min(in_features, out_features)

        Returns:
            StackedLowRankLinear with multiple low-rank layers
        """
        from llmcompressor.modifiers.experimental.adtn_linear import (
            StackedLowRankLinear,
            LowRankLayer,
        )

        # Collect activations
        input_activations = self._collect_input_activations(name, linear)
        output_activations = self._collect_output_activations(name, linear)

        # Initialize stacked low-rank model
        stacked = StackedLowRankLinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            layers=[],
            dtype=linear.weight.dtype,
        )

        # Compute rank for each layer
        rank = int(rank_ratio * min(linear.in_features, linear.out_features))
        rank = max(rank, 32)  # Minimum rank

        for layer_idx in tqdm(range(max_layers), desc=f"{name} (low-rank)"):
            # Step 1: Compute residual
            with torch.no_grad():
                if layer_idx == 0:
                    residual = output_activations.clone()
                else:
                    current_approx = stacked(input_activations)
                    residual = output_activations - current_approx

            # Step 2: Low-rank approximation via SVD of OLS solution
            X = input_activations.float()
            Y = residual.float()

            # Solve OLS
            W_full = torch.linalg.lstsq(X, Y).solution  # (in_features, out_features)

            # SVD for low-rank approximation
            U_svd, S, Vh = torch.linalg.svd(W_full, full_matrices=False)

            # Truncate to rank
            U_svd = U_svd[:, :rank]  # (in_features, rank)
            S_trunc = S[:rank]  # (rank,)
            Vh_trunc = Vh[:rank, :]  # (rank, out_features)

            # Absorb singular values into V
            V_weighted = torch.diag(S_trunc) @ Vh_trunc  # (rank, out_features)

            # Create low-rank layer
            layer = LowRankLayer(
                in_features=linear.in_features,
                out_features=linear.out_features,
                rank=rank,
                dtype=linear.weight.dtype,
            )
            layer.U.weight.data = U_svd.T.to(linear.weight.dtype)  # (rank, in_features)
            layer.V.weight.data = V_weighted.to(
                linear.weight.dtype
            )  # (out_features, rank)

            # Add layer to stack
            stacked.append_layer(layer)

            # Step 3: Compute SNR
            with torch.no_grad():
                current_approx = stacked(input_activations)

            current_snr = compute_snr(output_activations, current_approx)

            # Log progress
            logger.info(
                f"{name} | Layer {layer_idx}: "
                f"SNR = {current_snr:.2f} dB, "
                f"Params = {stacked.num_params} "
                f"({100*stacked.num_params/(linear.weight.numel()):.1f}%)"
            )

            # Step 4: Check if target SNR achieved
            if current_snr >= target_snr_threshold_db:
                break

        return stacked.to(linear.weight.device)

    def _get_column_sparse_linear(
        self,
        name: str,
        linear: torch.nn.Linear,
        target_sparsity: float = 0.5,
        k_cols_per_iter: int = 32,
        auto_stack_on_low_snr: bool = True,
        min_acceptable_snr: float = 25.0,
    ) -> ColumnSparseLinear | StackedColumnSparseLinear:
        """
        Create a ColumnSparseLinear via OLS-based greedy column selection.

        If single-layer SNR is too low, automatically switches to stacked layers.

        Args:
            name: Layer name
            linear: Original linear layer
            target_sparsity: Fraction of columns to keep (default: 0.5 for 50% compression)
            k_cols_per_iter: Number of columns to add per iteration
            auto_stack_on_low_snr: If True, use stacked layers when single-layer SNR < min_acceptable_snr
            min_acceptable_snr: Minimum SNR for single layer (default: 25.0 dB)

        Returns:
            ColumnSparseLinear or StackedColumnSparseLinear
        """
        # Collect activations
        input_activations = self._collect_input_activations(name, linear)

        logger.info(
            f"{name} | Creating column-sparse approximation "
            f"(target: {target_sparsity:.1%} of columns)"
        )

        # Create single-layer column-sparse first
        column_sparse = ColumnSparseLinear.from_linear(
            linear=linear,
            input_activations=input_activations,
            target_sparsity=target_sparsity,
            k_cols_per_iter=k_cols_per_iter,
            target_snr_db=None,  # Don't stop early, get full target sparsity
        )

        # Compute final SNR for logging
        with torch.no_grad():
            output_activations = self._collect_output_activations(name, linear)
            approx_output = column_sparse(input_activations)
            single_layer_snr = compute_snr(output_activations, approx_output)

        num_selected = len(column_sparse.selected_columns)
        compression_ratio = num_selected / linear.in_features

        logger.info(
            f"{name} | Single-layer SNR = {single_layer_snr:.2f} dB, "
            f"Params = {column_sparse.num_params:,} ({100*column_sparse.num_params/linear.weight.numel():.1f}%)"
        )

        # Check if we should use stacked layers instead
        if auto_stack_on_low_snr and single_layer_snr < min_acceptable_snr:
            logger.info(
                f"{name} | SNR {single_layer_snr:.2f} dB < {min_acceptable_snr} dB threshold, "
                f"switching to stacked layers..."
            )

            # Calculate per-layer sparsity to maintain similar total params
            # Single layer: target_sparsity * in_features
            # 2 layers: 2 * per_layer_sparsity * in_features ≈ target_sparsity * in_features
            # per_layer_sparsity ≈ sqrt(target_sparsity)
            per_layer_sparsity = min(0.85, (target_sparsity ** 0.5) + 0.15)  # Add margin

            stacked = StackedColumnSparseLinear.from_linear(
                linear=linear,
                input_activations=input_activations,
                target_sparsity_per_layer=per_layer_sparsity,
                max_layers=3,
                target_snr_db=self.target_snr_db,
                k_cols_per_iter=k_cols_per_iter,
            )

            with torch.no_grad():
                stacked_output = stacked(input_activations)
                stacked_snr = compute_snr(output_activations, stacked_output)

            logger.info(
                f"{name} | Stacked column-sparse: {len(stacked.layers)} layers, "
                f"SNR = {stacked_snr:.2f} dB, "
                f"Params = {stacked.num_params:,} ({100*stacked.num_params/linear.weight.numel():.1f}%)"
            )

            return stacked.to(linear.weight.device)

        logger.info(
            f"{name} | Using single-layer (SNR {single_layer_snr:.2f} dB >= {min_acceptable_snr} dB)"
        )
        return column_sparse.to(linear.weight.device)

    def _get_stacked_column_sparse_linear(
        self,
        name: str,
        linear: torch.nn.Linear,
        target_sparsity_per_layer: float = 0.7,
        max_layers: int = 5,
        k_cols_per_iter: int = 32,
    ) -> StackedColumnSparseLinear:
        """
        Create a StackedColumnSparseLinear via iterative residual fitting.

        Stacks multiple column-sparse layers, each fitting the residual from
        previous layers. This achieves higher SNR than single layer with
        similar total parameter count.

        Example: 2 layers @ 70% sparsity = 0.49x params, much higher SNR

        Args:
            name: Layer name
            linear: Original linear layer
            target_sparsity_per_layer: Fraction of columns per layer (default: 0.7)
            max_layers: Maximum layers to stack (default: 5)
            k_cols_per_iter: Columns to add per iteration

        Returns:
            StackedColumnSparseLinear with multiple layers
        """
        # Collect activations
        input_activations = self._collect_input_activations(name, linear)

        logger.info(
            f"{name} | Creating stacked column-sparse approximation "
            f"(target: {target_sparsity_per_layer:.1%} per layer, max {max_layers} layers)"
        )

        # Create stacked column-sparse via iterative residual fitting
        stacked = StackedColumnSparseLinear.from_linear(
            linear=linear,
            input_activations=input_activations,
            target_sparsity_per_layer=target_sparsity_per_layer,
            max_layers=max_layers,
            target_snr_db=self.target_snr_db,
            k_cols_per_iter=k_cols_per_iter,
        )

        # Compute final SNR for logging
        with torch.no_grad():
            output_activations = self._collect_output_activations(name, linear)
            approx_output = stacked(input_activations)
            final_snr = compute_snr(output_activations, approx_output)

        total_params = stacked.num_params
        param_ratio = total_params / linear.weight.numel()

        logger.info(
            f"{name} | Stacked column-sparse created: "
            f"{len(stacked.layers)} layers, "
            f"SNR = {final_snr:.2f} dB, "
            f"Params = {total_params:,} ({param_ratio:.1%})"
        )

        return stacked.to(linear.weight.device)

    def _get_trained_tensorized_layer(
        self,
        name: str,
        linear: torch.nn.Linear,
        rank_reduction_factor=None,  # 0.05,  # Reduce rank by 5% each iteration (num_params ~ rank**2)
    ) -> TensorizedLinear | BlockTensorizedLinear:
        """
        Create a tensorized equivalent of the input Linear matrix with adaptive rank pruning.

        Strategy:
        1. Start with rank=1.0 (full rank, lossless reconstruction)
        2. Train until snr >= threshold target_snr_db
        3. If achieved, re-decompose with reduced rank
        4. Retrain and check if threshold can still be maintained
        5. Repeat until threshold can no longer be met
        6. Return the maximally compressed layer that meets threshold

        This adaptively finds the minimal rank needed for acceptable reconstruction.
        """

        # Collect input activations for spectral reordering
        input_activations = self._collect_input_activations(name, linear)

        # Start with full rank for lossless reconstruction
        best_tensorized_linear = tensorized_linear = (
            TensorizedLinear.from_linear(
                linear,
                num_cores=self.num_cores,
                rank=2.0,
                input_activations=input_activations,
            )
            if self.block_size is None
            else BlockTensorizedLinear.from_linear(
                linear,
                self.block_size,
                num_cores=self.num_cores,
                rank=2.0,
                input_activations=input_activations,
            )
        ).to(linear.weight.device)

        total_num_epochs = 0
        while True:
            # Train this rank level
            final_snr, total_num_epochs = self._train_tensorized_layer(
                tensorized_linear,
                name,
                linear,
                total_num_epochs=total_num_epochs,
            )

            if final_snr >= self.target_snr_db:
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
                    energy_threshold=self.energy_threshold,
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
        total_num_epochs=0,
    ) -> tuple[float, int]:
        """
        Train a tensorized layer and return final average cosine similarity.

        Early exits if target_snr_db achieved.

        Returns:
            Average snr across all batches after training
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

        # Compute hybrid loss: MSE + (1 - cosine_similarity)
        # MSE forces the model to get the physical scale right
        # loss_criterion = torch.nn.MSELoss()
        # Cosine similarity encourages vectors to align in the same direction
        # Use SNR loss
        loss_criterion = SNRLoss()

        # Training loop with early stopping
        max_total_num_epochs = 100
        # Stop if no improvement for this many epochs
        patience = 3
        # Minimum improvement over previous loss to be considered progress
        min_frac_delta = 5e-3
        loss_history = []

        best_loss = float("inf")
        epochs_without_improvement = 0
        final_avg_snr = 0.0

        pbar = tqdm(
            total=max_total_num_epochs,
            initial=total_num_epochs,
            desc=f"{name} | ",
        )
        for epoch in range(max_total_num_epochs - total_num_epochs):
            total_loss = 0.0
            num_batches = 0
            cosine_similarities = []
            snrs = []
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

                loss = loss_criterion(dense_output, tensorized_batch_output)

                loss.backward()
                optimizer.step()

                dense_output = dense_output.detach()
                tensorized_batch_output = tensorized_batch_output.detach()

                cosine_similarities.append(
                    F.cosine_similarity(
                        dense_output, tensorized_batch_output, dim=-1
                    ).mean()
                )
                snrs.append(compute_snr(dense_output, tensorized_batch_output))
                mses.append(F.mse_loss(dense_output, tensorized_batch_output).item())
                matching_signs.append(
                    count_matching_signs(dense_output, tensorized_batch_output)
                )
                sign_norms.append(
                    compute_sign_norm(dense_output, tensorized_batch_output)
                )
                total_loss += loss.item()
                num_batches += 1

            epoch_loss = total_loss / num_batches

            # Calculate average cosine similarity for this epoch
            final_avg_snr = sum(snrs) / len(snrs) if snrs else 0.0

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
                f"{name} | # : "
                f"{tensorized_params:.1e} ({compression_pct}) | "
                f"mse: {sum(mses)/len(mses):.2e} | "
                f"snr: {final_avg_snr:.2e} | "
                f"signs: {sum(matching_signs)/len(matching_signs):.3f} | "
                f"sign norm: {sum(sign_norms)/len(sign_norms):.3f} | "
                f"cos(): {sum(cosine_similarities)/len(cosine_similarities):.3f}"
            )
            pbar.update(1)
            total_num_epochs += 1

            # Check early stopping conditions
            if (
                (epoch > 2 and final_avg_snr > self.target_snr_db)
                or epochs_without_improvement >= patience
                or epoch > 10
            ):
                break

        # Return final average snr
        return final_avg_snr, total_num_epochs

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


# weight-based sqnr
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


# activation-based snr
def compute_snr(original_output: torch.Tensor, tensorized_output: torch.Tensor):
    """
    Calculates SNR in dB using activation tensors directly.

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
    # mse_noise = torch.mean((y_true - y_pred) ** 2)
    mse_noise = torch.nn.functional.mse_loss(y_true, y_pred)

    # SNR calculation (10 * log10 of the power ratio)
    # 1e-10 added to denominator to prevent division by zero or log(0)
    snr_linear = signal_power / (mse_noise + 1e-10)
    snr_db = 10 * torch.log10(snr_linear)

    return snr_db.item()


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


class SNRLoss(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        # Calculate Variance of the signal (The "Target Power")
        signal_var = torch.var(y_true)

        # Calculate Mean Squared Error (The "Noise Power")
        mse_noise = torch.mean((y_true - y_pred) ** 2)

        # Linear SNR
        snr_linear = signal_var / (mse_noise + self.eps)

        # We want to maximize 10 * log10(snr_linear),
        # so we minimize -10 * log10(snr_linear)
        loss_db = -10 * torch.log10(snr_linear)

        return loss_db
