"""Spectral Block-TT Compression for Linear Layers.

Pipeline:
1. Sparsity-inducing permutation (groups high-magnitude rows/columns together)
2. DCT transforms activations to frequency domain
3. Frequency truncation (keep modes with 99.9% energy)
4. LSTSQ learns spectral weight W_freq (out_features × num_active)
5. Block-TT compresses W_freq (same infrastructure as BlockTensorizedLinear)

Forward: x → permute → DCT → select active freqs → Block-TT(W_freq) → IDCT → unpermute → y
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np

from llmcompressor.modifiers.experimental.block_tensorized_linear import BlockTensorizedLinear

__all__ = ["SpectralMPOLinear"]


class SpectralMPOLinear(nn.Module):
    """Spectral compression: sparsity permutation + DCT + freq truncation + Block-TT."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int,
        spectral_stages: nn.ModuleList,
        active_freq_indices: Optional[torch.Tensor] = None,
        input_perm: Optional[torch.Tensor] = None,
        input_inv_perm: Optional[torch.Tensor] = None,
        output_perm: Optional[torch.Tensor] = None,
        output_inv_perm: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        calib_inputs: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.dtype = dtype

        # Multi-scale Block-TT stages (coarse → fine)
        self.spectral_stages = spectral_stages

        # Active input frequency indices (for truncated spectrum)
        if active_freq_indices is not None:
            self.register_buffer("active_freq_indices", active_freq_indices.long())
            self.num_active_freqs = len(active_freq_indices)
        else:
            self.active_freq_indices = None
            self.num_active_freqs = in_features

        # Topological permutations
        if input_perm is not None:
            self.register_buffer("input_perm", input_perm.long())
            self.register_buffer("input_inv_perm", input_inv_perm.long())
        else:
            self.input_perm = None
            self.input_inv_perm = None

        if output_perm is not None:
            self.register_buffer("output_perm", output_perm.long())
            self.register_buffer("output_inv_perm", output_inv_perm.long())
        else:
            self.output_perm = None
            self.output_inv_perm = None

        if bias is not None:
            self.bias = nn.Parameter(bias.to(dtype), requires_grad=False)
        else:
            self.bias = None

        # Store calibration inputs for to_matrix()
        if calib_inputs is not None:
            num_samples = min(512, calib_inputs.shape[0])
            self.register_buffer("calib_inputs", calib_inputs[:num_samples].to(dtype))
        else:
            self.calib_inputs = None

    @staticmethod
    def _compute_input_permutation(weight: torch.Tensor) -> torch.Tensor:
        """Sparsity-inducing permutation for input (column) dimensions.

        Sorts columns by descending L2 norm so high-magnitude columns
        cluster at the start, making the weight block-compressible.
        """
        col_norms = weight.float().norm(dim=0)  # (in_features,)
        return torch.argsort(col_norms, descending=True)

    @staticmethod
    def _compute_output_permutation(
        weight: torch.Tensor, num_heads: int,
    ) -> torch.Tensor:
        """Sparsity-inducing permutation for output (row) dimensions.

        Sorts rows by descending L2 norm per head, so high-magnitude rows
        cluster at the start within each head block.
        """
        out_features = weight.shape[0]
        head_dim = out_features // num_heads

        if num_heads == 1:
            row_norms = weight.float().norm(dim=1)  # (out_features,)
            return torch.argsort(row_norms, descending=True)

        # Per-head: sort rows within each head block
        weight_heads = weight.float().reshape(num_heads, head_dim, -1)
        full_perm = []
        for h in range(num_heads):
            head_row_norms = weight_heads[h].norm(dim=1)  # (head_dim,)
            local_perm = torch.argsort(head_row_norms, descending=True)
            full_perm.append(local_perm + h * head_dim)
        return torch.cat(full_perm)

    @staticmethod
    def _dct_per_head(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Blocked DCT per head (preserves head boundaries)."""
        from scipy.fftpack import dct
        orig_dtype = tensor.dtype
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        num_samples = tensor.shape[0]
        features = tensor.shape[1]
        head_dim = features // num_heads
        tensor_heads = tensor.reshape(num_samples, num_heads, head_dim)
        np_tensor = tensor_heads.detach().cpu().numpy()
        dct_result = dct(np_tensor, axis=-1, norm='ortho')
        dct_result = dct_result.reshape(num_samples, features)
        result = torch.from_numpy(dct_result).to(tensor.device)
        if orig_dtype == torch.bfloat16:
            result = result.to(torch.bfloat16)
        return result

    @staticmethod
    def _idct_per_head(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Blocked IDCT per head (preserves head boundaries)."""
        from scipy.fftpack import idct
        orig_dtype = tensor.dtype
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        num_samples = tensor.shape[0]
        features = tensor.shape[1]
        head_dim = features // num_heads
        tensor_heads = tensor.reshape(num_samples, num_heads, head_dim)
        np_tensor = tensor_heads.detach().cpu().numpy()
        idct_result = idct(np_tensor, axis=-1, norm='ortho')
        idct_result = idct_result.reshape(num_samples, features)
        result = torch.from_numpy(idct_result).to(tensor.device)
        if orig_dtype == torch.bfloat16:
            result = result.to(torch.bfloat16)
        return result

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        num_heads: int,
        input_activations: torch.Tensor,
        target_snr_db: float = 40.0,
        param_budget: float = 0.5,
        block_size: int = 512,
        num_cores: int = 3,
        rank: float = 0.5,
        method: str = "activation",
        energy_threshold: float = 0.99,
        verbose: bool = True,
        # Legacy kwargs (ignored)
        mpo_bond_dim_ratio: float = 0.3,
        use_tucker_residual: bool = False,
        tucker_rank: float = 0.2,
        use_sparse_residual: bool = False,
        sparse_percentile: float = 99.9,
        block_mpo_heads: int = 0,
    ) -> "SpectralMPOLinear":
        """Build SpectralMPOLinear: RCM + DCT + freq truncation + Block-TT.

        Args:
            linear: Original linear layer
            num_heads: Number of (virtual) heads
            input_activations: Calibration data (num_samples, in_features)
            param_budget: Parameter budget as fraction of original
            block_size: Block size for Block-TT compression
            num_cores: Number of TT cores per block
            rank: TT rank ratio for Block-TT
            method: "activation" (LSTSQ) or "weight" (DCT of weight)
            energy_threshold: Fraction of input energy to retain (0.999 = 99.9%)
            verbose: Print progress
        """
        device = linear.weight.device
        dtype = linear.weight.dtype
        weight = linear.weight.detach().clone().float()
        bias = linear.bias.detach().clone() if linear.bias is not None else None

        out_features, in_features = weight.shape
        head_dim = out_features // num_heads
        original_params = weight.numel()

        # Move to CPU
        input_activations_cpu = input_activations.float().cpu()
        weight_cpu = weight.cpu()
        bias_cpu = bias.cpu() if bias is not None else None

        with torch.no_grad():
            output_activations_cpu = F.linear(input_activations_cpu, weight_cpu, bias_cpu)

        # Step 1: Sparsity-inducing permutation
        input_perm = cls._compute_input_permutation(weight_cpu)
        output_perm = cls._compute_output_permutation(weight_cpu, num_heads)
        input_inv_perm = torch.argsort(input_perm)
        output_inv_perm = torch.argsort(output_perm)

        if verbose:
            print(f"\nSpectral Block-TT ({out_features}x{in_features}, {num_heads} heads)")
            print(f"  Step 1: Sparsity permutation applied (sort by column/row L2 norm)")

        # Step 2: DCT + frequency truncation + LSTSQ
        from scipy.fftpack import dct

        active_freq_indices = None

        if method == "activation":
            input_perm_acts = input_activations_cpu[:, input_perm]
            output_perm_acts = output_activations_cpu[:, output_perm]

            # Global DCT on input
            input_freq = torch.from_numpy(
                dct(input_perm_acts.float().numpy(), axis=-1, norm='ortho')
            )

            # Frequency truncation
            freq_energy = (input_freq ** 2).sum(dim=0)
            total_energy = freq_energy.sum()
            sorted_energy, sorted_indices = torch.sort(freq_energy, descending=True)
            cumulative_energy = torch.cumsum(sorted_energy, dim=0) / total_energy

            num_active_raw = max(2, int((cumulative_energy < energy_threshold).sum().item()) + 1)
            num_active_raw = min(num_active_raw, in_features)

            # Round down to nearest multiple of block_size for Block-TT compatibility
            num_active = max(block_size, (num_active_raw // block_size) * block_size)
            num_active = min(num_active, in_features)
            # If in_features itself isn't divisible, round to largest divisor
            if in_features % block_size != 0 or out_features % block_size != 0:
                # Find a block size that divides both
                for bs in [block_size, 256, 128, 64, 32]:
                    if num_active_raw >= bs and out_features % bs == 0:
                        num_active = max(bs, (num_active_raw // bs) * bs)
                        block_size = bs
                        break

            active_freq_indices = sorted_indices[:num_active].sort().values
            input_freq_active = input_freq[:, active_freq_indices]

            actual_energy = cumulative_energy[num_active - 1].item()

            if verbose:
                print(f"  Step 2: DCT + freq truncation: {in_features} -> {num_active} modes "
                      f"({actual_energy:.4%} energy, raw={num_active_raw})")

            # Blocked DCT on output
            output_freq = cls._dct_per_head(output_perm_acts, num_heads)

            # LSTSQ
            weight_freq_T, _, _, _ = torch.linalg.lstsq(
                input_freq_active, output_freq, rcond=None
            )
            weight_freq = weight_freq_T.T  # (out_features, num_active)

            if verbose:
                recon = weight_freq @ input_freq_active.T
                freq_snr = 20 * torch.log10(
                    output_freq.T.norm() / ((output_freq.T - recon).norm() + 1e-10)
                ).item()
                # Also measure spatial-domain SNR (what actually matters)
                spatial_recon = cls._idct_per_head(
                    (weight_freq @ input_freq_active.T).T, num_heads
                )
                spatial_target = cls._idct_per_head(output_freq, num_heads)
                spatial_snr = 20 * torch.log10(
                    spatial_target.norm() / ((spatial_target - spatial_recon).norm() + 1e-10)
                ).item()
                print(f"  Step 2: LSTSQ fit — freq SNR: {freq_snr:.2f} dB, "
                      f"spatial SNR: {spatial_snr:.2f} dB "
                      f"({num_active} modes -> {out_features} outputs)")

        elif method == "weight":
            weight_perm_cpu = weight_cpu[output_perm, :][:, input_perm]
            weight_3d = weight_perm_cpu.reshape(num_heads, head_dim, in_features)
            weight_spectral_np = dct(weight_3d.numpy(), axis=1, norm='ortho')
            weight_freq = torch.from_numpy(weight_spectral_np).reshape(out_features, in_features)
            num_active = in_features

            if verbose:
                print(f"  Step 2: DCT(weight), no truncation ({in_features} modes)")
        else:
            raise ValueError(f"Unknown method: {method}")

        # Step 3: Multi-scale Block-TT cascade on W_freq
        # Coarse-to-fine: large blocks capture global patterns,
        # smaller blocks capture local residual patterns (like sub-band coding)
        spectral_weight = weight_freq.float()
        original_spectral = spectral_weight.clone()

        if verbose:
            print(f"  Step 3: Multi-scale Block-TT cascade")

        # Generate candidate block sizes: num_active, num_active/2, /4, ... down to 256
        candidate_sizes = []
        bs = num_active
        while bs >= 256:
            if out_features % bs == 0 and num_active % bs == 0:
                candidate_sizes.append(bs)
            bs //= 2

        if not candidate_sizes:
            # Fallback: find any valid block size
            for bs in [512, 256, 128, 64, 32]:
                if out_features % bs == 0 and num_active % bs == 0:
                    candidate_sizes = [bs]
                    break

        if verbose:
            print(f"    Candidate block sizes: {candidate_sizes}")

        spectral_stages = nn.ModuleList()
        residual = spectral_weight.clone()
        total_stage_params = 0
        target_params = int(original_params * param_budget)

        for scale_idx, bs in enumerate(candidate_sizes):
            # Distribute total budget evenly across all scales
            per_scale_budget = target_params // len(candidate_sizes)

            # Binary search on rank to hit per-scale budget
            temp_linear = nn.Linear(num_active, out_features, bias=False)
            temp_linear.weight.data = residual

            num_row_blocks = out_features // bs
            num_col_blocks = num_active // bs

            # Scale cores with block size: larger blocks get more cores
            scale_cores = max(2, min(5, int(math.log2(bs)) - 6))

            # Binary search on rank to fit per-scale budget
            lo_r, hi_r = 0.01, 0.95
            best_rank = lo_r  # Start at minimum, only grow if budget allows
            for _ in range(20):
                mid = (lo_r + hi_r) / 2
                try:
                    trial = BlockTensorizedLinear.from_linear(
                        temp_linear, block_size=bs, rank=mid, num_cores=scale_cores,
                    )
                    if trial.num_params <= per_scale_budget:
                        best_rank = mid
                        lo_r = mid
                    else:
                        hi_r = mid
                except Exception:
                    hi_r = mid

            stage = BlockTensorizedLinear.from_linear(
                temp_linear, block_size=bs, rank=best_rank, num_cores=scale_cores,
            )

            # Update residual
            with torch.no_grad():
                stage_recon = stage.to_matrix()
                residual = residual - stage_recon

            # Measure cumulative SNR
            total_recon = original_spectral - residual
            cumulative_snr = 20 * torch.log10(
                original_spectral.norm() / (residual.norm() + 1e-10)
            ).item()

            # Compute marginal SNR gain from this scale
            if scale_idx == 0:
                prev_snr = 0.0
            marginal_snr = cumulative_snr - prev_snr
            prev_snr = cumulative_snr

            total_stage_params += stage.num_params
            spectral_stages.append(stage)

            if verbose:
                param_pct = stage.num_params / original_params
                # Get actual TT ranks from first block
                sample_block = stage.blocks[0][0]
                tt_ranks = [1] + [core.shape[-1] for core in sample_block.factors]
                print(f"    Scale {scale_idx}: bs={bs} ({num_row_blocks}x{num_col_blocks} blocks), "
                      f"rank_ratio={best_rank:.3f}, TT ranks={tt_ranks}, {scale_cores} cores")
                print(f"      params: {stage.num_params:,} ({param_pct:.1%}), "
                      f"SNR gain: +{marginal_snr:.2f} dB, cumulative: {cumulative_snr:.1f} dB")

        if verbose:
            print(f"  Total spectral params: {total_stage_params:,} / {original_params:,} "
                  f"({total_stage_params/original_params:.1%})")

        # Move to device
        spectral_stages = spectral_stages.to(device)

        module = cls(
            in_features=in_features,
            out_features=out_features,
            num_heads=num_heads,
            spectral_stages=spectral_stages,
            active_freq_indices=active_freq_indices.to(device) if active_freq_indices is not None else None,
            input_perm=input_perm.to(device),
            input_inv_perm=input_inv_perm.to(device),
            output_perm=output_perm.to(device),
            output_inv_perm=output_inv_perm.to(device),
            bias=bias.to(device) if bias is not None else None,
            calib_inputs=input_activations_cpu.to(device),
            dtype=dtype,
        )

        return module

    def forward(self, x):
        """x → permute → DCT → select active freqs → Block-TT → IDCT → unpermute"""
        from scipy.fftpack import dct

        original_shape = x.shape
        batch_dims = original_shape[:-1]
        x_flat = x.reshape(-1, self.in_features)

        # Permute input
        if self.input_perm is not None:
            x_flat = x_flat[:, self.input_perm]

        # Global DCT
        x_f32 = x_flat.float()
        x_freq = torch.from_numpy(
            dct(x_f32.cpu().numpy(), axis=-1, norm='ortho')
        ).to(x.device)

        # Select active frequency modes
        if self.active_freq_indices is not None:
            x_freq_active = x_freq[:, self.active_freq_indices]
        else:
            x_freq_active = x_freq

        # Multi-scale Block-TT forward: sum outputs from all scales
        bt_dtype = next(self.spectral_stages.parameters()).dtype
        x_freq_bt = x_freq_active.to(bt_dtype)
        y_freq = sum(stage(x_freq_bt) for stage in self.spectral_stages)

        # Blocked IDCT per head
        y_flat = self._idct_per_head(y_freq, self.num_heads)

        # Unpermute output
        if self.output_inv_perm is not None:
            y_flat = y_flat[:, self.output_inv_perm]

        # Bias
        if self.bias is not None:
            y_flat = y_flat + self.bias.to(y_flat.dtype)

        return y_flat.reshape(*batch_dims, self.out_features).to(x.dtype)

    def to_matrix(self) -> torch.Tensor:
        """Compute effective weight via calibration LSTSQ."""
        if self.calib_inputs is None:
            return torch.zeros(self.out_features, self.in_features, dtype=self.dtype)

        with torch.no_grad():
            calib_outputs = self.forward(self.calib_inputs)

        calib_inputs_f32 = self.calib_inputs.float()
        calib_outputs_f32 = calib_outputs.float()
        W_eff_T, _, _, _ = torch.linalg.lstsq(
            calib_inputs_f32, calib_outputs_f32, rcond=None
        )
        return W_eff_T.T.to(self.dtype)

    @property
    def num_params(self):
        return sum(stage.num_params for stage in self.spectral_stages)
