import math
from typing import Tuple, Optional

import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
import torch.distributed as dist

from .kernel import act_quant, fp8_index, bf16_index
from .config import ModelConfig
from transformers.models.deepseek_v3 import DeepseekV3ForCausalLM

from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention

world_size = 1
rank = 0
block_size = 128


class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert (
            vocab_size % world_size == 0
        ), f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = vocab_size // world_size
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.

        Raises:
            ValueError: If `world_size` is not defined.
        """
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        dtype = x.dtype
        if residual is None:
            x = x.float()
            var = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return (self.weight * x).to(dtype)
        else:
            x = residual = x.float() + residual.float()
            var = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return (self.weight * x).to(dtype), residual.to(dtype)


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(
            x.float(), (self.dim,), self.weight.float(), self.bias.float(), self.eps
        ).type_as(x)


def precompute_freqs_cis(args: ModelConfig) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelConfig): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, args.original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True
) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    shape = x.shape
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    return y.to(dtype)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    # from fast_hadamard_transform import hadamard_transform
    from compressed_tensors.transform.utils.hadamard import (
        deterministic_hadamard_matrix,
    )

    hidden_size = x.size(-1)
    hadamard_matrix = deterministic_hadamard_matrix(
        hidden_size, torch.bfloat16, x.device
    )
    return F.linear(x, hadamard_matrix) * hidden_size**-0.5
    # return F.linear(x, torch.tensor(scipy.linalg.hadamard(hidden_size))) * hidden_size**-0.5
    # return hadamard_transform(x, scale=hidden_size**-0.5)


class Indexer(torch.nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.dim: int = args.dim
        self.n_heads: int = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim: int = args.index_head_dim
        self.rope_head_dim: int = args.qk_rope_head_dim
        self.index_topk: int = args.index_topk
        self.q_lora_rank: int = args.q_lora_rank
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = Linear(self.dim, self.head_dim, bias=False)
        self.k_norm = LayerNorm(self.head_dim)
        # weights_proj in the checkpoint is stored in bf16, while the parameters here are stored in fp32 for convenient.
        self.weights_proj = Linear(
            self.dim,
            self.n_heads,
            # dtype=torch.float32,
            bias=False,
        )
        self.softmax_scale = self.head_dim**-0.5
        self.scale_fmt = args.scale_fmt

        self.register_buffer(
            "k_cache",
            torch.zeros(
                args.max_batch_size,
                args.max_seq_len,
                self.head_dim,
                dtype=torch.bfloat16,
                # dtype=torch.float8_e4m3fn,
            ),
            persistent=False,
        )
        self.register_buffer(
            "k_scale_cache",
            torch.zeros(
                args.max_batch_size,
                args.max_seq_len,
                self.head_dim // block_size,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        # rope in indexer is not interleaved
        q_pe = apply_rotary_emb(q_pe, freqs_cis, False)
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        # rope in indexer is not interleaved
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, False).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)
        q = rotate_activation(q)
        k = rotate_activation(k)
        # q_fp8, q_scale = act_quant(q, block_size, self.scale_fmt)
        # k_fp8, k_scale = act_quant(k, block_size, self.scale_fmt)
        # self.k_cache[:bsz, start_pos:end_pos] = k_fp8
        # self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale
        self.k_cache[:bsz, start_pos:end_pos] = k

        weights = self.weights_proj(x) * self.n_heads**-0.5

        weights = weights.unsqueeze(-1) * q * self.softmax_scale
        # weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        # # re-squeeze https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/issues/43

        weights = weights.squeeze(-1).to(torch.bfloat16).contiguous()

        # index_score = fp8_index(
        #     q_fp8.contiguous(),
        #     weights,
        #     self.k_cache[:bsz, :end_pos].contiguous(),
        #     self.k_scale_cache[:bsz, :end_pos].squeeze(-1).contiguous(),
        # )

        k_ = self.k_cache[:bsz, :end_pos].contiguous()
        index_score = bf16_index(
            weights,
            k_,
        )

        if mask is not None:
            index_score += mask
        topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]

        if world_size > 1:
            topk_indices_ = topk_indices.clone()
            dist.broadcast(topk_indices_, src=0)
            assert torch.all(
                topk_indices == topk_indices_
            ), f"{topk_indices=} {topk_indices_=}"

        return topk_indices


def weight_dequant(weight, scale):
    shape = weight.shape
    assert weight.dim() == 2
    weight = (
        weight.view(
            shape[0] // block_size, block_size, shape[1] // block_size, block_size
        )
        .transpose(1, 2)
        .contiguous()
        .view(-1, block_size * block_size)
    )
    weight = (
        (weight.float() * scale.view(-1, 1).float())
        .to(torch.get_default_dtype())
        .view(shape[0] // block_size, shape[1] // block_size, block_size, block_size)
        .transpose(1, 2)
        .contiguous()
        .view(shape)
    )
    return weight


class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """

    def __init__(self, args: ModelConfig):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        self.q_a_proj = Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank)
        self.q_b_proj = Linear(
            self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = Linear(
            self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)
        self.softmax_scale = self.qk_head_dim**-0.5
        self.scale_fmt = args.scale_fmt
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.indexer = Indexer(args)

        self.register_buffer(
            "kv_cache",
            torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank),
            persistent=False,
        )
        self.register_buffer(
            "pe_cache",
            torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim),
            persistent=False,
        )
        self.dequant_wkv_b = None

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.kv_a_proj_with_mqa(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_a_layernorm(kv)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        # we use fp8 kv cache in actual deployment, so here we simulate the precision by casting kv to fp8 and then back to bf16.
        # kv_fp8, kv_scale = act_quant(kv, block_size, self.scale_fmt)
        # kv = (
        #     (kv_fp8.view(-1, block_size).float() * kv_scale.view(-1, 1))
        #     .to(kv.dtype)
        #     .view_as(kv)
        # )
        self.kv_cache[:bsz, start_pos:end_pos] = kv
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
        if mask is not None:  # MHA prefill
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.kv_b_proj(kv)
            kv = kv.view(
                bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, v = torch.split(
                kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            scores = torch.einsum("bshd,bthd->bsht", q, k).mul_(self.softmax_scale)

            # indexer
            topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)
            index_mask = torch.full(
                (bsz, seqlen, seqlen), float("-inf"), device=x.device
            ).scatter_(-1, topk_indices, 0)
            index_mask += mask
            scores += index_mask.unsqueeze(2)

            scores = scores.softmax(dim=-1)
            x = torch.einsum("bsht,bthd->bshd", scores, v)
        else:  # MQA decode
            if self.dequant_wkv_b is None and self.kv_b_proj.scale is not None:
                self.dequant_wkv_b = weight_dequant(
                    self.kv_b_proj.weight, self.kv_b_proj.scale
                )
            kv_b_proj = (
                self.kv_b_proj.weight
                if self.dequant_wkv_b is None
                else self.dequant_wkv_b
            )
            kv_b_proj = kv_b_proj.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum(
                "bshd,hdc->bshc", q_nope, kv_b_proj[:, : self.qk_nope_head_dim]
            )
            scores = (
                torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])
                + torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
            ) * self.softmax_scale

            # indexer
            topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)
            index_mask = torch.full(
                (bsz, 1, end_pos), float("-inf"), device=x.device
            ).scatter_(-1, topk_indices, 0)
            scores += index_mask.unsqueeze(2)

            scores = scores.softmax(dim=-1)
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, kv_b_proj[:, -self.v_head_dim :])
        x = self.o_proj(x.flatten(2))
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        gate_proj (nn.Module): Linear layer for input-to-hidden transformation.
        down_proj (nn.Module): Linear layer for hidden-to-output transformation.
        up_proj (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int, reduce_output: bool = True):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.gate_proj = Linear(dim, inter_dim, bias=False)
        self.down_proj = Linear(inter_dim, dim, bias=False)
        self.up_proj = Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.down_proj(
            (F.silu(self.gate_proj(x).float()) * self.up_proj(x).float()).type_as(x)
        )


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """

    def __init__(self, args: ModelConfig):
        """
        Initializes the Gate module.

        Args:
            args (ModelConfig): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.e_score_correction_bias = (
            nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))
            if self.dim == 7168
            else None
        )

    @property
    def bias(self):
        return self.e_score_correction_bias

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.e_score_correction_bias is not None:
            scores = scores + self.e_score_correction_bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(
                1, indices, False
            )
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        gate_proj (nn.Module): Linear layer for input-to-hidden transformation.
        down_proj (nn.Module): Linear layer for hidden-to-output transformation.
        up_proj (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.gate_proj = Linear(dim, inter_dim, bias=False)
        self.down_proj = Linear(inter_dim, dim, bias=False)
        self.up_proj = Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.down_proj(
            (F.silu(self.gate_proj(x).float()) * self.up_proj(x).float()).type_as(x)
        )


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """

    def __init__(self, args: ModelConfig):
        """
        Initializes the MoE module.

        Args:
            args (ModelConfig): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.dim
        assert (
            args.n_routed_experts % world_size == 0
        ), f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList(
            [
                (
                    Expert(args.dim, args.moe_inter_dim)
                    if self.experts_start_idx <= i < self.experts_end_idx
                    else None
                )
                for i in range(self.n_routed_experts)
            ]
        )
        self.shared_experts = MLP(
            args.dim, args.n_shared_experts * args.moe_inter_dim, reduce_output=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts
        ).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        y += self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        return y.type_as(x).view(shape)


class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        self_attn (nn.Module): Attention layer (MLA).
        mlp (nn.Module): Feed-forward network (MLP or MoE).
        input_layernorm (nn.Module): Layer normalization for attention.
        post_attention_layernorm (nn.Module): Layer normalization for feed-forward network.
    """

    def __init__(self, layer_id: int, args: ModelConfig):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelConfig): Model arguments containing block parameters.
        """
        super().__init__()

        # self.self_attn = DeepseekV3Attention(args, layer_id)
        self.self_attn = MLA(args)
        self.mlp = (
            MLP(args.dim, args.inter_dim)
            if layer_id < args.n_dense_layers
            else MoE(args)
        )
        self.input_layernorm = RMSNorm(args.dim)
        self.post_attention_layernorm = RMSNorm(args.dim)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        if residual is None:
            x, residual = self.input_layernorm(x), x
        else:
            x, residual = self.input_layernorm(x, residual)
        x = self.self_attn(x, start_pos, freqs_cis, mask)
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        return x, residual


class Transformer(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed_tokens (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        lm_head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """

    def __init__(self, args: ModelConfig):
        """
        Initializes the Transformer model.

        Args:
            args (ModelConfig): Model arguments containing transformer parameters.
        """
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        # Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        # Linear.scale_fmt = args.scale_fmt
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed_tokens = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim)
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0):
        """
        Forward pass for the Transformer model.

        Args:
            input_ids (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        seqlen = input_ids.size(1)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        mask = (
            torch.full((seqlen, seqlen), float("-inf"), device=input_ids.device).triu_(
                1
            )
            if seqlen > 1
            else None
        )
        h, residual = self.embed_tokens(input_ids), None
        for layer in self.layers:
            h, residual = layer(h, residual, start_pos, freqs_cis, mask)
        h, _ = self.norm(h, residual)
        return h


class DeepseekV32PreTrainedModel(PreTrainedModel):
    config: ModelConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Block"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Block,
        "attentions": MLA,
    }
    # _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.61.*"]

    # @torch.no_grad()
    # def _init_weights(self, module):
    #     super()._init_weights(module)
    #     # if isinstance(module, DeepseekV3TopkRouter):
    #     #     init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    #     #     init.zeros_(module.e_score_correction_bias)
    #     # elif isinstance(module, DeepseekV3NaiveMoe):
    #     #     init.normal_(
    #     #         module.gate_up_proj, mean=0.0, std=self.config.initializer_range
    #     #     )
    #     #     init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)


class DeepseekV32ForCausalLM(DeepseekV32PreTrainedModel, GenerationMixin):

    config: ModelConfig

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Transformer(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, DeepseekV3ForCausalLM

        >>> model = DeepseekV3ForCausalLM.from_pretrained("meta-deepseek_v3/DeepseekV3-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-deepseek_v3/DeepseekV3-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        hidden_states = self.model(
            input_ids=input_ids,
            # attention_mask=attention_mask,
            # position_ids=position_ids,
            # past_key_values=past_key_values,
            # inputs_embeds=inputs_embeds,
            # use_cache=use_cache,
            # **kwargs,
        )

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            # past_key_values=outputs.past_key_values,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelConfig()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)
    print(model(x).size())
