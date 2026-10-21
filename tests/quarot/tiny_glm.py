# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small synthetic MLA decoder, not a GLM checkpoint or deployment model.

Includes RoPE, indexer scores/top-k reuse, gated dense MLP, routed/shared experts,
non-unit RMSNorm gains and an untied LM head. All weights are generated locally.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TinyConfig:
    hidden_size: int = 64
    q_lora_rank: int = 32
    kv_lora_rank: int = 32
    v_head_dim: int = 32
    qk_nope_head_dim: int = 32
    qk_rope_head_dim: int = 32
    num_attention_heads: int = 2
    num_hidden_layers: int = 2
    first_k_dense_replace: int = 1
    n_routed_experts: int = 2
    n_shared_experts: int = 1
    vocab_size: int = 96
    intermediate_size: int = 96
    indexer_types: tuple[str, ...] = ("full", "shared")


class RMSNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.6, 1.4, size))

    def forward(self, x):
        return x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * self.weight


def rope(x):
    length, dim = x.shape[1], x.shape[-1]
    phase = torch.outer(
        torch.arange(length, dtype=x.dtype), torch.arange(dim // 2, dtype=x.dtype) / dim
    )
    cos, sin = phase.cos()[None, :, None, :], phase.sin()[None, :, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


class Indexer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.wk = nn.Linear(c.hidden_size, 16, bias=False)
        self.weights_proj = nn.Linear(c.hidden_size, 2, bias=False)
        self.wq_b = nn.Linear(c.q_lora_rank, 32, bias=False)
        self.k_norm = nn.LayerNorm(16)

    def forward(self, x, qr):
        q = self.wq_b(qr).unflatten(-1, (2, 16))
        k = self.k_norm(self.wk(x))
        dots = torch.einsum("bshd,btd->bsht", q, k).relu()
        return (dots * self.weights_proj(x).unsqueeze(-1)).sum(2)


class MLA(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.config = c
        self.q_a_proj = nn.Linear(c.hidden_size, c.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(c.q_lora_rank)
        self.q_b_proj = nn.Linear(
            c.q_lora_rank,
            c.num_attention_heads * (c.qk_nope_head_dim + c.qk_rope_head_dim),
            bias=False,
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            c.hidden_size, c.kv_lora_rank + c.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = RMSNorm(c.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            c.kv_lora_rank,
            c.num_attention_heads * (c.qk_nope_head_dim + c.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            c.num_attention_heads * c.v_head_dim, c.hidden_size, bias=False
        )
        self.indexer = Indexer(c) if c.indexer_types[index] == "full" else None

    def forward(self, x, previous_mask):
        c = self.config
        qr = self.q_a_layernorm(self.q_a_proj(x))
        qn, qp = (
            self.q_b_proj(qr)
            .unflatten(-1, (c.num_attention_heads, -1))
            .split([c.qk_nope_head_dim, c.qk_rope_head_dim], -1)
        )
        kv, kp = self.kv_a_proj_with_mqa(x).split(
            [c.kv_lora_rank, c.qk_rope_head_dim], -1
        )
        kv = self.kv_a_layernorm(kv)
        kn, v = (
            self.kv_b_proj(kv)
            .unflatten(-1, (c.num_attention_heads, -1))
            .split([c.qk_nope_head_dim, c.v_head_dim], -1)
        )
        qp, kp = rope(qp), rope(kp.unsqueeze(2))
        q = torch.cat((qn, qp), -1)
        k = torch.cat((kn, kp.expand(-1, -1, c.num_attention_heads, -1)), -1)
        logits = torch.einsum("bshd,bthd->bsht", q, k) / q.shape[-1] ** 0.5
        length = x.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool).tril()
        scores = None
        if self.indexer is not None:
            scores = self.indexer(x, qr)
            indices = (
                scores.masked_fill(~causal, -torch.inf).topk(min(3, length), -1).indices
            )
            mask = (
                torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, indices, True)
                & causal
            )
        else:
            mask = previous_mask if previous_mask is not None else causal
        probabilities = logits.masked_fill(~mask.unsqueeze(-2), -torch.inf).softmax(-1)
        result = self.o_proj(
            torch.einsum("bsht,bthd->bshd", probabilities, v).flatten(2)
        )
        return (
            result,
            mask,
            {
                "q": q,
                "k": k,
                "v": v,
                "index_scores": scores,
                "probabilities": probabilities,
            },
        )


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    """Exercise a raw weight router, not just nn.Linear targets."""

    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(c.n_routed_experts, c.hidden_size) / c.hidden_size**0.5
        )

    def forward(self, x):
        return F.linear(x, self.weight)


class MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = Router(c)
        self.experts = nn.ModuleList(MLP(c) for _ in range(c.n_routed_experts))
        self.shared_experts = MLP(c)

    def forward(self, x):
        logits = self.gate(x)
        probabilities = logits.softmax(-1)
        selection = probabilities.argmax(-1, keepdim=True)
        weights = torch.zeros_like(probabilities).scatter_(
            -1, selection, probabilities.gather(-1, selection)
        )
        results = torch.stack([expert(x) for expert in self.experts], -2)
        return self.shared_experts(x) + (results * weights.unsqueeze(-1)).sum(
            -2
        ), logits


class Decoder(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.input_layernorm = RMSNorm(c.hidden_size)
        self.self_attn = MLA(c, index)
        self.post_attention_layernorm = RMSNorm(c.hidden_size)
        self.mlp = MLP(c) if index < c.first_k_dense_replace else MoE(c)

    def forward(self, x, mask):
        attn, mask, trace = self.self_attn(self.input_layernorm(x), mask)
        x = x + attn
        mlp = self.mlp(self.post_attention_layernorm(x))
        if isinstance(mlp, tuple):
            mlp, trace["router_logits"] = mlp
        x = x + mlp
        trace["hidden"] = x
        return x, mask, trace


class TinyGLM(nn.Module):
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or TinyConfig()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.model.layers = nn.ModuleList(
            Decoder(c, i) for i in range(c.num_hidden_layers)
        )
        self.model.norm = RMSNorm(c.hidden_size)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)

    def forward(self, input_ids):
        x, mask, traces = self.model.embed_tokens(input_ids), None, []
        for layer in self.model.layers:
            x, mask, trace = layer(x, mask)
            traces.append(trace)
        return self.lm_head(self.model.norm(x)), traces
