"""
Central source of truth for vLLM-aligned fused module layouts.

Defines which submodules form a single "fused" group for TENSOR_GROUP (e.g. NVFP4)
global-scale sharing in vLLM. All callers that need to detect or iterate over
fused attention / MLP linears should use the functions in this module.

Fused attention (vLLM)
----------------------
- **Traditional:** Three linears `q_proj`, `k_proj`, `v_proj` that share one
  global scale in vLLM.
- **Fused QKV:** Single linear `qkv_proj` (Phi, etc.). No multi-layer fusion
  needed; already one tensor.
- **MLA (Multi-head Latent Attention):** Two linears that share one global
  scale: query projection and fused KV projection. Common attribute names:
  - `q_a_proj` (or `q_proj`) for query
  - `kv_a_proj_with_mqa` for key/value (fused KV with MQA).
  Used in DeepSeek V2/V3, Kimi K2, Mistral Large 3, and similar architectures.

Fused MLP (vLLM)
----------------
- **Gate/Up:** Two linears `gate_proj`, `up_proj` that share one global scale.
- **Fused Gate-Up:** Single linear `gate_up_proj` (Phi, etc.). No multi-layer
  fusion needed.
"""

from __future__ import annotations

from typing import List, Optional

from torch.nn import Linear, Module

__all__ = [
    "get_fused_attention_linears",
    "get_fused_mlp_linears",
    "is_fused_attention_module",
    "is_fused_mlp_module",
]


def get_fused_attention_linears(module: Module) -> Optional[List[Linear]]:
    """
    Return the list of Linear submodules that form one fused attention group
    for vLLM TENSOR_GROUP global scale, or None if this module is not a known
    fused attention container.

    Definitions (vLLM-aligned):
    - **Traditional:** `q_proj`, `k_proj`, `v_proj` (three linears).
    - **Fused QKV:** single `qkv_proj` → returns None (no cross-layer fusion).
    - **MLA:** `q_a_proj` (or `q_proj`) + `kv_a_proj_with_mqa` (two linears).

    :param module: A candidate attention container (e.g. parent of q/k/v or MLA).
    :return: List of Linear modules that should share one global scale, or None.
    """
    # Already fused as one layer; no cross-layer global scale to apply
    if hasattr(module, "qkv_proj"):
        return None

    # Traditional: q_proj, k_proj, v_proj
    if (
        hasattr(module, "q_proj")
        and hasattr(module, "k_proj")
        and hasattr(module, "v_proj")
    ):
        q, k, v = module.q_proj, module.k_proj, module.v_proj
        # Avoid treating MLA blocks as traditional (MLA has q_proj + kv_a_proj_with_mqa)
        if hasattr(module, "kv_a_proj_with_mqa"):
            return None
        if isinstance(q, Linear) and isinstance(k, Linear) and isinstance(v, Linear):
            return [q, k, v]

    # MLA: q_a_proj (or q_proj) + kv_a_proj_with_mqa
    if hasattr(module, "kv_a_proj_with_mqa"):
        kv = module.kv_a_proj_with_mqa
        q_linear = getattr(module, "q_a_proj", None) or getattr(module, "q_proj", None)
        if (
            q_linear is not None
            and isinstance(q_linear, Linear)
            and isinstance(kv, Linear)
        ):
            return [q_linear, kv]

    return None


def get_fused_mlp_linears(module: Module) -> Optional[List[Linear]]:
    """
    Return the list of Linear submodules that form one fused MLP group for
    vLLM TENSOR_GROUP global scale, or None if not a known fused MLP container.

    Definitions (vLLM-aligned):
    - **Gate/Up:** `gate_proj`, `up_proj` (two linears).
    - **Fused Gate-Up:** single `gate_up_proj` → returns None (no cross-layer fusion).

    :param module: A candidate MLP container (e.g. parent of gate_proj/up_proj).
    :return: List of Linear modules that should share one global scale, or None.
    """
    # Already fused as one layer
    if hasattr(module, "gate_up_proj"):
        return None

    # Gate/Up: gate_proj, up_proj (require "mlp" in class name to avoid false positives)
    if "mlp" not in module.__class__.__name__.lower():
        return None
    if hasattr(module, "gate_proj") and hasattr(module, "up_proj"):
        gate = module.gate_proj
        up = module.up_proj
        if isinstance(gate, Linear) and isinstance(up, Linear):
            return [gate, up]

    return None


def is_fused_attention_module(module: Module) -> bool:
    """True if this module is a fused attention container (traditional or MLA)."""
    return get_fused_attention_linears(module) is not None


def is_fused_mlp_module(module: Module) -> bool:
    """True if this module is a fused MLP container (gate/up)."""
    return get_fused_mlp_linears(module) is not None
