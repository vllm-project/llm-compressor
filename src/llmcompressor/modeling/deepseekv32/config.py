from typing import Tuple, Optional, Literal
from transformers.configuration_utils import PretrainedConfig


class ModelConfig(PretrainedConfig):
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        scale_fmt (Optional[str]): Format for quantization scale.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
        index_head_dim (int): Dimension for index head.
        index_topk (int): Top-k for index head.
    """

    model_type = "deepseek_v32"

    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    scale_fmt: Optional[str] = None
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    # index
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048

    @property
    def qk_head_dim(self) -> int:
        return self.qk_rope_head_dim + self.qk_nope_head_dim

    def ___init__(
        self,
        max_batch_size: int = 8,
        max_seq_len: int = 4096 * 4,
        dtype: Literal["bf16", "fp8"] = "bf16",
        scale_fmt: Optional[str] = None,
        vocab_size: int = 102400,
        dim: int = 2048,
        inter_dim: int = 10944,
        moe_inter_dim: int = 1408,
        n_layers: int = 27,
        n_dense_layers: int = 1,
        n_heads: int = 16,
        # moe
        n_routed_experts: int = 64,
        n_shared_experts: int = 2,
        n_activated_experts: int = 6,
        n_expert_groups: int = 1,
        n_limited_groups: int = 1,
        score_func: Literal["softmax", "sigmoid"] = "softmax",
        route_scale: float = 1.0,
        # mla
        q_lora_rank: int = 0,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        # yarn
        original_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        rope_factor: float = 40,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1.0,
        # index
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 2048,
        **kwargs,
    ):

        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.scale_fmt = scale_fmt
        self.vocab_size = vocab_size
        self.dim = dim
        self.inter_dim = inter_dim
        self.moe_inter_dim = moe_inter_dim
        self.n_layers = n_layers
        self.n_dense_layers = n_dense_layers
        self.n_heads = n_heads
        # moe
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_activated_experts = n_activated_experts
        self.n_expert_groups = n_expert_groups
        self.n_limited_groups = n_limited_groups
        self.score_func = score_func
        self.route_scale = route_scale
        # mla
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        # yarn
        self.original_seq_len = original_seq_len
        self.rope_theta = rope_theta
        self.rope_factor = rope_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        # index
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk

        super().__init__(**kwargs)
