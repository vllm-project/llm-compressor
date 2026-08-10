# Differences from https://huggingface.co/moonshotai/Kimi-K3/tree/main:
# - Formatting only (no functional changes)

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class KimiLinearConfig(PretrainedConfig):
    model_type = "kimi_linear"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        model_type="kimi_linear",
        vocab_size=163840,
        hidden_size=4096,
        head_dim=None,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=10000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
        moe_intermediate_size: Optional[int] = None,
        moe_renormalize: bool = True,
        moe_router_activation_func: str = "sigmoid",
        num_experts: Optional[int] = None,
        num_experts_per_token: Optional[int] = None,
        num_shared_experts: int = 0,
        routed_scaling_factor: float = 1.0,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        use_grouped_topk: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: Optional[int] = None,
        qk_nope_head_dim: Optional[int] = None,
        qk_rope_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        mla_use_nope: Optional[bool] = False,
        mla_use_output_gate: Optional[bool] = False,
        num_nextn_predict_layers: int = 0,
        linear_attn_config: Optional[dict] = None,
        attn_res_block_size: Optional[int] = None,
        latent_moe_use_norm: bool = False,
        activation_situ_beta: Optional[float] = None,
        activation_situ_linear_beta: Optional[float] = None,
        max_position_embeddings: int = 4096,
        routed_expert_hidden_size: Optional[int] = None,
        topk_method: str = "noaux_tc",
        **kwargs,
    ):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope
        self.mla_use_output_gate = mla_use_output_gate
        # moe config
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_renormalize = moe_renormalize
        self.num_shared_experts = num_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.moe_router_activation_func = moe_router_activation_func
        assert self.moe_router_activation_func in ("softmax", "sigmoid")
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.num_nextn_predict_layers = num_nextn_predict_layers

        self.attn_res_block_size = attn_res_block_size
        self.latent_moe_use_norm = latent_moe_use_norm
        self.activation_situ_beta = activation_situ_beta
        self.activation_situ_linear_beta = activation_situ_linear_beta
        self.max_position_embeddings = max_position_embeddings
        self.routed_expert_hidden_size = routed_expert_hidden_size
        self.topk_method = topk_method

        if linear_attn_config is not None:
            assert linear_attn_config["kda_layers"] is not None
            assert linear_attn_config["full_attn_layers"] is not None
        self.linear_attn_config = linear_attn_config

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def is_mla(self):
        return (
            self.q_lora_rank is not None
            or self.kv_lora_rank is not None
            or self.qk_nope_head_dim is not None
            or self.qk_rope_head_dim is not None
            or self.v_head_dim is not None
            or self.mla_use_nope is True
        )

    @property
    def is_moe(self):
        return self.num_experts is not None

    @property
    def is_linear_attn(self) -> bool:
        return not (
            self.linear_attn_config is None
            or (
                isinstance(self.linear_attn_config, dict)
                and self.linear_attn_config["kda_layers"] is not None
                and len(self.linear_attn_config["kda_layers"]) == 0
            )
        )

    def is_kda_layer(self, layer_idx: int):
        return (
            self.linear_attn_config is not None
            and (layer_idx + 1) in self.linear_attn_config["kda_layers"]
        )


class KimiK3VisionConfig(PretrainedConfig):
    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        vt_num_attention_heads: int = 12,
        vt_num_hidden_layers: int = 27,
        vt_hidden_size: int = 1024,
        vt_intermediate_size: int = 4096,
        merge_kernel_size: tuple = (2, 2),
        merge_type: str = "sd2_tpool",
        _attn_implementation: str = "flash_attention_2",
        # MM Projector parameters
        mm_projector_type: str = "patchmergerv2",
        mm_hidden_size: int | None = None,
        projector_hidden_act: str = "gelu",
        projector_ln_eps: float = 1e-5,
        # vision tower parameters
        qkv_hidden_size: int = 1536,
        norm_type: str = "rmsnorm",
        attn_bias: bool = False,
        patch_embed_proj_bias: bool = False,
        mlp_type: str = "mlp2",
        linear_bias: bool = False,
        activation_func: str = "gelu_pytorch_tanh",
        pos_emb_interpolation_mode: str = "bilinear",
        # Other parameters
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        text_hidden_size=7168,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.vt_num_attention_heads = vt_num_attention_heads
        self.vt_num_hidden_layers = vt_num_hidden_layers
        self.vt_hidden_size = vt_hidden_size
        self.vt_intermediate_size = vt_intermediate_size
        self.merge_kernel_size = merge_kernel_size
        self.merge_type = merge_type
        self._attn_implementation = _attn_implementation

        # MM Projector config
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = (
            mm_hidden_size if mm_hidden_size is not None else vt_hidden_size
        )
        self.projector_hidden_act = projector_hidden_act
        self.projector_ln_eps = projector_ln_eps
        self.text_hidden_size = text_hidden_size

        # vision tower parameters
        self.qkv_hidden_size = qkv_hidden_size
        self.norm_type = norm_type
        self.attn_bias = attn_bias
        self.patch_embed_proj_bias = patch_embed_proj_bias
        self.mlp_type = mlp_type
        self.linear_bias = linear_bias
        self.activation_func = activation_func
        self.pos_emb_interpolation_mode = pos_emb_interpolation_mode

        super().__init__(**kwargs)


class KimiK3Config(PretrainedConfig):
    """Kimi-K3 model configuration.

    Args:
        text_config (dict | KimiLinearConfig): Configuration for the text model.

        Vision Tower Parameters (from MoonViT3dConfig):
            patch_size (int): Patch size for vision tower.
            init_pos_emb_height (int): Initial position embedding height.
            init_pos_emb_width (int): Initial position embedding width.
            init_pos_emb_time (int): Initial position embedding time dimension.
            pos_emb_type (str): Type of position embedding.
            vt_num_attention_heads (int): Number of attention heads in vision tower.
            vt_num_hidden_layers (int): Number of hidden layers in vision tower.
            vt_hidden_size (int): Hidden size of vision tower.
            vt_intermediate_size (int): Intermediate size in vision tower FFN.
            merge_kernel_size (tuple): Kernel size for patch merging.
            merge_type (str): Type of merge operation.
            _attn_implementation (str): Attention implementation type.

        MM Projector Parameters (from MultiModalProjectorConfig):
            mm_projector_type (str): Type of multimodal projector.
            mm_hidden_size (int): Hidden size from vision tower (should match vt_hidden_size).
            projector_hidden_act (str): Activation function for projector.
            projector_ln_eps (float): Layer norm epsilon for projector.

        Other Parameters:
            ignore_index (int): The ignore index for the loss function.
            media_placeholder_token_id (int): The token ID to use for media placeholders.
            pad_token_id (int): The token ID to use for padding.
    """

    model_type = "kimi_k3"

    def __init__(
        self,
        text_config: dict | KimiLinearConfig = None,
        vision_config: dict | KimiK3VisionConfig = None,
        # Other parameters
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = KimiLinearConfig(**text_config)
        if isinstance(vision_config, dict):
            vision_config = KimiK3VisionConfig(**vision_config)
        self.text_config = text_config
        self.vision_config = vision_config
        # Other config
        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        if getattr(self.text_config, "quantization_config", None) is not None:
            self.quantization_config = self.text_config.quantization_config

        super().__init__(pad_token_id=pad_token_id, **kwargs)
