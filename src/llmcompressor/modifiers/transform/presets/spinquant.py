from compressed_tensors.transform import TransformArgs, TransformConfig, TransformScheme

# Ref: https://arxiv.org/pdf/2405.16406 Fig 1

# Mergeable rotations R1 and R2 only
LLAMA_SPINQUANT_R1R2 = TransformConfig(
    config_groups={
        "R1": TransformScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["re:.*embed_tokens$", "re:.*o_proj$", "re:.*down_proj$"],
                    location="weight_output",
                ),
                TransformArgs(
                    targets=[
                        "re:.*q_proj$",
                        "re:.*k_proj$",
                        "re:.*v_proj$",
                        "re:.*up_proj$",
                        "re:.*gate_proj$",
                        "lm_head",
                    ],
                    location="weight_input",
                    inverse=True,
                ),
            ],
        ),
        "R2": TransformScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["re:.*v_proj$"],
                    location="weight_output",
                ),
                TransformArgs(
                    targets=["re:.*o_proj$"], location="weight_input", inverse=True
                ),
            ],
        ),
    }
)

# All rotations
LLAMA_SPINQUANT = TransformConfig(
    config_groups={
        "R1": LLAMA_SPINQUANT_R1R2.config_groups["R1"],
        "R2": LLAMA_SPINQUANT_R1R2.config_groups["R2"],
        "R3": TransformScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["re:.*self_attn$"],
                    location="k_cache",
                ),
                TransformArgs(
                    targets=["re:.*self_attn$"],
                    location="q_attn",
                ),
            ],
        ),
        "R4": TransformScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["re:.*down_proj$"],
                    location="input",
                ),
                TransformArgs(
                    targets=["re:.*down_proj$"], location="weight_input", inverse=True
                ),
            ],
        ),
    }
)
