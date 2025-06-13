from compressed_tensors.transform import TransformArgs, TransformConfig, TransformScheme

LLAMA_SPINQUANT = TransformConfig(
    transform_groups={
        "R1": TransformScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["embed_tokens", "o_proj", "down_proj"],
                    location="weight_output",
                ),
                TransformArgs(
                    targets=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "up_proj",
                        "gate_proj",
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
                    targets=["v_proj"],
                    location="weight_output",
                ),
                TransformArgs(
                    targets=["o_proj"], location="weight_input", inverse=True
                ),
            ],
        ),
        "R3": TransformScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["self_attn"],
                    location="k_cache",
                ),
                TransformArgs(
                    targets=["self_attn"],
                    location="q_attn",
                ),
            ],
        ),
        "R4": TransformScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["down_proj"],
                    location="input",
                ),
                TransformArgs(
                    targets=["down_proj"], location="weight_input", inverse=True
                ),
            ],
        ),
    }
)
