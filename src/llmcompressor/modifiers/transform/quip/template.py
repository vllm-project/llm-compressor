from compressed_tensors.transform import TransformArgs, TransformConfig, TransformScheme

QUIP = TransformConfig(
    config_groups={
        "v": TransformScheme(
            type="random-hadamard",
            apply=[
                TransformArgs(
                    targets=["Linear"],
                    location="input",  # non-mergable
                    ignore="lm_head",
                ),
                TransformArgs(
                    targets=["Linear"],
                    location="weight_input",
                    inverse=True,
                    ignore="lm_head",
                ),
            ],
            randomize=True,
        ),
        "u": TransformScheme(
            type="random-hadamard",
            apply=[
                TransformArgs(
                    targets=["Linear"],
                    location="weight_output",
                    ignore="lm_head",
                ),
                TransformArgs(
                    targets=["Linear"],
                    location="output",  # non-mergable
                    inverse=True,
                    ignore="lm_head",
                ),
            ],
            randomize=True,
        ),
    }
)

# https://github.com/vllm-project/llm-compressor/blob/b43b27a2f277a5e62be4f8c713b84fd1c7aa116b/weight_transform.py#L24-L105
QUIP_ONLINE = TransformConfig(
    config_groups={
        "u_transform_q_o_down_proj": TransformScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=[
                        "re:.*.attn.q_proj$",
                        "re:.*.attn.o_proj$",
                        "re:.*.mlp.down_proj$",
                    ],
                    location="weight_input",
                )
            ],
        ),
        "u_transform_k_v_proj": TransformScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["re:.*.attn.k_proj$", "re:.*.attn.v_proj$"],
                    location="weight_input",
                )
            ],
        ),
        "u_transform_gate_up_proj": TransformScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["re:.*.mlp.gate_proj$", "re:.*.mlp.up_proj$"],
                    location="weight_input",
                )
            ],
        ),
        "v_transform_linear": TransformScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["Linear"],
                    location="weight_output",
                    ignore=["re:.*.mlp.down_proj$", "lm_head"],
                    inverse=True,
                )
            ],
        ),
        "v_transform_down_proj": TransformScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["re:.*.mlp.down_proj$"],
                    location="weight_output",
                    inverse=True,
                )
            ],
        ),
    }
)
