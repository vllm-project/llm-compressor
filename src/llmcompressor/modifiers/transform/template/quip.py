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
