import pytest

pytest.importorskip("transformers.models.llama4")

import torch  # noqa: E402
from transformers import (  # noqa: E402
    Llama4Config,
    Llama4ForConditionalGeneration,
    Llama4TextConfig,
    Llama4VisionConfig,
)

from llmcompressor.args.dataset_arguments import DatasetArguments  # noqa: E402
from llmcompressor.modeling.moe.context import moe_calibration_context  # noqa: E402
from llmcompressor.modeling.moe.linearize import linearize_moe  # noqa: E402
from llmcompressor.modeling.moe.llama4 import Llama4LinearExperts  # noqa: E402
from llmcompressor.pipelines.sequential.helpers import trace_subgraphs  # noqa: E402
from llmcompressor.utils.dev import skip_weights_initialize  # noqa: E402


def test_llama4_decoder_layer_trace():
    """The full decoder layer remains a valid sequential target for Llama4."""
    config = Llama4Config(
        text_config=Llama4TextConfig(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            intermediate_size_mlp=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=32,
            max_position_embeddings=128,
            num_local_experts=2,
            num_experts_per_tok=1,
            moe_layers=[0],
            pad_token_id=0,
            boi_token_index=1,
            eoi_token_index=2,
            image_token_index=3,
        ),
        vision_config=Llama4VisionConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=14,
            patch_size=14,
            vision_output_dim=128,
            projector_input_dim=128,
            projector_output_dim=128,
        ),
        boi_token_index=1,
        eoi_token_index=2,
        image_token_index=3,
    )

    with skip_weights_initialize():
        model = Llama4ForConditionalGeneration(config)
    linearize_moe(model)

    subgraphs = trace_subgraphs(
        model,
        sample_input={"input_ids": torch.zeros(1, 8, dtype=torch.long)},
        sequential_targets=["Llama4TextDecoderLayer"],
        ignore=DatasetArguments().tracing_ignore,
    )

    assert subgraphs


def test_llama4_all_expert_calibration_is_chunked():
    """Chunking preserves all-expert outputs while bounding projection activations."""
    config = Llama4TextConfig(
        hidden_size=4,
        intermediate_size=8,
        num_local_experts=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
    )
    experts = Llama4LinearExperts(2, 4, 8, config)
    inputs = torch.randn(8, 4)

    with moe_calibration_context():
        expected = torch.cat(
            [
                experts[index](inputs).view(2, 4, 4)[index]
                for index in range(experts.num_experts)
            ]
        )

        old_chunk_size = experts._calibration_chunk_size
        experts._calibration_chunk_size = 2
        try:
            actual = experts(inputs)
        finally:
            experts._calibration_chunk_size = old_chunk_size

    torch.testing.assert_close(actual, expected)
