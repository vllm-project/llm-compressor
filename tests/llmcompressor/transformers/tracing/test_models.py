import pytest
from transformers import AutoModelForCausalLM

from llmcompressor.transformers.tracing import (
    TraceableIdefics3ForConditionalGeneration,
    TraceableLlavaForConditionalGeneration,
    TraceableMllamaForConditionalGeneration,
    TraceableQwen2_5_VLForConditionalGeneration,
    TraceableQwen2VLForConditionalGeneration,
)
from llmcompressor.transformers.tracing.debug import trace


@pytest.mark.parametrize(
    "model_id,model_class,targets",
    [
        ("meta-llama/Meta-Llama-3-8B-Instruct", AutoModelForCausalLM, None),
    ],
)
def test_text_trace(model_id, model_class, targets):
    trace(
        model_id,
        model_class,
        targets,
        ignore=[],
        modality="text",
        trust_remote_code=True,
    )


@pytest.mark.parametrize(
    "model_id,model_class,targets,ignore",
    [
        (
            "Qwen/Qwen2-VL-2B-Instruct",
            TraceableQwen2VLForConditionalGeneration,
            None,
            ["lm_head", "re:visual.*"],
        ),
        (
            "Qwen/Qwen2.5-VL-7B-Instruct",
            TraceableQwen2_5_VLForConditionalGeneration,
            None,
            ["lm_head", "re:visual.*"],
        ),
        (
            "mgoin/pixtral-12b",
            TraceableLlavaForConditionalGeneration,
            ["MistralDecoderLayer"],
            ["re:.*lm_head", "re:vision_tower.*", "re:multi_modal_projector.*"],
        ),
        (
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            TraceableMllamaForConditionalGeneration,
            None,
            ["re:.*lm_head", "re:multi_modal_projector.*", "re:vision_model.*"],
        ),
        (
            "llava-hf/llava-1.5-7b-hf",
            TraceableLlavaForConditionalGeneration,
            ["LlamaDecoderLayer"],
            ["re:.*lm_head", "re:vision_tower.*", "re:multi_modal_projector.*"],
        ),
        (
            "HuggingFaceM4/Idefics3-8B-Llama3",
            TraceableIdefics3ForConditionalGeneration,
            ["Idefics3EncoderLayer", "LlamaDecoderLayer"],
            ["re:.*lm_head", "re:model.vision_model.*", "re:model.connector.*"],
        ),
    ],
)
def test_vision_trace(model_id, model_class, targets, ignore):
    trace(
        model_id,
        model_class,
        targets,
        ignore=ignore,
        modality="vision",
        trust_remote_code=True,
    )
