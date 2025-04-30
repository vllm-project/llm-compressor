import pytest
from transformers import AutoModelForCausalLM, WhisperForConditionalGeneration

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
            "HuggingFaceM4/Idefics3-8B-Llama3",
            TraceableIdefics3ForConditionalGeneration,
            ["LlamaDecoderLayer"],
            [],
        ),
        (
            "llava-hf/llava-1.5-7b-hf",
            TraceableLlavaForConditionalGeneration,
            ["LlamaDecoderLayer"],
            [],
        ),
        (
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            TraceableMllamaForConditionalGeneration,
            ["MllamaSelfAttentionDecoderLayer"],
            [],
        ),
        # skip phi3_v because of its processor is annoying and requires special code
        (
            "mgoin/pixtral-12b",
            TraceableLlavaForConditionalGeneration,
            ["MistralDecoderLayer"],
            [],
        ),
        (
            "Qwen/Qwen2.5-VL-7B-Instruct",
            TraceableQwen2_5_VLForConditionalGeneration,
            ["Qwen2_5_VLDecoderLayer"],
            [],
        ),
        (
            "Qwen/Qwen2-VL-2B-Instruct",
            TraceableQwen2VLForConditionalGeneration,
            ["Qwen2VLDecoderLayer"],
            [],
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


@pytest.mark.parametrize(
    "model_id,model_class,targets,ignore",
    [
        (
            "openai/whisper-large-v3",
            WhisperForConditionalGeneration,
            None,
            [],
        ),
    ],
)
def test_audio_trace(model_id, model_class, targets, ignore):
    pytest.importorskip("librosa")
    pytest.importorskip("soundfile")

    trace(
        model_id,
        model_class,
        targets,
        ignore=ignore,
        modality="audio",
        trust_remote_code=True,
    )
