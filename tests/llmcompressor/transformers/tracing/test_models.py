import pytest
from transformers import (
    AutoModelForCausalLM,
    Gemma3ForConditionalGeneration,
    Idefics3ForConditionalGeneration,
    LlavaForConditionalGeneration,
    MllamaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    WhisperForConditionalGeneration,
)

from llmcompressor.transformers.tracing.debug import trace


@pytest.mark.parametrize(
    "model_id,model_class,targets",
    [
        ("meta-llama/Meta-Llama-3-8B-Instruct", AutoModelForCausalLM, None),
    ],
)
def test_text_trace(model_id, model_class, targets):
    model, subgraphs, sample_input = trace(
        model_id,
        model_class,
        targets,
        modality="text",
        trust_remote_code=True,
        skip_weights=True,
        device_map="auto",
    )


@pytest.mark.parametrize(
    "model_id,model_class,targets",
    [
        (
            "HuggingFaceM4/Idefics3-8B-Llama3",
            Idefics3ForConditionalGeneration,
            ["LlamaDecoderLayer"],
        ),
        (
            "llava-hf/llava-1.5-7b-hf",
            LlavaForConditionalGeneration,
            ["LlamaDecoderLayer"],
        ),
        (
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            MllamaForConditionalGeneration,
            ["MllamaSelfAttentionDecoderLayer"],
        ),
        # skip phi3_v because of its processor is annoying and requires special code
        (
            "mgoin/pixtral-12b",
            LlavaForConditionalGeneration,
            ["MistralDecoderLayer"],
        ),
        (
            "Qwen/Qwen2.5-VL-7B-Instruct",
            Qwen2_5_VLForConditionalGeneration,
            ["Qwen2_5_VLDecoderLayer"],
        ),
        (
            "Qwen/Qwen2-VL-2B-Instruct",
            Qwen2VLForConditionalGeneration,
            ["Qwen2VLDecoderLayer"],
        ),
        (
            "google/gemma-3-4b-it",
            Gemma3ForConditionalGeneration,
            ["Gemma3DecoderLayer"],
        ),
    ],
)
def test_vision_trace(model_id, model_class, targets):
    model, subgraphs, sample_input = trace(
        model_id,
        model_class,
        targets,
        modality="vision",
        trust_remote_code=True,
        skip_weights=True,
        device_map="auto",
    )


@pytest.mark.parametrize(
    "model_id,model_class,targets",
    [
        (
            "openai/whisper-large-v3",
            WhisperForConditionalGeneration,
            ["WhisperDecoderLayer"],
        ),
    ],
)
def test_audio_trace(model_id, model_class, targets):
    pytest.importorskip("librosa")
    pytest.importorskip("soundfile")

    model, subgraphs, sample_input = trace(
        model_id,
        model_class,
        targets,
        modality="audio",
        trust_remote_code=True,
        skip_weights=True,
        device_map="auto",
    )


def run_subgraphs(model, subgraphs, inputs):
    namespace = dict()
    namespace.update(inputs)
    for subgraph in subgraphs:
        inputs = {name: namespace[name] for name in subgraph.input_names}
        output = subgraph.forward(model, **inputs)
        namespace.update(output)

    return output
