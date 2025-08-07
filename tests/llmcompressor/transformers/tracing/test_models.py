import os

import pytest
from transformers import (
    AutoModelForCausalLM,
    Gemma3ForConditionalGeneration,
    Idefics3ForConditionalGeneration,
    Llama4ForConditionalGeneration,
    LlavaForConditionalGeneration,
    Mistral3ForConditionalGeneration,
    MllamaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    WhisperForConditionalGeneration,
)

from llmcompressor.pipelines.sequential.helpers import match_modules
from llmcompressor.transformers.tracing.debug import trace
from llmcompressor.utils.pytorch.module import get_no_split_params


@pytest.mark.skipif(
    (not os.getenv("HF_TOKEN")),
    reason="Skipping tracing tests requiring gated model access",
)
@pytest.mark.parametrize(
    "model_id,model_class,targets,modality,backends",
    [
        # --- text ---
        ("meta-llama/Meta-Llama-3-8B-Instruct", AutoModelForCausalLM, None, "text", []),
        (
            "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            AutoModelForCausalLM,
            None,
            "text",
            [],
        ),
        (
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            AutoModelForCausalLM,
            None,
            "text",
            [],
        ),
        (
            "ibm-granite/granite-20b-code-instruct-8k",
            AutoModelForCausalLM,
            None,
            "text",
            [],
        ),
        ("unsloth/DeepSeek-R1-0528-BF16", AutoModelForCausalLM, None, "text", []),
        # --- vision ---
        (
            "HuggingFaceM4/Idefics3-8B-Llama3",
            Idefics3ForConditionalGeneration,
            ["LlamaDecoderLayer"],
            "vision",
            [],
        ),
        (
            "llava-hf/llava-1.5-7b-hf",
            LlavaForConditionalGeneration,
            ["LlamaDecoderLayer"],
            "vision",
            [],
        ),
        (
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            MllamaForConditionalGeneration,
            ["MllamaSelfAttentionDecoderLayer"],
            "vision",
            [],
        ),
        # skip phi3_v because of its processor is annoying and requires special code
        (
            "mgoin/pixtral-12b",
            LlavaForConditionalGeneration,
            ["MistralDecoderLayer"],
            "vision",
            [],
        ),
        (
            "Qwen/Qwen2.5-VL-7B-Instruct",
            Qwen2_5_VLForConditionalGeneration,
            ["Qwen2_5_VLDecoderLayer"],
            "vision",
            ["torchvision"],
        ),
        (
            "Qwen/Qwen2-VL-2B-Instruct",
            Qwen2VLForConditionalGeneration,
            ["Qwen2VLDecoderLayer"],
            "vision",
            ["torchvision"],
        ),
        (
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            Mistral3ForConditionalGeneration,
            ["MistralDecoderLayer"],
            "vision",
            [],
        ),
        (
            "google/gemma-3-4b-it",
            Gemma3ForConditionalGeneration,
            ["Gemma3DecoderLayer"],
            "vision",
            [],
        ),
        (
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            Llama4ForConditionalGeneration,
            "Llama4TextDecoderLayer",
            "vision",
            [],
        ),
        (
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
            Llama4ForConditionalGeneration,
            "Llama4TextDecoderLayer",
            "vision",
            [],
        ),
        # --- audio ---
        (
            "openai/whisper-large-v3",
            WhisperForConditionalGeneration,
            ["WhisperDecoderLayer"],
            "audio",
            ["librosa", "soundfile", "torchcodec"],
        ),
    ],
)
def test_model_trace(model_id, model_class, targets, modality, backends):
    for backend in backends:
        pytest.importorskip(backend)

    model, subgraphs, sample_input = trace(
        model_id,
        model_class,
        targets,
        modality=modality,
        trust_remote_code=True,
        skip_weights=True,
    )

    target_modules = get_target_modules(model, targets)
    assert len(subgraphs) == len(target_modules) + 1


def get_target_modules(model, sequential_targets):
    if sequential_targets is None:
        sequential_targets = get_no_split_params(model)
    if isinstance(sequential_targets, str):
        sequential_targets = [sequential_targets]

    return match_modules(model, sequential_targets)


def run_subgraphs(model, subgraphs, inputs):
    namespace = dict()
    namespace.update(inputs)
    for subgraph in subgraphs:
        inputs = {name: namespace[name] for name in subgraph.input_names}
        output = subgraph.forward(model, **inputs)
        namespace.update(output)

    return output
