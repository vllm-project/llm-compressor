import pytest
from compressed_tensors.utils.match import match_named_modules
from transformers import (
    AutoModelForCausalLM,
    Gemma4ForConditionalGeneration,
    Gemma4UnifiedForConditionalGeneration,
    Glm5NextForConditionalGeneration,
    Granite4VisionForConditionalGeneration,
    InklingForConditionalGeneration,
    Llama4ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    Qwen4ExpForConditionalGeneration,
    WhisperForConditionalGeneration,
)

from llmcompressor.modeling.kimi_k3 import KimiK3ForConditionalGeneration
from llmcompressor.transformers.tracing.debug import trace
from llmcompressor.utils.pytorch.module import get_no_split_params
from tests.testing_utils import requires_hf_token


@requires_hf_token
@pytest.mark.parametrize(
    "model_id,model_class,targets,modality,backends",
    [
        # --- text ---
        ("meta-llama/Meta-Llama-3-8B-Instruct", AutoModelForCausalLM, None, "text", []),
        (
            "inference-optimization/gemma-4-1B-0.8B-tiny",
            Gemma4ForConditionalGeneration,
            ["Gemma4TextDecoderLayer"],
            "text",
            [],
        ),
        (
            "inference-optimization/DSV4-tiny-empty",
            AutoModelForCausalLM,
            ["DeepseekV4DecoderLayer"],
            "text",
            [],
        ),
        (
            "inference-optimization/Qwen3-1.6B-A0.9B",
            AutoModelForCausalLM,
            None,
            "text",
            [],
        ),
        (
            "inference-optimization/Llama-3.2-0.5B-Instruct",
            AutoModelForCausalLM,
            None,
            "text",
            [],
        ),
        (
            "inference-optimization/GLM-5.2-0.8B-A0.8B",
            AutoModelForCausalLM,
            None,
            "text",
            [],
        ),
        (
            "inference-optimization/Phi-3.5-MoE-0.8B-A0.2B",
            AutoModelForCausalLM,
            None,
            "text",
            [],
        ),
        (
            "inference-optimization/gpt-oss-2.5B-A1.3B",
            AutoModelForCausalLM,
            None,
            "text",
            [],
        ),
        (
            "inference-optimization/DeepSeek-V4-Pro-0.5B-A0.37B",
            AutoModelForCausalLM,
            ["DeepseekV4DecoderLayer"],
            "text",
            [],
        ),
        (
            "inference-optimization/Qwen3.8-1.0B-A0.6B",
            AutoModelForCausalLM,
            None,
            "text",
            [],
        ),
        (
            "inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B",
            Qwen4ExpForConditionalGeneration,
            ["Qwen4ExpTextDecoderLayer"],
            "text",
            [],
        ),
        (
            "inference-optimization/NemotronH-0.3B-A0.3B",
            AutoModelForCausalLM,
            None,
            "text",
            [],
        ),
        (
            "inference-optimization/GLM-5.3-0.6B-A0.4B",
            AutoModelForCausalLM,
            None,
            "text",
            [],
        ),
        # --- vision ---
        (
            "inference-optimization/Llama-4-Scout-1.7B-0.4B-Instruct",
            Llama4ForConditionalGeneration,
            ["Llama4TextDecoderLayer"],
            "vision",
            [],
        ),
        (
            "inference-optimization/Qwen3-VL-1.0B-A0.4B-Instruct",
            Qwen3VLMoeForConditionalGeneration,
            ["Qwen3VLMoeTextDecoderLayer"],
            "vision",
            ["torchvision"],
        ),
        (
            "inference-optimization/Inkling-0.6B-A0.6B",
            InklingForConditionalGeneration,
            ["InklingDecoderLayer"],
            "vision",
            [],
        ),
        (
            "inference-optimization/Kimi-K3-0.40B",
            KimiK3ForConditionalGeneration,
            ["KimiDecoderLayer"],
            "vision",
            ["einops", "fla-core", "tiktoken"],
        ),
        (
            "inference-optimization/gemma-4-unified-0.8B-tiny",
            Gemma4UnifiedForConditionalGeneration,
            ["Gemma4UnifiedTextDecoderLayer"],
            "vision",
            [],
        ),
        (
            "inference-optimization/granite-vision-4.1-0.2B-tiny",
            Granite4VisionForConditionalGeneration,
            ["Granite4VisionTextDecoderLayer"],
            "vision",
            [],
        ),
        (
            "inference-optimization/Qwen3-VL-Reranker-0.1B-tiny",
            Qwen3VLForConditionalGeneration,
            ["Qwen3VLTextDecoderLayer"],
            "vision",
            ["torchvision"],
        ),
        (
            "inference-optimization/GLM-5.3-Flash-0.1B-A0.1B",
            Glm5NextForConditionalGeneration,
            ["Glm5NextTextDecoderLayer"],
            "vision",
            [],
        ),
        (
            "inference-optimization/Qwen3.6-8B-A1.6B",
            Qwen3_5MoeForConditionalGeneration,
            ["Qwen3_5MoeDecoderLayer"],
            "vision",
            ["torchvision"],
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

    trust_remote_code = "kimi" in model_id.lower()
    model, subgraphs, sample_input = trace(
        model_id,
        model_class,
        targets,
        modality=modality,
        trust_remote_code=trust_remote_code,
        device_map="meta",
        skip_weights=True,
    )

    target_modules = get_target_modules(model, targets)
    assert len(subgraphs) == len(target_modules) + 1


def get_target_modules(model, sequential_targets):
    if sequential_targets is None:
        sequential_targets = get_no_split_params(model)
    if isinstance(sequential_targets, str):
        sequential_targets = [sequential_targets]

    return set(module for _, module in match_named_modules(model, sequential_targets))


def run_subgraphs(model, subgraphs, inputs):
    namespace = dict()
    namespace.update(inputs)
    for subgraph in subgraphs:
        inputs = {name: namespace[name] for name in subgraph.input_names}
        output = subgraph.forward(model, **inputs)
        namespace.update(output)

    return output
