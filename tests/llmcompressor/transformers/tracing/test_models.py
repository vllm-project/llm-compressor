import pytest
import torch
import transformers
from compressed_tensors.utils import patch_attr
from compressed_tensors.utils.match import match_named_modules

from llmcompressor.modeling.kimi_k3 import KimiK3ForConditionalGeneration
from llmcompressor.pipelines.sequential.helpers import (
    _resolve_sequential_targets,
    find_modules_outside_subgraphs,
    get_sequential_ancestors,
)
from llmcompressor.transformers.tracing.debug import trace
from llmcompressor.utils.helpers import calibration_forward_context
from llmcompressor.utils.pytorch.module import get_no_split_params
from tests.testing_utils import requires_hf_token

# Modules called inside wrapped functions which cannot execute on meta tensors are not
# attributed to any subgraph and stay onloaded (trace_subgraphs warns about them)
KNOWN_UNOWNED_MODULES = {
    "inference-optimization/Llama-4-Scout-1.7B-0.4B-Instruct": {
        "multi_modal_projector.linear_1",  # wrapped fn mutates an immutable_dict
    },
}


@requires_hf_token
@pytest.mark.parametrize(
    "model_id,model_class,targets,modality,backends",
    [
        # --- text ---
        (
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "AutoModelForCausalLM",
            None,
            "text",
            [],
        ),
        (
            "inference-optimization/gemma-4-1B-0.8B-tiny",
            "Gemma4ForConditionalGeneration",
            ["Gemma4TextDecoderLayer"],
            "text",
            [],
        ),
        (
            "inference-optimization/DSV4-tiny-empty",
            "AutoModelForCausalLM",
            ["DeepseekV4DecoderLayer"],
            "text",
            [],
        ),
        (
            "inference-optimization/Qwen3-1.6B-A0.9B",
            "AutoModelForCausalLM",
            None,
            "text",
            [],
        ),
        (
            "inference-optimization/Llama-3.2-0.5B-Instruct",
            "AutoModelForCausalLM",
            None,
            "text",
            [],
        ),
        (
            "inference-optimization/GLM-5.2-0.8B-A0.8B",
            "AutoModelForCausalLM",
            None,
            "text",
            [],
        ),
        (
            "inference-optimization/Phi-3.5-MoE-0.8B-A0.2B",
            "AutoModelForCausalLM",
            None,
            "text",
            [],
        ),
        (
            "inference-optimization/gpt-oss-2.5B-A1.3B",
            "AutoModelForCausalLM",
            None,
            "text",
            [],
        ),
        (
            "inference-optimization/DeepSeek-V4-Pro-0.5B-A0.37B",
            "AutoModelForCausalLM",
            ["DeepseekV4DecoderLayer"],
            "text",
            [],
        ),
        (
            "inference-optimization/Qwen3.8-1.0B-A0.6B",
            "AutoModelForCausalLM",
            None,
            "text",
            [],
        ),
        (
            "inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B",
            "Qwen4ExpForConditionalGeneration",
            ["Qwen4ExpTextDecoderLayer"],
            "text",
            ["transformers>=5.17.0"],
        ),
        (
            "inference-optimization/NemotronH-0.3B-A0.3B",
            "AutoModelForCausalLM",
            None,
            "text",
            [],
        ),
        (
            "inference-optimization/GLM-5.3-0.6B-A0.4B",
            "AutoModelForCausalLM",
            None,
            "text",
            [],
        ),
        # --- vision ---
        (
            "inference-optimization/Llama-4-Scout-1.7B-0.4B-Instruct",
            "Llama4ForConditionalGeneration",
            ["Llama4TextDecoderLayer"],
            "vision",
            [],
        ),
        (
            "inference-optimization/Qwen3-VL-1.0B-A0.4B-Instruct",
            "Qwen3VLMoeForConditionalGeneration",
            ["Qwen3VLMoeTextDecoderLayer"],
            "vision",
            ["torchvision"],
        ),
        (
            "inference-optimization/Inkling-0.6B-A0.6B",
            "InklingForConditionalGeneration",
            ["InklingDecoderLayer"],
            "vision",
            [],
        ),
        (
            "inference-optimization/Kimi-K3-0.40B",
            "KimiK3ForConditionalGeneration",
            ["KimiDecoderLayer"],
            "vision",
            ["einops", "fla-core", "tiktoken"],
        ),
        (
            "inference-optimization/gemma-4-unified-0.8B-tiny",
            "Gemma4UnifiedForConditionalGeneration",
            ["Gemma4UnifiedTextDecoderLayer"],
            "vision",
            [],
        ),
        (
            "inference-optimization/granite-vision-4.1-0.2B-tiny",
            "Granite4VisionForConditionalGeneration",
            ["Granite4VisionTextDecoderLayer"],
            "vision",
            [],
        ),
        (
            "inference-optimization/Qwen3-VL-Reranker-0.1B-tiny",
            "Qwen3VLForConditionalGeneration",
            ["Qwen3VLTextDecoderLayer"],
            "vision",
            ["torchvision"],
        ),
        # dense Qwen3-VL: the vision tower is not traceable and must stay out
        # of the traced subgraphs (#3071)
        (
            "Qwen/Qwen3-VL-2B-Instruct",
            "Qwen3VLForConditionalGeneration",
            ["Qwen3VLTextDecoderLayer"],
            "vision",
            ["torchvision"],
        ),
        (
            "inference-optimization/GLM-5.3-Flash-0.1B-A0.1B",
            "Glm5NextForConditionalGeneration",
            ["Glm5NextTextDecoderLayer"],
            "vision",
            ["transformers>=5.16.0"],
        ),
        (
            "inference-optimization/Qwen3.6-8B-A1.6B",
            "Qwen3_5MoeForConditionalGeneration",
            ["Qwen3_5MoeDecoderLayer"],
            "vision",
            ["torchvision"],
        ),
        # --- audio ---
        (
            "openai/whisper-large-v3",
            "WhisperForConditionalGeneration",
            ["WhisperDecoderLayer"],
            "audio",
            ["librosa", "soundfile", "torchcodec"],
        ),
    ],
)
def test_model_trace(model_id, model_class, targets, modality, backends):
    for backend in backends:
        pytest.importorskip(*backend.split(">="))

    if "kimi" in model_id.lower():
        model_class = KimiK3ForConditionalGeneration
        trust_remote_code = True
    else:
        model_class = getattr(transformers, model_class)
        trust_remote_code = False

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

    # every parameterized module which executes and is not inlined as an ancestor
    # must be owned by a subgraph, including modules called inside wrapped functions
    executed = run_model_with_hooks(model, sample_input)
    resolved = _resolve_sequential_targets(
        model, get_sequential_targets(model, targets)
    )
    ancestors = get_sequential_ancestors(model, resolved)
    missing = [
        name
        for name, module in find_modules_outside_subgraphs(model, subgraphs).items()
        if name in executed and module not in ancestors and module._parameters
    ]
    assert set(missing) <= KNOWN_UNOWNED_MODULES.get(model_id, set()), missing


def run_model_with_hooks(model, sample_input) -> set[str]:
    """
    Names of modules which finished executing during a plain forward on meta. The
    forward may not complete for data-dependent ops (e.g. MoE routing), in which case
    the modules executed before the failure are still a valid witness set
    """
    executed = set()
    handles = [
        module.register_forward_hook(lambda *_, _name=name: executed.add(_name))
        for name, module in model.named_modules()
        if name
    ]
    try:
        with calibration_forward_context(model):
            with patch_attr(torch.compiler, "_is_compiling_flag", True):
                model(**sample_input)
    except (RuntimeError, NotImplementedError):
        pass
    finally:
        for handle in handles:
            handle.remove()
    return executed


def get_sequential_targets(model, sequential_targets):
    """Resolve a parametrized target spec the same way `trace()` does."""
    if sequential_targets is None:
        sequential_targets = get_no_split_params(model)
    if isinstance(sequential_targets, str):
        sequential_targets = [sequential_targets]
    return sequential_targets


def get_target_modules(model, sequential_targets):
    sequential_targets = get_sequential_targets(model, sequential_targets)
    return set(module for _, module in match_named_modules(model, sequential_targets))


def run_subgraphs(model, subgraphs, inputs):
    namespace = dict()
    namespace.update(inputs)
    for subgraph in subgraphs:
        inputs = {name: namespace[name] for name in subgraph.input_names}
        output = subgraph.forward(model, **inputs)
        namespace.update(output)

    return output
