from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
import yaml
from compressed_tensors.offload import get_offloaded_device, set_onload_device
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from datasets import Dataset
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoModelForCausalLM,
    GemmaConfig,
    GemmaForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from llmcompressor import oneshot
from llmcompressor.core import State
from llmcompressor.modifiers.qad import NormDistillationModifier
from llmcompressor.modifiers.qad.norm_distillation import _ScaledWeight
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.recipe import Recipe
from tests.testing_utils import requires_gpu


def _tiny_llama():
    torch.manual_seed(5)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    )
    model.config._attn_implementation = "eager"
    return model


def _batches(count=8):
    generator = torch.Generator().manual_seed(11)
    return [
        {
            "input_ids": torch.randint(0, 64, (1, 16), generator=generator),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }
        for _ in range(count)
    ]


def _tokenizer():
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    vocab.update({f"token{i}": i for i in range(4, 64)})
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )


def _dataset(batches):
    return Dataset.from_list(
        [{key: value[0].tolist() for key, value in b.items()} for b in batches]
    )


@torch.no_grad()
def _kl(student, teacher, batches):
    total = 0.0
    for batch in batches:
        target = teacher(**batch).logits.float().log_softmax(-1)
        prediction = student(**batch).logits.float().log_softmax(-1)
        total += F.kl_div(prediction, target, log_target=True, reduction="batchmean")
    return total / len(batches)


def _load_tensors(path):
    tensors = {}
    for file in path.glob("*.safetensors"):
        tensors.update(load_file(file))
    return tensors


def _scale_projections(model, factor):
    # Scaled projections, which the norm gains or weight scales can undo
    with torch.no_grad():
        for name, module in model.named_modules():
            if name.endswith(("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")):
                module.weight.mul_(factor)
    return model


def _compressed_checkpoint(tmp_path, recipe, factor=1.0):
    teacher = _tiny_llama()
    source = tmp_path / "source"
    teacher.save_pretrained(source)
    teacher.config.name_or_path = str(source)
    compressed = tmp_path / "compressed"
    oneshot(
        model=_scale_projections(deepcopy(teacher), factor),
        processor=_tokenizer(),
        recipe=recipe,
        output_dir=str(compressed),
    )
    return teacher, compressed


def _distill_checkpoint(tmp_path, teacher, compressed, batches, **kwargs):
    distilled = tmp_path / "distilled"
    oneshot(
        model=AutoModelForCausalLM.from_pretrained(compressed),
        processor=_tokenizer(),
        dataset=_dataset(batches),
        recipe=NormDistillationModifier(teacher=teacher, **kwargs),
        output_dir=str(distilled),
        num_calibration_samples=len(batches),
    )
    return distilled


def _changed_tensors(before, after):
    assert before.keys() == after.keys()
    return {
        name
        for name in before
        if before[name].dtype != after[name].dtype
        or not torch.equal(before[name], after[name])
    }


def test_compressed_checkpoint_changes_only_norm_gains(tmp_path):
    # An already-compressed W4A16 checkpoint
    teacher, compressed = _compressed_checkpoint(
        tmp_path,
        QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
    )
    student = AutoModelForCausalLM.from_pretrained(compressed)
    assert hasattr(student.model.layers[0].mlp.up_proj, "weight_packed")
    batches = _batches()

    distilled = _distill_checkpoint(
        tmp_path, teacher, compressed, batches, num_epochs=4, lr=0.01
    )

    before, after = _load_tensors(compressed), _load_tensors(distilled)
    changed = _changed_tensors(before, after)
    norms = {name for name in before if name.endswith("norm.weight")}
    # Two layers with two norms each, plus the final norm
    assert changed == norms and len(norms) == 5

    # The distilled checkpoint reloads compressed and is closer to the teacher
    original = AutoModelForCausalLM.from_pretrained(compressed)
    reloaded = AutoModelForCausalLM.from_pretrained(distilled)
    assert hasattr(reloaded.model.layers[0].mlp.up_proj, "weight_packed")
    assert _kl(reloaded, teacher, batches) < _kl(original, teacher, batches)


_FP8_BLOCK_WEIGHTS = {
    "group_0": {
        "targets": ["Linear"],
        "weights": {
            "num_bits": 8,
            "type": "float",
            "symmetric": True,
            "strategy": "block",
            "block_structure": [32, 64],
        },
    }
}


@pytest.mark.parametrize(
    "recipe",
    [
        QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
        QuantizationModifier(config_groups=_FP8_BLOCK_WEIGHTS, ignore=["lm_head"]),
    ],
    ids=["w4a16", "fp8-block"],
)
def test_compressed_checkpoint_changes_only_weight_scales(tmp_path, recipe):
    teacher, compressed = _compressed_checkpoint(tmp_path, recipe, factor=0.8)
    batches = _batches()

    distilled = _distill_checkpoint(
        tmp_path,
        teacher,
        compressed,
        batches,
        tune_norms=False,
        tune_scales=True,
        num_epochs=4,
        lr=0.01,
    )

    before, after = _load_tensors(compressed), _load_tensors(distilled)
    changed = _changed_tensors(before, after)
    scales = {name for name in before if name.endswith("weight_scale")}
    # Seven projections in each of two layers
    assert changed and changed <= scales and len(scales) == 14

    original = AutoModelForCausalLM.from_pretrained(compressed)
    reloaded = AutoModelForCausalLM.from_pretrained(distilled)
    assert _kl(reloaded, teacher, batches) < _kl(original, teacher, batches)


def test_scale_tuning_after_quantization_in_one_recipe(tmp_path):
    # After a quantization modifier the forward pass fake-quantizes the weights:
    # scale tuning must keep the quantized values that compression produces
    teacher = _tiny_llama()
    teacher.save_pretrained(tmp_path / "source")
    teacher.config.name_or_path = str(tmp_path / "source")
    batches = _batches()

    outputs = {}
    for name, extra in (("quantized", []), ("tuned", ["distill"])):
        recipe = [
            QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
        ]
        if extra:
            recipe.append(
                NormDistillationModifier(
                    teacher=teacher,
                    tune_norms=False,
                    tune_scales=True,
                    num_epochs=4,
                    lr=0.01,
                )
            )
        outputs[name] = tmp_path / name
        oneshot(
            model=deepcopy(teacher),
            processor=_tokenizer(),
            dataset=_dataset(batches),
            recipe=recipe,
            output_dir=str(outputs[name]),
            num_calibration_samples=len(batches),
        )

    before = _load_tensors(outputs["quantized"])
    after = _load_tensors(outputs["tuned"])
    changed = _changed_tensors(before, after)
    assert changed and all(name.endswith("weight_scale") for name in changed)

    original = AutoModelForCausalLM.from_pretrained(outputs["quantized"])
    tuned = AutoModelForCausalLM.from_pretrained(outputs["tuned"])
    assert _kl(tuned, teacher, batches) < _kl(original, teacher, batches)


@pytest.mark.parametrize(
    "weights,shape",
    [
        ({"strategy": "tensor"}, (1,)),
        ({"strategy": "channel"}, (96, 1)),
        ({"strategy": "group", "group_size": 32}, (96, 4)),
        (
            {
                "strategy": "tensor_group",
                "group_size": 16,
                "type": "float",
                "num_bits": 4,
            },
            (96, 8),
        ),
        # Blocks that do not divide the weight
        ({"strategy": "block", "block_structure": [64, 48]}, (2, 3)),
    ],
)
def test_scale_gradient_matches_autograd(weights, shape):
    module = torch.nn.Linear(128, 96, bias=False)
    module.quantization_scheme = QuantizationScheme(
        targets=["Linear"], weights=QuantizationArgs(**weights)
    )
    module.weight_scale = torch.nn.Parameter(
        torch.rand(shape) + 0.5, requires_grad=False
    )
    scaled = _ScaledWeight(module)
    scaled.snap()
    module.weight.grad = torch.randn_like(module.weight)

    # The forward pass uses the dequantized weight times the expanded multiplier
    ratio = torch.rand(shape, requires_grad=True)
    (
        module.weight.grad * module.weight.detach() * scaled.expand(ratio)
    ).sum().backward()
    torch.testing.assert_close(scaled.ratio_grad(), ratio.grad)


def test_offset_norms_through_oneshot(tmp_path):
    # oneshot swaps Gemma's (1 + weight) norms for standard norms during
    # calibration and restores the original modules before finalization
    torch.manual_seed(5)
    teacher = GemmaForCausalLM(
        GemmaConfig(
            vocab_size=64,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            max_position_embeddings=32,
        )
    )
    teacher.config._attn_implementation = "eager"
    teacher.save_pretrained(tmp_path / "source")
    teacher.config.name_or_path = str(tmp_path / "source")
    student = deepcopy(teacher)
    _scale_projections(student, 0.8)
    reference = deepcopy(student)
    batches = _batches()

    oneshot(
        model=student,
        processor=_tokenizer(),
        dataset=_dataset(batches),
        recipe=NormDistillationModifier(teacher=teacher, num_epochs=4, lr=0.01),
        num_calibration_samples=len(batches),
    )

    assert isinstance(student.model.norm, type(teacher.model.norm))
    changed = {
        name
        for (name, param), ref in zip(
            student.named_parameters(), reference.parameters(), strict=True
        )
        if not torch.equal(param, ref)
    }
    assert changed and all(name.endswith("norm.weight") for name in changed)
    assert _kl(student, teacher, batches) < _kl(reference, teacher, batches)


def _distill(model, teacher, batches, **kwargs):
    state = State(model=model)
    state.data.calib = batches
    modifier = NormDistillationModifier(teacher=teacher, **kwargs)
    modifier.initialize(state)
    modifier.finalize(state)
    return modifier


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_trains_only_targeted_gains(dtype):
    teacher = _tiny_llama().to(dtype)
    student = deepcopy(teacher)
    _scale_projections(student, 0.8)
    reference = deepcopy(student)
    batches = _batches()

    _distill(
        student,
        teacher,
        batches,
        ignore=["model.norm"],
        num_epochs=4,
        lr=0.01,
        gradient_accumulation_steps=2,
    )

    trained = {
        f"model.layers.{i}.{norm}.weight"
        for i in range(2)
        for norm in ("input_layernorm", "post_attention_layernorm")
    }
    for (name, param), ref in zip(
        student.named_parameters(), reference.parameters(), strict=True
    ):
        assert param.dtype == dtype and param.requires_grad == ref.requires_grad
        assert torch.equal(param, ref) != (name in trained), name
    assert _kl(student, teacher, batches) < _kl(reference, teacher, batches)


@requires_gpu
def test_offloaded_student():
    # oneshot onloads a CPU model to the accelerator on every forward pass, which
    # gradients do not reach: the modifier onloads it for training
    teacher = _tiny_llama()
    teacher.config.tie_word_embeddings = True
    teacher.tie_weights()
    student = deepcopy(teacher)
    _scale_projections(student, 0.8)
    reference = deepcopy(student)
    batches = _batches()
    device = torch.accelerator.current_accelerator()
    set_onload_device(student, device)

    def tied():
        return (
            student.lm_head._parameters.offloaded_values["weight"]
            is student.model.embed_tokens._parameters.offloaded_values["weight"]
        )

    assert tied()

    _distill(student, teacher.to(device), batches, num_epochs=4, lr=0.01)

    assert get_offloaded_device(student.model.norm).type == "cpu"
    assert tied()
    trained = dict(student.named_parameters(remove_duplicate=False))
    changed = {
        name
        for name, ref in reference.named_parameters(remove_duplicate=False)
        if not torch.equal(trained[name].cpu(), ref)
    }
    assert changed and all(name.endswith("norm.weight") for name in changed)


def test_keeps_original_gains_without_validation_improvement():
    teacher = _tiny_llama()
    student = deepcopy(teacher)
    with torch.no_grad():
        student.model.layers[0].mlp.down_proj.weight.mul_(0.5)
    reference = deepcopy(student)

    # A diverging learning rate never beats the original gains on validation
    _distill(student, teacher, _batches(), lr=100.0)

    for param, ref in zip(student.parameters(), reference.parameters(), strict=True):
        assert torch.equal(param, ref)


def test_requires_calibration_data():
    with pytest.raises(ValueError, match="calibration dataset"):
        _distill(_tiny_llama(), _tiny_llama(), None)


def test_requires_norms_or_scales():
    with pytest.raises(ValueError, match="norms, scales, or both"):
        _distill(_tiny_llama(), _tiny_llama(), _batches(), tune_norms=False)


def test_requires_targets():
    with pytest.raises(ValueError, match="no normalization gains"):
        _distill(_tiny_llama(), _tiny_llama(), _batches(), targets=["re:.*proj$"])


def test_recipe_records_teacher_path():
    teacher = _tiny_llama()
    teacher.config._name_or_path = "org/teacher"
    recipe = Recipe.from_modifiers(NormDistillationModifier(teacher=teacher))
    args = yaml.safe_load(recipe.yaml())["default_stage"]["default_modifiers"]
    assert args["NormDistillationModifier"]["teacher"] == "org/teacher"
