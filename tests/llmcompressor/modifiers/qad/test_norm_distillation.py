from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
import yaml
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
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.recipe import Recipe


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


def test_compressed_checkpoint_changes_only_norm_gains(tmp_path):
    teacher = _tiny_llama()
    source = tmp_path / "source"
    teacher.save_pretrained(source)
    teacher.config.name_or_path = str(source)
    tokenizer = _tokenizer()
    batches = _batches()

    # An already-compressed W4A16 checkpoint
    compressed = tmp_path / "compressed"
    oneshot(
        model=deepcopy(teacher),
        processor=tokenizer,
        recipe=QuantizationModifier(
            targets="Linear", scheme="W4A16", ignore=["lm_head"]
        ),
        output_dir=str(compressed),
    )
    student = AutoModelForCausalLM.from_pretrained(compressed)
    assert hasattr(student.model.layers[0].mlp.up_proj, "weight_packed")

    distilled = tmp_path / "distilled"
    oneshot(
        model=student,
        processor=tokenizer,
        dataset=_dataset(batches),
        recipe=NormDistillationModifier(teacher=teacher, num_epochs=4, lr=0.01),
        output_dir=str(distilled),
        num_calibration_samples=len(batches),
    )

    before, after = _load_tensors(compressed), _load_tensors(distilled)
    assert before.keys() == after.keys()
    changed = {
        name
        for name in before
        if before[name].dtype != after[name].dtype
        or not torch.equal(before[name], after[name])
    }
    norms = {name for name in before if name.endswith("norm.weight")}
    # Two layers with two norms each, plus the final norm
    assert changed == norms and len(norms) == 5

    # The distilled checkpoint reloads compressed and is closer to the teacher
    original = AutoModelForCausalLM.from_pretrained(compressed)
    reloaded = AutoModelForCausalLM.from_pretrained(distilled)
    assert hasattr(reloaded.model.layers[0].mlp.up_proj, "weight_packed")
    assert _kl(reloaded, teacher, batches) < _kl(original, teacher, batches)


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
    with torch.no_grad():
        for name, module in student.named_modules():
            if name.endswith(("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")):
                module.weight.mul_(0.8)
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
    # Scaled projections after each norm, which the norm gains can undo
    with torch.no_grad():
        for name, module in student.named_modules():
            if name.endswith(("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")):
                module.weight.mul_(0.8)
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


def test_requires_targets():
    with pytest.raises(ValueError, match="no normalization gains"):
        _distill(_tiny_llama(), _tiny_llama(), _batches(), targets=["re:.*proj$"])


def test_recipe_records_teacher_path():
    teacher = _tiny_llama()
    teacher.config._name_or_path = "org/teacher"
    recipe = Recipe.from_modifiers(NormDistillationModifier(teacher=teacher))
    args = yaml.safe_load(recipe.yaml())["default_stage"]["default_modifiers"]
    assert args["NormDistillationModifier"]["teacher"] == "org/teacher"
