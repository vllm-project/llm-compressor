from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from pydantic import ValidationError
from torch import nn

from llmcompressor.core import Event, State
from llmcompressor.modifiers.transform.itersmooth import (
    IterSmoothMapping,
    IterSmoothModifier,
)

pytestmark = pytest.mark.unit


def prepare(model, mappings, **kwargs):
    modifier = IterSmoothModifier(mappings=mappings, **kwargs)
    state = State(model=model)
    state.data.calib = [object()]
    modifier.initialize(state)
    modifier.on_calibration_start(state, Event())
    return modifier, state


def finish(modifier, state):
    modifier.on_sequential_epoch_end(state, Event(), list(state.model.modules()))
    modifier.on_calibration_end(state, Event())
    modifier.finalize(state)


class NormLinears(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.norm = nn.LayerNorm(4, bias=bias)
        self.fc1 = nn.Linear(4, 3, bias=bias)
        self.fc2 = nn.Linear(4, 5, bias=bias)

    def forward(self, x):
        x = self.norm(x)
        return torch.cat([self.fc1(x), self.fc2(x)], dim=-1)


@pytest.mark.parametrize("symmetric", [True, False])
def test_norm_linears_equivalence_and_reload(symmetric):
    torch.manual_seed(1)
    model = NormLinears().eval()
    with torch.no_grad():
        model.norm.bias.fill_(3)
        model.norm.weight.copy_(torch.tensor([0.1, 1, 10, 100]))
    x = torch.randn(2, 7, 4).transpose(0, 1)
    expected = model(x).detach()
    original = deepcopy(model)
    modifier, state = prepare(
        model,
        [IterSmoothMapping(source="norm", targets=["fc1", "fc2"])],
        symmetric=symmetric,
    )
    # Exercise keyword inputs and multiple calibration batches.
    model(x)
    model.fc1(input=model.norm(x * 2))
    finish(modifier, state)
    torch.testing.assert_close(model(x), expected, atol=1e-4, rtol=1e-5)
    assert not torch.equal(model.fc1.weight, original.fc1.weight)
    restored = NormLinears()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(x), model(x))
    assert all(not m._forward_pre_hooks for m in model.modules())


def test_absmax_formula_and_zero_channels():
    model = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 2))
    with torch.no_grad():
        model[1].weight.copy_(torch.tensor([[-4.0, 0.0, 2.0], [-1.0, 0.0, -8.0]]))
    original = model[1].weight.detach().clone()
    modifier, state = prepare(
        model,
        [IterSmoothMapping(source="0", targets=["1"], subgraph_type="linear-linear")],
        alpha=0.5,
    )
    model[1](torch.tensor([[-16.0, 0.0, 2.0], [-2.0, 0.0, 32.0]]))
    finish(modifier, state)
    expected_scales = torch.tensor([2.0, 1e-5, 2.0])
    torch.testing.assert_close(model[1].weight, original * expected_scales)
    assert all(torch.isfinite(p).all() for p in model.parameters())


@pytest.mark.parametrize("alpha", [0.0, 0.9, 1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_linear_linear_preserves_bias(alpha, dtype):
    torch.manual_seed(2)
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 3)).to(dtype)
    x = torch.randn(3, 4).to(dtype)
    expected = model(x).detach()
    modifier, state = prepare(
        model,
        [IterSmoothMapping(source="0", targets=["1"], subgraph_type="linear-linear")],
        alpha=alpha,
    )
    model(x)
    finish(modifier, state)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(model(x), expected, atol=tolerance, rtol=tolerance)


class GatedMLP(nn.Module):
    def __init__(self, fused):
        super().__init__()
        self.fused = fused
        self.up_proj = nn.Linear(4, 16 if fused else 8)
        self.gate_proj = nn.Linear(4, 8)
        self.down_proj = nn.Linear(8, 4)

    def forward(self, x):
        if self.fused:
            gate, up = self.up_proj(x).chunk(2, dim=-1)
        else:
            gate, up = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


@pytest.mark.parametrize("fused", [False, True])
def test_up_down_collects_gated_input(fused):
    torch.manual_seed(3)
    model = GatedMLP(fused)
    x = torch.randn(2, 6, 4)
    expected = model(x).detach()
    original = model.up_proj.weight.detach().clone()
    modifier, state = prepare(
        model,
        [
            IterSmoothMapping(
                source="up_proj",
                targets=["down_proj"],
                subgraph_type="up-down",
                fused=fused,
            )
        ],
    )
    captured = []
    handle = model.down_proj.register_forward_pre_hook(
        lambda m, args: captured.append(args[0])
    )
    model(x)
    handle.remove()
    low, high = modifier._stats["up_proj"]
    torch.testing.assert_close(high, captured[0].reshape(-1, 8).amax(0))
    torch.testing.assert_close(low, captured[0].reshape(-1, 8).amin(0))
    finish(modifier, state)
    torch.testing.assert_close(model(x), expected)
    if fused:
        torch.testing.assert_close(model.up_proj.weight[:8], original[:8])


class Attention(nn.Module):
    def __init__(self, heads, kv_heads, fused):
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=heads, num_key_value_heads=kv_heads
        )
        self.heads, self.kv_heads, self.fused = heads, kv_heads, fused
        self.q_proj = nn.Linear(8, heads * 2)
        self.k_proj = nn.Linear(8, kv_heads * 2)
        self.v_proj = nn.Linear(
            8, (heads + 2 * kv_heads) * 2 if fused else kv_heads * 2
        )
        self.o_proj = nn.Linear(heads * 2, 8)

    def forward(self, x):
        if self.fused:
            q, k, v = self.v_proj(x).split(
                [self.heads * 2, self.kv_heads * 2, self.kv_heads * 2], dim=-1
            )
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.reshape(x.shape[0], -1, self.heads, 2).transpose(1, 2)
        k = k.reshape(x.shape[0], -1, self.kv_heads, 2).transpose(1, 2)
        v = v.reshape(x.shape[0], -1, self.kv_heads, 2).transpose(1, 2)
        k, v = [a.repeat_interleave(self.heads // self.kv_heads, dim=1) for a in (k, v)]
        out = (
            F.scaled_dot_product_attention(q, k, v)
            .transpose(1, 2)
            .reshape(x.shape[0], -1, self.heads * 2)
        )
        return self.o_proj(out)


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("fused", [False, True])
def test_attention_mha_gqa_mqa(kv_heads, fused):
    torch.manual_seed(4)
    model = Attention(4, kv_heads, fused)
    x = torch.randn(2, 7, 8)
    expected = model(x).detach()
    original = model.v_proj.weight.detach().clone()
    modifier, state = prepare(
        model,
        [
            IterSmoothMapping(
                source="v_proj", targets=["o_proj"], subgraph_type="ov", fused=fused
            )
        ],
    )
    model(x)
    finish(modifier, state)
    torch.testing.assert_close(model(x), expected)
    if fused:
        torch.testing.assert_close(
            model.v_proj.weight[: -kv_heads * 2], original[: -kv_heads * 2]
        )


def test_unobserved_expert_and_sequential_callbacks():
    model = nn.ModuleList([NormLinears(), NormLinears()])
    modifier, state = prepare(
        model, [IterSmoothMapping(source="re:.*norm", targets=["re:.*fc1", "re:.*fc2"])]
    )
    original = deepcopy(model)
    x = torch.randn(3, 4)
    model[0](x)
    modifier.on_sequential_epoch_end(state, Event(), list(model[0].modules()))
    torch.testing.assert_close(model[0](x), original[0](x))
    first_weights = model[0].fc1.weight.detach().clone()
    # Repeated callbacks and propagation passes must not apply smoothing twice.
    modifier.on_sequential_epoch_end(state, Event(), list(model[0].modules()))
    finish(modifier, state)
    torch.testing.assert_close(model[0].fc1.weight, first_weights)
    for name, tensor in original[1].state_dict().items():
        torch.testing.assert_close(model[1].state_dict()[name], tensor)


def test_filtering_and_priority():
    model = nn.ModuleDict({"a": GatedMLP(False), "b": NormLinears()})
    modifier = IterSmoothModifier(
        mappings=[
            IterSmoothMapping(source="b.norm", targets=["b.fc1", "b.fc2"]),
            IterSmoothMapping(
                source="a.up_proj", targets=["a.down_proj"], subgraph_type="up-down"
            ),
        ],
    )
    assert [m.config.subgraph_type for m in modifier._resolve_mappings(model)] == [
        "up-down",
        "norm-linear",
    ]
    modifier.include = ["b.*"]
    assert [m.name for m in modifier._resolve_mappings(model)] == ["b.norm"]
    modifier.exclude = ["*.norm"]
    assert modifier._resolve_mappings(model) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": -0.1},
        {"alpha": 1.1},
        {"alpha": float("nan")},
        {"scale_min": 0},
        {"scale_min": float("inf")},
        {"enable_subgraph_type": ["unknown"]},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValidationError):
        IterSmoothModifier(**kwargs)


def test_reject_asymmetric_bias_free_before_mutation():
    model = NormLinears(bias=False)
    original = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="existing biases"):
        prepare(
            model,
            [IterSmoothMapping(source="norm", targets=["fc1", "fc2"])],
            symmetric=False,
        )
    for name, tensor in original.items():
        torch.testing.assert_close(model.state_dict()[name], tensor)


def test_requires_calibration():
    with pytest.raises(ValueError, match="calibration dataset"):
        IterSmoothModifier().initialize(State(model=NormLinears()))


def test_cross_subgraph_rejected():
    model = NormLinears()
    modifier, state = prepare(
        model, [IterSmoothMapping(source="norm", targets=["fc1", "fc2"])]
    )
    model(torch.randn(3, 4))
    with pytest.raises(ValueError, match="crosses sequential subgraphs"):
        modifier.on_sequential_epoch_end(state, Event(), [model.fc1])
    modifier.remove_hooks()


@pytest.mark.usefixtures("setup_modifier_factory")
def test_factory_and_recipe_roundtrip():
    from llmcompressor.modifiers import ModifierFactory
    from llmcompressor.recipe import Recipe

    modifier = ModifierFactory.create(
        type_="IterSmoothModifier",
        allow_registered=True,
        allow_experimental=False,
        alpha=0.8,
        mappings=[{"source": "norm", "targets": ["fc1", "fc2"]}],
    )
    recipe = Recipe.create_instance([modifier])
    restored = Recipe.create_instance(recipe.yaml())
    restored_modifier = restored.modifiers[0]
    assert isinstance(restored_modifier, IterSmoothModifier)
    assert restored_modifier.alpha == 0.8
    assert restored_modifier.mappings == modifier.mappings


def test_distributed_rank_without_local_samples(mocker):
    from llmcompressor.modifiers.transform.itersmooth import base

    model = NormLinears()
    modifier, state = prepare(
        model, [IterSmoothMapping(source="norm", targets=["fc1", "fc2"])]
    )
    mocker.patch.object(base, "is_distributed", return_value=True)

    def reduce(tensor, op):
        tensor.fill_(-2 if op == torch.distributed.ReduceOp.MIN else 4)

    all_reduce = mocker.patch.object(base.dist, "all_reduce", side_effect=reduce)
    original = model.fc1.weight.detach().clone()
    finish(modifier, state)
    assert all_reduce.call_count == 2
    assert not torch.equal(original, model.fc1.weight)


@pytest.mark.parametrize("pipeline", ["basic", "independent"])
def test_local_llama_oneshot_save_reload(tmp_path, pipeline):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForCausalLM,
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedTokenizerFast,
    )

    from llmcompressor import oneshot

    torch.manual_seed(5)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).eval()
    model.config._attn_implementation = "eager"
    original_path = tmp_path / "original"
    model.save_pretrained(original_path)
    model.config._name_or_path = str(original_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    data = [
        {
            "input_ids": torch.randint(2, 32, (8,)),
            "attention_mask": torch.ones(8, dtype=torch.long),
        }
        for _ in range(3)
    ]
    loader = DataLoader(data, batch_size=1)
    sample = next(iter(loader))
    expected = model(**sample).logits.detach()
    original = model.model.layers[0].self_attn.v_proj.weight.detach().clone()
    modifier = IterSmoothModifier()
    # Verify automatic discovery includes both norms and both internal pairs.
    resolved = modifier._resolve_mappings(model)
    assert len(resolved) == 8
    output_path = tmp_path / "smoothed"
    oneshot(
        model=model,
        processor=tokenizer,
        dataset=loader,
        recipe=modifier,
        pipeline=pipeline,
        output_dir=str(output_path),
    )
    assert not torch.equal(model.model.layers[0].self_attn.v_proj.weight, original)
    torch.testing.assert_close(model(**sample).logits, expected, atol=1e-5, rtol=1e-4)
    restored = AutoModelForCausalLM.from_pretrained(
        output_path, attn_implementation="eager"
    )
    torch.testing.assert_close(
        restored(**sample).logits, expected, atol=1e-5, rtol=1e-4
    )


def test_local_llama_w8a8_independent(tmp_path):
    import json

    from safetensors.torch import load_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForCausalLM,
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedTokenizerFast,
    )
    from transformers.utils.quantization_config import CompressedTensorsConfig

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    torch.manual_seed(6)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            tie_word_embeddings=False,
        )
    ).eval()
    model.config._attn_implementation = "eager"
    original_path = tmp_path / "original"
    model.save_pretrained(original_path)
    model.config._name_or_path = str(original_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    data = [
        {
            "input_ids": torch.randint(2, 32, (8,)),
            "attention_mask": torch.ones(8, dtype=torch.long),
        }
        for _ in range(3)
    ]
    output_path = tmp_path / "quantized"
    original_norm = model.model.layers[0].input_layernorm.weight.detach().clone()
    oneshot(
        model=model,
        processor=tokenizer,
        dataset=DataLoader(data, batch_size=1),
        recipe=[
            IterSmoothModifier(),
            QuantizationModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]),
        ],
        pipeline="independent",
        output_dir=str(output_path),
    )
    tensors = load_file(str(output_path / "model.safetensors"))
    assert not torch.equal(model.model.layers[0].input_layernorm.weight, original_norm)
    assert any(tensor.dtype == torch.int8 for tensor in tensors.values())
    assert any(name.endswith("input_scale") for name in tensors)
    saved_config = json.loads((output_path / "config.json").read_text())
    assert saved_config["quantization_config"]["quant_method"] == "compressed-tensors"
    sample = next(iter(DataLoader(data)))
    expected = model(**sample).logits.detach()
    assert torch.isfinite(expected).all()
    restored = AutoModelForCausalLM.from_pretrained(
        output_path,
        quantization_config=CompressedTensorsConfig(dequantize=True),
        attn_implementation="eager",
    )
    torch.testing.assert_close(
        restored(**sample).logits, expected, atol=5e-3, rtol=1e-3
    )


def test_asymmetric_statistics_independent_of_batch_partition():
    torch.manual_seed(7)
    model = NormLinears().eval()
    x = torch.randn(4, 3, 4)
    batched_model = deepcopy(model)
    mapping = [IterSmoothMapping(source="norm", targets=["fc1", "fc2"])]
    for candidate, batches in [(model, [x]), (batched_model, list(x.split(1)))]:
        modifier, state = prepare(candidate, mapping, symmetric=False)
        for batch in batches:
            candidate(batch)
        finish(modifier, state)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(batched_model.state_dict()[name], tensor)


def test_nonfinite_activations_rejected_before_mutation():
    model = NormLinears()
    original = deepcopy(model.state_dict())
    modifier, state = prepare(
        model, [IterSmoothMapping(source="norm", targets=["fc1", "fc2"])]
    )
    model.fc1(torch.full((1, 4), float("inf")))
    with pytest.raises(ValueError, match="non-finite calibration"):
        modifier.on_sequential_epoch_end(state, Event(), list(model.modules()))
    modifier.remove_hooks()
    for name, tensor in original.items():
        torch.testing.assert_close(model.state_dict()[name], tensor)


def test_fp16_inverse_overflow_rejected_before_mutation():
    model = NormLinears().half()
    original = deepcopy(model.state_dict())
    modifier, state = prepare(
        model, [IterSmoothMapping(source="norm", targets=["fc1", "fc2"])]
    )
    model.fc1(torch.zeros(1, 4, dtype=torch.float16))
    with pytest.raises(ValueError, match="inverse smoothing overflows"):
        modifier.on_sequential_epoch_end(state, Event(), list(model.modules()))
    modifier.remove_hooks()
    for name, tensor in original.items():
        torch.testing.assert_close(model.state_dict()[name], tensor)
