from types import SimpleNamespace

import pytest
import torch
import torch.fx
from datasets import Dataset
from tokenizers import Tokenizer, models
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast

from llmcompressor import oneshot
from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.datasets.utils import DataCollatorWithFlattening
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.pipelines.sequential import helpers, pipeline
from llmcompressor.pipelines.sequential.helpers import resize_batches
from llmcompressor.pipelines.sequential.pipeline import SequentialPipeline


def _packed_dataloader():
    dataset = Dataset.from_dict({"input_ids": [[1] * n for n in (5, 3, 4)]})
    return DataLoader(dataset, batch_size=1, collate_fn=DataCollatorWithFlattening())


def _traced(attn_implementation):
    module = torch.nn.Linear(2, 2)
    module.config = SimpleNamespace(
        _attn_implementation=attn_implementation, model_type="llama"
    )
    return module, torch.fx.symbolic_trace(module)


@pytest.mark.unit
@pytest.mark.parametrize("attn_implementation", ["sdpa", "eager"])
def test_auto_batches_are_not_packed_without_flex_or_flash(
    monkeypatch, attn_implementation
):
    def plan(*args, **kwargs):
        raise AssertionError("batches should not be planned")

    monkeypatch.setattr(helpers, "plan_batch_tokens", plan)
    module, graph = _traced(attn_implementation)
    dataloader = _packed_dataloader()

    args = DatasetArguments(batch_size="auto", data_collator="flatten")
    resized = resize_batches(module, graph, {module}, dataloader, [], args)
    assert resized is dataloader


@pytest.mark.unit
def test_auto_batches_are_packed_with_flex_attention(monkeypatch):
    monkeypatch.setattr(helpers, "plan_batch_tokens", lambda *args, **kwargs: 8)
    module, graph = _traced("flex_attention")

    args = DatasetArguments(batch_size="auto", data_collator="flatten")
    resized = resize_batches(module, graph, {module}, _packed_dataloader(), [], args)
    assert [tuple(batch["input_ids"].shape) for batch in resized] == [(1, 8), (1, 4)]


class AutoRoundModifier:
    """Stands in for llmcompressor's AutoRoundModifier, which is matched by name"""


@pytest.mark.unit
def test_flattening_is_refused_with_autoround(monkeypatch):
    recipe = SimpleNamespace(modifiers=[AutoRoundModifier()])
    session = SimpleNamespace(lifecycle=SimpleNamespace(recipe=recipe))
    monkeypatch.setattr(pipeline, "active_session", lambda: session)

    module, _ = _traced("flex_attention")
    with pytest.raises(ValueError, match="AutoRoundModifier"):
        SequentialPipeline()(module, _packed_dataloader(), DatasetArguments())


def _calibrate(tmp_path, **oneshot_kwargs):
    """Calibrate static int8 activations with the moving-average observer"""
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    AutoModelForCausalLM.from_config(config).save_pretrained(tmp_path)
    model = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32)

    generator = torch.Generator().manual_seed(0)
    lengths = (9, 4, 13, 7, 11, 5)
    dataset = Dataset.from_dict(
        {
            "input_ids": [
                torch.randint(0, 128, (n,), generator=generator).tolist()
                for n in lengths
            ]
        }
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>")),
        unk_token="<unk>",
    )
    int8 = dict(
        num_bits=8, type="int", strategy="tensor", dynamic=False, symmetric=True
    )
    recipe = QuantizationModifier(
        ignore=["lm_head"],
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "weights": {**int8, "strategy": "channel"},
                "input_activations": {**int8, "observer": "minmax"},
            }
        },
    )
    oneshot(
        model=model,
        processor=tokenizer,
        dataset=dataset,
        recipe=recipe,
        num_calibration_samples=len(lengths),
        shuffle_calibration_samples=False,
        **oneshot_kwargs,
    )
    return {
        name: value.clone()
        for name, value in model.state_dict().items()
        if name.endswith("input_scale")
    }


@pytest.mark.unit
def test_flattening_calibrates_like_batch_size_1(tmp_path):
    # samples of different lengths are packed into two rows, in the same order as
    # batch size 1 sees them, and the observers average them one at a time
    single = _calibrate(tmp_path / "single", batch_size=1)
    packed = _calibrate(tmp_path / "packed", batch_size=3, data_collator="flatten")

    assert single.keys() == packed.keys() and len(single) > 0
    for name in single:
        assert torch.allclose(packed[name], single[name], rtol=1e-4), name
