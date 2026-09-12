import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from llmcompressor.pipelines.streaming.checkpoint import (
    CheckpointMap,
    CheckpointReferenceError,
    materialize_buffers,
    materialize_modules,
    release_modules,
)
from llmcompressor.utils import load_context


@pytest.fixture(scope="module")
def tiny_moe_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_qwen3moe")
    config = Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_experts=4,
        num_experts_per_tok=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    model.save_pretrained(path, safe_serialization=True)
    del model
    return str(path)


@pytest.fixture
def meta_model(tiny_moe_path):
    with load_context(meta=True):
        model = AutoModelForCausalLM.from_pretrained(tiny_moe_path, device_map="meta")
    yield model
    del model


@pytest.mark.unit
def test_checkpoint_map_covers_all_params(meta_model):
    ckpt_map = CheckpointMap.from_model(meta_model)
    for name, param in meta_model.named_parameters():
        entry = ckpt_map.entry(name)
        assert entry is not None, f"{name} has no checkpoint entry"
        assert tuple(entry.shape) == tuple(param.shape)


@pytest.mark.unit
def test_checkpoint_map_requires_local_checkpoint(meta_model):
    meta_model.name_or_path = "not-a-local-dir"
    with pytest.raises(CheckpointReferenceError):
        CheckpointMap.from_model(meta_model)


@pytest.mark.unit
def test_materialize_matches_from_pretrained(tiny_moe_path, meta_model):
    ckpt_map = CheckpointMap.from_model(meta_model)
    materialize_modules(
        meta_model, list(meta_model.modules()), ckpt_map, torch.device("cpu")
    )
    materialize_buffers(meta_model, torch.device("cpu"), ckpt_map)

    reference = AutoModelForCausalLM.from_pretrained(tiny_moe_path, device_map="cpu")
    # reference loads with fused (non-linearized) experts; compare via the
    # linearized view of the same weights using named param intersection
    stream_params = dict(meta_model.named_parameters())
    matched = 0
    for name, ref_param in reference.named_parameters():
        if name in stream_params:
            assert torch.equal(stream_params[name], ref_param), name
            matched += 1
    # all non-expert params are shared; expert params are linearized
    assert matched > 0
    assert all(p.device.type != "meta" for p in meta_model.parameters())


@pytest.mark.unit
def test_materialize_buffers_recomputes_rotary(tiny_moe_path, meta_model):
    assert meta_model.model.rotary_emb.inv_freq.device.type == "meta"
    ckpt_map = CheckpointMap.from_model(meta_model)
    materialize_buffers(meta_model, torch.device("cpu"), ckpt_map)
    assert meta_model.model.rotary_emb.inv_freq.device.type == "cpu"

    reference = AutoModelForCausalLM.from_pretrained(tiny_moe_path, device_map="cpu")
    assert torch.equal(
        meta_model.model.rotary_emb.inv_freq, reference.model.rotary_emb.inv_freq
    )


@pytest.mark.unit
def test_release_modules_moves_to_cpu(meta_model):
    ckpt_map = CheckpointMap.from_model(meta_model)
    materialize_buffers(meta_model, torch.device("cpu"), ckpt_map)
    materialize_modules(
        meta_model, list(meta_model.modules()), ckpt_map, torch.device("cpu")
    )
    release_modules(list(meta_model.modules()), torch.device("cpu"))
    assert all(p.device.type == "cpu" for p in meta_model.parameters())


@pytest.mark.unit
def test_fused_checkpoint_expert_slices(tmp_path):
    """Linearized experts must resolve to slices of fused checkpoint tensors."""
    from safetensors.torch import save_file

    from llmcompressor.modeling.moe.linearize import linearize_moe

    config = Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_experts=4,
        num_experts_per_tok=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    # save the raw (fused 3D expert) state dict, bypassing save conversions
    save_file(dict(model.state_dict()), str(tmp_path / "model.safetensors"))
    # reference values before freeing the model
    ref_gate_up = model.model.layers[0].mlp.experts.gate_up_proj.detach().clone()
    ref_down = model.model.layers[0].mlp.experts.down_proj.detach().clone()
    del model
    config.save_pretrained(tmp_path)

    # meta-load and linearize structurally (no weight data is copied)
    meta_model = AutoModelForCausalLM.from_pretrained(tmp_path, device_map="meta")
    linearize_moe(meta_model)
    assert all(p.device.type == "meta" for p in meta_model.parameters())

    ckpt_map = CheckpointMap.from_model(meta_model)
    entry = ckpt_map.entry("model.layers.0.mlp.experts.1.gate_proj.weight")
    assert entry is not None and entry.slices is not None

    materialize_modules(
        meta_model, list(meta_model.modules()), ckpt_map, torch.device("cpu")
    )
    inter = config.moe_intermediate_size
    expert = meta_model.model.layers[0].mlp.experts[1]
    assert torch.equal(expert.gate_proj.weight, ref_gate_up[1, :inter])
    assert torch.equal(expert.up_proj.weight, ref_gate_up[1, inter:])
    assert torch.equal(expert.down_proj.weight, ref_down[1])


@pytest.mark.regression
def test_streaming_matches_sequential(tiny_moe_path):
    """Streaming (meta) and sequential (offloaded) pipelines must produce
    bitwise-identical quantized weights on the same data."""
    from compressed_tensors.quantization.quant_scheme import (
        NVFP4,
        QuantizationScheme,
    )
    from torch.utils.data import DataLoader, TensorDataset

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    def run(meta: bool):
        with load_context(meta=meta):
            model = AutoModelForCausalLM.from_pretrained(
                tiny_moe_path, device_map="meta" if meta else "cpu"
            )
        recipe = QuantizationModifier(
            config_groups={
                "mlp": QuantizationScheme(
                    targets=[r"re:.*(self_attn|mlp)\..*"], **NVFP4
                )
            },
            ignore=[r"lm_head"],
        )
        torch.manual_seed(1)
        input_ids = torch.randint(0, 256, (8, 16))

        def collate(batch):
            return {
                "input_ids": torch.stack([b[0] for b in batch]),
                "attention_mask": torch.ones(len(batch), 16, dtype=torch.long),
            }

        kwargs = {"recipe": recipe, "num_calibration_samples": 8}
        if not meta:
            kwargs["pipeline"] = "sequential"
        # meta models default to the streaming pipeline via oneshot inference
        oneshot(
            model=model,
            dataset=DataLoader(
                TensorDataset(input_ids), batch_size=2, collate_fn=collate
            ),
            **kwargs,
        )
        return dict(model.named_parameters())

    reference = run(meta=False)
    streamed = run(meta=True)
    assert set(reference) == set(streamed)
    for name in reference:
        assert torch.equal(reference[name].cpu(), streamed[name].cpu()), name
