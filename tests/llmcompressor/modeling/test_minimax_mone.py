import json
import re

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
from transformers.models.minimax_m2 import modeling_minimax_m2 as minimax_modeling
from transformers.models.minimax_m2.configuration_minimax_m2 import MiniMaxM2Config

from llmcompressor.args import ModelArguments
from llmcompressor.entrypoints.utils import initialize_model_from_path
from llmcompressor.modeling.moe.linear_experts import NoviceExpertMLP
from llmcompressor.modeling.moe.linearize import linearize_moe
from llmcompressor.modeling.moe.minimax_mone import (
    MINIMAX_M2_STOCK_FP8_TRITON_PATCH,
    MiniMaxMoNELayout,
    _restore_minimax_mone_config,
    load_minimax_mone_model,
    minimax_mone_modeling_context,
    normalize_minimax_m2_rope_parameters,
    postprocess_minimax_mone_export,
    scan_minimax_mone_layout,
)

_GATE_UP_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj$")
_DOWN_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.down_proj$")


def _tiny_minimax_config() -> MiniMaxM2Config:
    config = MiniMaxM2Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        num_local_experts=4,
        num_experts_per_tok=2,
        router_jitter_noise=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config.approximate_experts = {"0": [1, 3], "1": [0, 2]}
    config.approximate_expert_init_tokens = {"0": [5, 7], "1": [11, 13]}
    return config


@pytest.mark.unit
def test_minimax_rope_parameters_infer_partial_rotary_factor():
    config = {
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "head_dim": 128,
        "rotary_dim": 64,
        "rope_theta": 5000000,
    }

    normalize_minimax_m2_rope_parameters(config)

    assert config["rope_parameters"] == {
        "rope_type": "default",
        "rope_theta": 5000000,
        "partial_rotary_factor": 0.5,
    }


@pytest.mark.unit
def test_minimax_rope_parameters_fill_existing_dict():
    config = {
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "head_dim": 128,
        "rotary_dim": 64,
        "rope_theta": 5000000,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 5000000,
        },
    }

    normalize_minimax_m2_rope_parameters(config)

    assert config["rope_parameters"]["partial_rotary_factor"] == 0.5


@pytest.mark.unit
def test_minimax_rope_parameters_preserve_explicit_partial_factor():
    config = {
        "head_dim": 128,
        "rotary_dim": 64,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
        },
    }

    normalize_minimax_m2_rope_parameters(config)

    assert config["rope_parameters"]["partial_rotary_factor"] == 0.25


@pytest.mark.unit
def test_minimax_mone_context_routes_constant_experts():
    config = _tiny_minimax_config()
    layout = MiniMaxMoNELayout.from_config(config)
    original_block_cls = minimax_modeling.MiniMaxM2SparseMoeBlock

    with minimax_mone_modeling_context(layout):
        block = minimax_modeling.MiniMaxM2SparseMoeBlock(config)

        assert block.layer_idx == 0
        assert block.experts.logical_expert_ids == (0, 2)
        assert block.experts.constant_expert_ids == (1, 3)

        with torch.no_grad():
            block.experts.gate_up_proj.zero_()
            block.experts.down_proj.zero_()
            block.experts.constant_expert_values.copy_(
                torch.stack(
                    [
                        torch.ones(config.hidden_size),
                        torch.full((config.hidden_size,), 3.0),
                    ]
                )
            )

        hidden_states = torch.zeros(2, config.hidden_size)
        top_k_index = torch.tensor([[1, 3], [3, 1]])
        top_k_weights = torch.tensor([[0.25, 0.75], [1.0, 0.0]])

        output = block.experts(hidden_states, top_k_index, top_k_weights)

    assert minimax_modeling.MiniMaxM2SparseMoeBlock is original_block_cls
    torch.testing.assert_close(output[0], torch.full((config.hidden_size,), 2.5))
    torch.testing.assert_close(output[1], torch.full((config.hidden_size,), 3.0))


@pytest.mark.unit
def test_minimax_mone_loads_source_format_checkpoint(tmp_path):
    torch.manual_seed(5)
    config = _tiny_minimax_config()
    layout = MiniMaxMoNELayout.from_config(config)
    source_tensors, constants = _build_source_minimax_state(config, layout)

    config_dict = config.to_dict()
    config_dict["model_type"] = "minimax_m2_compressed"
    config_dict["architectures"] = ["MiniMaxM2ForCausalLM"]
    config_dict["approximate_experts"] = layout.as_approximate_experts_config()
    (tmp_path / "config.json").write_text(json.dumps(config_dict, indent=2) + "\n")
    save_file(source_tensors, tmp_path / "model.safetensors", metadata={"format": "pt"})

    assert scan_minimax_mone_layout(tmp_path) == layout

    loaded = load_minimax_mone_model(
        tmp_path,
        model_cls=AutoModelForCausalLM,
    ).eval()

    for layer_idx, layer in enumerate(loaded.model.layers):
        experts = layer.mlp.experts
        assert experts.logical_expert_ids == layout.real_experts_by_layer[layer_idx]
        assert (
            experts.constant_expert_ids == layout.constant_experts_by_layer[layer_idx]
        )
        torch.testing.assert_close(
            experts.constant_expert_values,
            constants[layer_idx].to(experts.constant_expert_values.dtype),
        )

    input_ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        output = loaded(input_ids=input_ids)

    assert output.logits.shape == (1, 3, config.vocab_size)
    assert not torch.isnan(output.logits).any()


@pytest.mark.unit
def test_minimax_mone_entrypoint_loader_uses_mone_adapter(tmp_path):
    config = _tiny_minimax_config()
    layout = MiniMaxMoNELayout.from_config(config)
    source_tensors, _ = _build_source_minimax_state(config, layout)
    config_dict = config.to_dict()
    config_dict["model_type"] = "minimax_m2_compressed"
    config_dict["architectures"] = ["MiniMaxM2ForCausalLM"]
    config_dict["approximate_experts"] = layout.as_approximate_experts_config()
    (tmp_path / "config.json").write_text(json.dumps(config_dict, indent=2) + "\n")
    save_file(source_tensors, tmp_path / "model.safetensors", metadata={"format": "pt"})

    model = initialize_model_from_path(
        ModelArguments(model=str(tmp_path), precision="float32")
    )

    assert getattr(model, "_llmcompressor_minimax_mone_view_dir")
    assert set(model.model.layers[0].mlp.experts.keys()) == {"0", "2"}
    assert set(model.model.layers[1].mlp.experts.keys()) == {"1", "3"}


@pytest.mark.unit
def test_minimax_mone_linearized_export_keeps_only_real_experts(tmp_path):
    torch.manual_seed(9)
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()

    config = _tiny_minimax_config()
    layout = MiniMaxMoNELayout.from_config(config)
    source_tensors, constants = _build_source_minimax_state(config, layout)
    config_dict = config.to_dict()
    config_dict["model_type"] = "minimax_m2_compressed"
    config_dict["architectures"] = ["MiniMaxM2ForCausalLM"]
    config_dict["approximate_experts"] = layout.as_approximate_experts_config()
    config_dict["auto_map"] = {
        "AutoModelForCausalLM": "modeling_minimax_m2.MiniMaxM2ForCausalLM"
    }
    (source_dir / "config.json").write_text(json.dumps(config_dict, indent=2) + "\n")
    save_file(
        source_tensors,
        source_dir / "model.safetensors",
        metadata={"format": "pt"},
    )

    model = load_minimax_mone_model(source_dir, linearize=True).eval()
    for layer_idx, layer in enumerate(model.model.layers):
        experts = layer.mlp.experts
        assert set(experts.keys()) == {
            str(expert_id) for expert_id in layout.real_experts_by_layer[layer_idx]
        }
        for expert in experts.children():
            expert.gate_proj.register_buffer(
                "weight_scale_inv",
                torch.ones(1, 1),
            )
            expert.up_proj.register_buffer(
                "weight_scale_inv",
                torch.ones(1, 1),
            )
            expert.down_proj.register_buffer(
                "weight_scale_inv",
                torch.ones(1, 1),
            )
        torch.testing.assert_close(
            experts.constant_expert_values,
            constants[layer_idx].to(experts.constant_expert_values.dtype),
        )

    model.save_pretrained(output_dir, safe_serialization=True)
    postprocess_minimax_mone_export(model, output_dir)

    exported_config = json.loads((output_dir / "config.json").read_text())
    assert exported_config["model_type"] == "minimax_m2_compressed"
    assert (
        exported_config["approximate_experts"] == layout.as_approximate_experts_config()
    )
    assert exported_config["approximate_expert_init_tokens"] == {
        "0": [5, 7],
        "1": [11, 13],
    }
    assert "auto_map" not in exported_config
    assert scan_minimax_mone_layout(output_dir) == layout

    weight_map = json.loads((output_dir / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    for layer_idx, expert_ids in layout.constant_experts_by_layer.items():
        for expert_id in expert_ids:
            assert (
                f"model.layers.{layer_idx}.block_sparse_moe.experts."
                f"{expert_id}.approx_value"
            ) in weight_map
            assert not any(
                key.startswith(f"model.layers.{layer_idx}.mlp.experts.{expert_id}.")
                for key in weight_map
            )


@pytest.mark.unit
def test_minimax_mone_export_from_locally_pruned_stock_minimax(tmp_path):
    torch.manual_seed(13)
    config = _tiny_minimax_config()
    layout = MiniMaxMoNELayout.from_config(config)
    model = minimax_modeling.MiniMaxM2ForCausalLM(config).eval()
    implementation_metadata = {
        "algorithm": "mone",
        "patches_enabled": [MINIMAX_M2_STOCK_FP8_TRITON_PATCH],
    }
    model.config.llmcompressor_mone_implementation = implementation_metadata
    linearize_moe(model)

    for layer_idx, expert_ids in layout.constant_experts_by_layer.items():
        experts = model.model.layers[layer_idx].mlp.experts
        for expert_id in expert_ids:
            novice = NoviceExpertMLP(config.hidden_size, torch.float32)
            with torch.no_grad():
                novice.approx_value.fill_(10 * layer_idx + expert_id)
            experts[expert_id] = novice
        for expert in experts.children():
            if not hasattr(expert, "gate_proj"):
                continue
            expert.gate_proj.register_buffer(
                "weight_scale_inv",
                torch.ones(1, 1),
            )
            expert.up_proj.register_buffer(
                "weight_scale_inv",
                torch.ones(1, 1),
            )
            expert.down_proj.register_buffer(
                "weight_scale_inv",
                torch.ones(1, 1),
            )

    model.save_pretrained(tmp_path, safe_serialization=True)
    postprocess_minimax_mone_export(model, tmp_path)

    exported_config = json.loads((tmp_path / "config.json").read_text())
    assert exported_config["llmcompressor_mone_implementation"] == (
        implementation_metadata
    )
    assert scan_minimax_mone_layout(tmp_path) == layout
    weight_map = json.loads((tmp_path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    for layer_idx, real_ids in layout.real_experts_by_layer.items():
        for expert_id in real_ids:
            prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}"
            assert f"{prefix}.w1.weight" in weight_map
            assert f"{prefix}.w2.weight" in weight_map
            assert f"{prefix}.w3.weight" in weight_map
            assert f"{prefix}.w1.weight_scale_inv" in weight_map
            assert f"{prefix}.w2.weight_scale_inv" in weight_map
            assert f"{prefix}.w3.weight_scale_inv" in weight_map
            assert f"{prefix}.gate_proj.weight" not in weight_map
            assert f"{prefix}.up_proj.weight" not in weight_map
            assert f"{prefix}.down_proj.weight" not in weight_map
            assert f"{prefix}.gate_proj.weight_scale_inv" not in weight_map
            assert f"{prefix}.up_proj.weight_scale_inv" not in weight_map
            assert f"{prefix}.down_proj.weight_scale_inv" not in weight_map

    loaded = load_minimax_mone_model(tmp_path).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        output = loaded(input_ids=input_ids)
    assert output.logits.shape == (1, 3, config.vocab_size)
    assert not torch.isnan(output.logits).any()


@pytest.mark.unit
def test_minimax_mone_export_splits_native_fp8_fused_experts(tmp_path):
    config = _tiny_minimax_config()
    layout = MiniMaxMoNELayout.from_config(config)
    tensors = {}

    for layer_idx in range(config.num_hidden_layers):
        gate_up = torch.arange(
            config.num_local_experts
            * 2
            * config.intermediate_size
            * config.hidden_size,
            dtype=torch.float32,
        ).reshape(
            config.num_local_experts,
            2 * config.intermediate_size,
            config.hidden_size,
        )
        down = torch.arange(
            config.num_local_experts * config.hidden_size * config.intermediate_size,
            dtype=torch.float32,
        ).reshape(
            config.num_local_experts,
            config.hidden_size,
            config.intermediate_size,
        )
        tensors[f"model.layers.{layer_idx}.mlp.experts.gate_up_proj"] = (
            gate_up + 1000 * layer_idx
        )
        tensors[f"model.layers.{layer_idx}.mlp.experts.down_proj"] = (
            down + 2000 * layer_idx
        )
        tensors[f"model.layers.{layer_idx}.mlp.experts.gate_up_proj_scale_inv"] = (
            torch.arange(config.num_local_experts * 4, dtype=torch.float32)
            .reshape(config.num_local_experts, 4, 1)
            .add(10 * layer_idx)
        )
        tensors[f"model.layers.{layer_idx}.mlp.experts.down_proj_scale_inv"] = (
            torch.arange(config.num_local_experts * 2, dtype=torch.float32)
            .reshape(config.num_local_experts, 2, 1)
            .add(20 * layer_idx)
        )

    (tmp_path / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    save_file(tensors, tmp_path / "model.safetensors", metadata={"format": "pt"})

    model = _NativeFP8MoNEExportModel(config, layout)
    postprocess_minimax_mone_export(model, tmp_path)

    assert scan_minimax_mone_layout(tmp_path) == layout
    weight_map = json.loads((tmp_path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    assert not any(key.endswith(".mlp.experts.gate_up_proj") for key in weight_map)
    assert not any(key.endswith(".mlp.experts.down_proj") for key in weight_map)

    for layer_idx, real_ids in layout.real_experts_by_layer.items():
        source_gate_up = tensors[f"model.layers.{layer_idx}.mlp.experts.gate_up_proj"]
        source_down = tensors[f"model.layers.{layer_idx}.mlp.experts.down_proj"]
        source_gate_scale = tensors[
            f"model.layers.{layer_idx}.mlp.experts.gate_up_proj_scale_inv"
        ]
        for expert_id in real_ids:
            prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}"
            assert f"{prefix}.w1.weight" in weight_map
            assert f"{prefix}.w2.weight" in weight_map
            assert f"{prefix}.w3.weight" in weight_map
            assert f"{prefix}.w1.weight_scale_inv" in weight_map
            assert f"{prefix}.w2.weight_scale_inv" in weight_map
            assert f"{prefix}.w3.weight_scale_inv" in weight_map
            torch.testing.assert_close(
                _read_saved_tensor(tmp_path, weight_map, f"{prefix}.w1.weight"),
                source_gate_up[expert_id, : config.intermediate_size],
            )
            torch.testing.assert_close(
                _read_saved_tensor(tmp_path, weight_map, f"{prefix}.w3.weight"),
                source_gate_up[expert_id, config.intermediate_size :],
            )
            torch.testing.assert_close(
                _read_saved_tensor(tmp_path, weight_map, f"{prefix}.w2.weight"),
                source_down[expert_id],
            )
            torch.testing.assert_close(
                _read_saved_tensor(
                    tmp_path,
                    weight_map,
                    f"{prefix}.w1.weight_scale_inv",
                ),
                source_gate_scale[expert_id, :2],
            )
            torch.testing.assert_close(
                _read_saved_tensor(
                    tmp_path,
                    weight_map,
                    f"{prefix}.w3.weight_scale_inv",
                ),
                source_gate_scale[expert_id, 2:],
            )

    for layer_idx, expert_ids in layout.constant_experts_by_layer.items():
        for expert_id in expert_ids:
            prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}"
            assert f"{prefix}.approx_value" in weight_map
            assert f"{prefix}.w1.weight" not in weight_map
            assert f"{prefix}.w2.weight" not in weight_map
            assert f"{prefix}.w3.weight" not in weight_map


@pytest.mark.unit
def test_minimax_mone_export_removes_source_converted_novice_weights(tmp_path):
    config = _tiny_minimax_config()
    layout = MiniMaxMoNELayout.from_config(config)
    tensors = {}

    for layer_idx in range(config.num_hidden_layers):
        for expert_id in range(config.num_local_experts):
            prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}"
            tensors[f"{prefix}.w1.weight"] = torch.full(
                (config.intermediate_size, config.hidden_size),
                float(10 * layer_idx + expert_id),
            )
            tensors[f"{prefix}.w2.weight"] = torch.full(
                (config.hidden_size, config.intermediate_size),
                float(20 * layer_idx + expert_id),
            )
            tensors[f"{prefix}.w3.weight"] = torch.full(
                (config.intermediate_size, config.hidden_size),
                float(30 * layer_idx + expert_id),
            )
            tensors[f"{prefix}.w1.weight_scale_inv"] = torch.ones(1, 1)
            tensors[f"{prefix}.w2.weight_scale_inv"] = torch.ones(1, 1)
            tensors[f"{prefix}.w3.weight_scale_inv"] = torch.ones(1, 1)

    (tmp_path / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    save_file(tensors, tmp_path / "model.safetensors", metadata={"format": "pt"})

    model = _NativeFP8MoNEExportModel(config, layout)
    postprocess_minimax_mone_export(model, tmp_path)

    assert scan_minimax_mone_layout(tmp_path) == layout
    weight_map = json.loads((tmp_path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    for layer_idx, real_ids in layout.real_experts_by_layer.items():
        for expert_id in real_ids:
            prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}"
            assert f"{prefix}.w1.weight" in weight_map
            assert f"{prefix}.w2.weight" in weight_map
            assert f"{prefix}.w3.weight" in weight_map

    for layer_idx, expert_ids in layout.constant_experts_by_layer.items():
        for expert_id in expert_ids:
            prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}"
            assert f"{prefix}.approx_value" in weight_map
            assert f"{prefix}.w1.weight" not in weight_map
            assert f"{prefix}.w2.weight" not in weight_map
            assert f"{prefix}.w3.weight" not in weight_map
            assert f"{prefix}.w1.weight_scale_inv" not in weight_map
            assert f"{prefix}.w2.weight_scale_inv" not in weight_map
            assert f"{prefix}.w3.weight_scale_inv" not in weight_map


@pytest.mark.unit
def test_minimax_mone_config_preserves_fp8_metadata(tmp_path):
    config = _tiny_minimax_config()
    layout = MiniMaxMoNELayout.from_config(config)
    config.quantization_config = {
        "quant_method": "fp8",
        "fmt": "float8_e4m3fn",
        "weight_block_size": [128, 128],
        "dequantize": True,
    }

    output_config = config.to_dict()
    output_config.pop("quantization_config", None)
    (tmp_path / "config.json").write_text(json.dumps(output_config, indent=2) + "\n")
    save_file(
        {"model.embed_tokens.weight": torch.zeros(1, 1, dtype=torch.float8_e4m3fn)},
        tmp_path / "model.safetensors",
        metadata={"format": "pt"},
    )

    class DummyModel:
        pass

    model = DummyModel()
    model.config = config
    _restore_minimax_mone_config(tmp_path, model, layout)

    restored = json.loads((tmp_path / "config.json").read_text())
    assert restored["quantization_config"]["quant_method"] == "fp8"
    assert "dequantize" not in restored["quantization_config"]
    assert "fmt" not in restored["quantization_config"]


class _NativeFP8MoNEExportModel:
    def __init__(self, config: MiniMaxM2Config, layout: MiniMaxMoNELayout):
        self.config = config
        self.model = _NativeFP8MoNEExportInner(config, layout)
        self.name_or_path = ""


class _NativeFP8MoNEExportInner:
    def __init__(self, config: MiniMaxM2Config, layout: MiniMaxMoNELayout):
        self.layers = [
            _NativeFP8MoNEExportLayer(config, layout, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]


class _NativeFP8MoNEExportLayer:
    def __init__(
        self,
        config: MiniMaxM2Config,
        layout: MiniMaxMoNELayout,
        layer_idx: int,
    ):
        self.mlp = _NativeFP8MoNEExportMLP(config, layout, layer_idx)


class _NativeFP8MoNEExportMLP:
    def __init__(
        self,
        config: MiniMaxM2Config,
        layout: MiniMaxMoNELayout,
        layer_idx: int,
    ):
        self.experts = _NativeFP8MoNEExportExperts(config, layout, layer_idx)


class _NativeFP8MoNEExportExperts:
    def __init__(
        self,
        config: MiniMaxM2Config,
        layout: MiniMaxMoNELayout,
        layer_idx: int,
    ):
        self.layer_idx = layer_idx
        self.constant_expert_to_row = {
            expert_id: row
            for row, expert_id in enumerate(layout.constant_experts_by_layer[layer_idx])
        }
        self.constant_expert_values = torch.stack(
            [
                torch.full((config.hidden_size,), float(100 * layer_idx + expert_id))
                for expert_id in layout.constant_experts_by_layer[layer_idx]
            ]
        )


def _read_saved_tensor(
    output_dir,
    weight_map: dict[str, str],
    key: str,
) -> torch.Tensor:
    with safe_open(
        output_dir / weight_map[key],
        framework="pt",
        device="cpu",
    ) as handle:
        return handle.get_tensor(key)


def _build_source_minimax_state(
    config: MiniMaxM2Config,
    layout: MiniMaxMoNELayout,
) -> tuple[dict[str, torch.Tensor], dict[int, torch.Tensor]]:
    with minimax_mone_modeling_context(layout):
        model = minimax_modeling.MiniMaxM2ForCausalLM(config)

    source_tensors = {}
    for key, tensor in model.state_dict().items():
        if match := _GATE_UP_RE.match(key):
            layer_idx = int(match.group(1))
            _add_gate_up_tensors(source_tensors, layout, layer_idx, tensor)
        elif match := _DOWN_RE.match(key):
            layer_idx = int(match.group(1))
            _add_down_tensors(source_tensors, layout, layer_idx, tensor)
        else:
            source_tensors[_source_key(key)] = tensor.detach().clone()

    constants = {}
    for layer_idx, expert_ids in layout.constant_experts_by_layer.items():
        values = []
        for expert_id in expert_ids:
            value = torch.full(
                (config.hidden_size,),
                float(10 * layer_idx + expert_id),
            )
            key = (
                f"model.layers.{layer_idx}.block_sparse_moe.experts."
                f"{expert_id}.approx_value"
            )
            source_tensors[key] = value
            values.append(value)
        constants[layer_idx] = torch.stack(values)

    return source_tensors, constants


def _source_key(key: str) -> str:
    if ".mlp.gate." in key or key.endswith(".mlp.e_score_correction_bias"):
        return key.replace(".mlp.", ".block_sparse_moe.")
    return key


def _add_gate_up_tensors(
    source_tensors: dict[str, torch.Tensor],
    layout: MiniMaxMoNELayout,
    layer_idx: int,
    tensor: torch.Tensor,
):
    intermediate = tensor.shape[1] // 2
    for physical_idx, logical_id in enumerate(layout.real_experts_by_layer[layer_idx]):
        prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{logical_id}"
        source_tensors[f"{prefix}.w1.weight"] = (
            tensor[
                physical_idx,
                :intermediate,
            ]
            .detach()
            .clone()
        )
        source_tensors[f"{prefix}.w3.weight"] = (
            tensor[
                physical_idx,
                intermediate:,
            ]
            .detach()
            .clone()
        )


def _add_down_tensors(
    source_tensors: dict[str, torch.Tensor],
    layout: MiniMaxMoNELayout,
    layer_idx: int,
    tensor: torch.Tensor,
):
    for physical_idx, logical_id in enumerate(layout.real_experts_by_layer[layer_idx]):
        source_tensors[
            f"model.layers.{layer_idx}.block_sparse_moe.experts.{logical_id}.w2.weight"
        ] = tensor[physical_idx].detach().clone()
