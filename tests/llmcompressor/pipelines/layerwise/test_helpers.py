"""
Unit tests for the layerwise pipeline helper functions.

Tests cover:
- build_weight_map: loading weight maps from local directories
- build_key_remapping: prefix detection and passthrough filtering
- get_subgraph_weight_names: subgraph partitioning logic
- _detect_tied_weights: tied weight detection
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from llmcompressor.pipelines.layerwise.helpers import (
    build_key_remapping,
    build_weight_map,
    get_subgraph_weight_names,
    move_subgraph_buffers,
    offload_subgraph_weights,
)
from llmcompressor.utils.dev import skip_weights_download, skip_weights_initialize

MODEL_ID = "Qwen/Qwen3-0.6B"


@pytest.fixture
def qwen3_model():
    """Load Qwen3-0.6B on meta device without downloading weights."""
    with skip_weights_download(AutoModelForCausalLM):
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
    return model


@pytest.fixture
def fake_safetensors_dir(qwen3_model):
    """Create a temporary directory with a fake safetensors index and shard files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Gather all parameter names from the model
        param_names = [name for name, _ in qwen3_model.named_parameters()]

        # Split into 2 shards for testing
        mid = len(param_names) // 2
        shard1_params = param_names[:mid]
        shard2_params = param_names[mid:]

        # Create fake tensors (small ones since we just need the structure)
        shard1_tensors = {}
        for name in shard1_params:
            shard1_tensors[name] = torch.zeros(1)
        shard2_tensors = {}
        for name in shard2_params:
            shard2_tensors[name] = torch.zeros(1)

        # Save shard files
        save_file(shard1_tensors, str(tmpdir / "model-00001-of-00002.safetensors"))
        save_file(shard2_tensors, str(tmpdir / "model-00002-of-00002.safetensors"))

        # Write index
        weight_map = {}
        for name in shard1_params:
            weight_map[name] = "model-00001-of-00002.safetensors"
        for name in shard2_params:
            weight_map[name] = "model-00002-of-00002.safetensors"

        index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
        with open(tmpdir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)

        yield tmpdir


class TestBuildWeightMap:
    """Tests for build_weight_map."""

    def test_build_from_local_dir(self, fake_safetensors_dir, qwen3_model):
        """build_weight_map should load all parameter names from a local dir."""
        weight_map = build_weight_map(str(fake_safetensors_dir))

        # Should have entries for all model parameters
        param_names = set(name for name, _ in qwen3_model.named_parameters())
        assert set(weight_map.keys()) == param_names

        # All values should be absolute paths
        for path in weight_map.values():
            assert Path(path).is_absolute()

    def test_build_from_nonexistent_dir_raises(self):
        """Should raise an error for non-existent local paths."""
        with pytest.raises((FileNotFoundError, OSError, ValueError)):
            build_weight_map("/nonexistent/model")

    def test_single_shard_model(self, qwen3_model):
        """build_weight_map should work with a single-file model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            tensors = {
                name: torch.zeros(1) for name, _ in qwen3_model.named_parameters()
            }
            save_file(tensors, str(tmpdir / "model.safetensors"))

            weight_map = build_weight_map(str(tmpdir))
            param_names = set(name for name, _ in qwen3_model.named_parameters())
            assert set(weight_map.keys()) == param_names


class TestBuildKeyRemapping:
    """Tests for build_key_remapping."""

    def test_no_remapping_needed(self, qwen3_model, fake_safetensors_dir):
        """When model params match safetensors keys, no remapping should occur."""
        weight_map = build_weight_map(str(fake_safetensors_dir))
        _remapped_wm, model_to_sf, passthrough, _tied = build_key_remapping(
            weight_map, qwen3_model
        )

        # No remapping needed — model_to_safetensors should be empty
        assert model_to_sf == {}
        # No passthrough for a standard causal LM
        assert passthrough == []

    def test_prefix_remapping(self, qwen3_model):
        """Simulate a VL model where safetensors uses a different prefix."""
        # Create a fake weight map with "model.language_model." prefix
        fake_map = {}
        for name, _ in qwen3_model.named_parameters():
            if name.startswith("model."):
                remapped = "model.language_model." + name[len("model.") :]
                fake_map[remapped] = "/fake/shard.safetensors"
            else:
                fake_map[name] = "/fake/shard.safetensors"

        remapped_wm, _model_to_sf, _passthrough, _tied = build_key_remapping(
            fake_map, qwen3_model
        )

        # Should have remapped keys back to model param names
        model_params = set(name for name, _ in qwen3_model.named_parameters())
        # All model params should be in the remapped weight map
        for param in model_params:
            assert param in remapped_wm, f"{param} not in remapped weight map"

    def test_passthrough_keys_detected(self, qwen3_model, fake_safetensors_dir):
        """Extra keys in safetensors (e.g., visual encoder) become passthrough."""
        weight_map = build_weight_map(str(fake_safetensors_dir))
        # Add some fake visual encoder keys
        weight_map["visual.encoder.layer.0.weight"] = "/fake/path.safetensors"
        weight_map["visual.encoder.layer.1.weight"] = "/fake/path.safetensors"

        _, _, passthrough, _ = build_key_remapping(weight_map, qwen3_model)
        assert "visual.encoder.layer.0.weight" in passthrough
        assert "visual.encoder.layer.1.weight" in passthrough

    def test_tied_weights_detected(self, qwen3_model, fake_safetensors_dir):
        """Tied weights (lm_head -> embed_tokens) should be detected."""
        weight_map = build_weight_map(str(fake_safetensors_dir))

        # Remove lm_head.weight from weight_map to simulate tied weights
        # (like in real models where lm_head is tied to embed_tokens)
        if "lm_head.weight" in weight_map:
            del weight_map["lm_head.weight"]

        # Force tie_word_embeddings in config
        qwen3_model.config.tie_word_embeddings = True

        _, _, _, tied = build_key_remapping(weight_map, qwen3_model)
        # Should detect lm_head tied to embed_tokens
        assert "lm_head.weight" in tied


class TestGetSubgraphWeightNames:
    """Tests for get_subgraph_weight_names."""

    def test_first_subgraph_is_embeddings(self, qwen3_model, fake_safetensors_dir):
        """Subgraph 0 should contain embedding weights."""
        weight_map = build_weight_map(str(fake_safetensors_dir))
        num_layers = len(qwen3_model.model.layers)
        # +1 for head subgraph, +1 because last subgraph includes lm_head
        num_subgraphs = num_layers + 1

        names = get_subgraph_weight_names(
            qwen3_model,
            weight_map,
            sequential_targets=["Qwen3DecoderLayer"],
            subgraph_index=0,
            num_subgraphs=num_subgraphs,
        )

        # Should include embed_tokens
        assert any("embed_tokens" in n for n in names)
        # Should NOT include any decoder layer weights
        assert not any("layers." in n for n in names)

    def test_middle_subgraph_is_one_layer(self, qwen3_model, fake_safetensors_dir):
        """Middle subgraphs should contain exactly one decoder layer's weights."""
        weight_map = build_weight_map(str(fake_safetensors_dir))
        num_layers = len(qwen3_model.model.layers)
        num_subgraphs = num_layers + 1

        # Subgraph 1 should be layer 0
        names = get_subgraph_weight_names(
            qwen3_model,
            weight_map,
            sequential_targets=["Qwen3DecoderLayer"],
            subgraph_index=1,
            num_subgraphs=num_subgraphs,
        )

        # Should have layer 0 weights only
        assert all("model.layers.0." in n for n in names)
        assert len(names) > 0

    def test_last_subgraph_includes_lm_head(self, qwen3_model, fake_safetensors_dir):
        """Last subgraph should include final norm and lm_head."""
        weight_map = build_weight_map(str(fake_safetensors_dir))
        num_layers = len(qwen3_model.model.layers)
        num_subgraphs = num_layers + 1

        names = get_subgraph_weight_names(
            qwen3_model,
            weight_map,
            sequential_targets=["Qwen3DecoderLayer"],
            subgraph_index=num_subgraphs - 1,
            num_subgraphs=num_subgraphs,
        )

        # Should include final norm or lm_head
        has_final = any("model.norm" in n or "lm_head" in n for n in names)
        assert has_final

    def test_all_weights_covered(self, qwen3_model, fake_safetensors_dir):
        """Union of all subgraph weights should cover all weight_map keys."""
        weight_map = build_weight_map(str(fake_safetensors_dir))
        num_layers = len(qwen3_model.model.layers)
        num_subgraphs = num_layers + 1

        all_names = set()
        for i in range(num_subgraphs):
            names = get_subgraph_weight_names(
                qwen3_model,
                weight_map,
                sequential_targets=["Qwen3DecoderLayer"],
                subgraph_index=i,
                num_subgraphs=num_subgraphs,
            )
            all_names.update(names)

        # All weight_map keys should be covered
        # (lm_head may be missing if tied, which is fine)
        uncovered = set(weight_map.keys()) - all_names
        # Filter out tied weights (lm_head) which may not be assigned
        uncovered = {k for k in uncovered if "lm_head" not in k}
        assert uncovered == set(), f"Uncovered weights: {uncovered}"


class TestOffloadSubgraphWeights:
    """Tests for offload_subgraph_weights."""

    def test_offload_to_meta(self):
        """Offloading should move params to meta device."""
        with skip_weights_initialize():
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 10),
                torch.nn.Linear(10, 10),
            )
        # Put real tensors
        model[0].weight = torch.nn.Parameter(torch.randn(10, 10))
        model[0].bias = torch.nn.Parameter(torch.randn(10))

        offload_subgraph_weights(model, ["0.weight", "0.bias"], device="meta")

        assert model[0].weight.device.type == "meta"
        assert model[0].bias.device.type == "meta"
        # Second layer should be untouched
        assert model[1].weight.device.type != "meta"


class TestMoveSubgraphBuffers:
    """Tests for move_subgraph_buffers."""

    def test_moves_buffers_to_target(self):
        """Buffers should be moved to the target device."""
        module = torch.nn.Module()
        module.register_buffer("test_buf", torch.zeros(4))

        # Move to CPU (since we may not have GPU in unit tests)
        move_subgraph_buffers(None, {module}, torch.device("cpu"))
        assert module.test_buf.device == torch.device("cpu")
