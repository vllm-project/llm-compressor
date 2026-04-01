"""
Tests for inverse_weights_map approach that eliminates the
reindex_fused_weights preprocessing step for microscale schemes.
"""

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from safetensors.torch import save_file

from llmcompressor.entrypoints.model_free.microscale import (
    build_microscale_inverse_weights_map,
)
from llmcompressor.entrypoints.model_free.process import (
    process_file_microscale_scheme,
)


def _make_nvfp4_scheme():
    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="float",
            strategy="tensor_group",
            group_size=16,
            symmetric=True,
            dynamic=False,
            scale_dtype=torch.float8_e4m3fn,
        ),
    )


def _rand_weight(*shape):
    return torch.randn(*shape, dtype=torch.float16)


class TestBuildWeightsMap:
    def test_basic_mapping(self, tmp_path):
        weight_map = {
            "model.layers.0.self_attn.q_proj.weight": "shard-00001.safetensors",
            "model.layers.0.self_attn.k_proj.weight": "shard-00002.safetensors",
        }
        model_files = {
            "shard-00001.safetensors": str(tmp_path / "shard-00001.safetensors"),
            "shard-00002.safetensors": str(tmp_path / "shard-00002.safetensors"),
        }
        result = build_microscale_inverse_weights_map(
            "shard-00001.safetensors", weight_map, model_files
        )
        # result is {file_path: [tensor_names]}, check tensor is in the list
        assert (
            "model.layers.0.self_attn.q_proj.weight"
            in result[str(tmp_path / "shard-00001.safetensors")]
        )
        assert (
            "model.layers.0.self_attn.k_proj.weight"
            in result[str(tmp_path / "shard-00002.safetensors")]
        )

    def test_missing_shard_skipped(self, tmp_path):
        weight_map = {
            "tensor.a": "shard-00001.safetensors",
            "tensor.b": "shard-00002.safetensors",
        }
        model_files = {
            "shard-00001.safetensors": str(tmp_path / "shard-00001.safetensors"),
        }
        result = build_microscale_inverse_weights_map(
            "shard-00001.safetensors", weight_map, model_files
        )
        # check tensor.a is in the result values
        assert any("tensor.a" in tensors for tensors in result.values())
        assert "tensor.b" not in result


class TestBuildInverseWeightsMap:
    def test_colocated_no_partners_needed(self, tmp_path):
        """All fused weights in same shard — no cross-shard fetching needed."""
        shard = "shard-00001.safetensors"
        weight_map = {
            "model.layers.0.self_attn.q_proj.weight": shard,
            "model.layers.0.self_attn.k_proj.weight": shard,
            "model.layers.0.self_attn.v_proj.weight": shard,
        }
        model_files = {shard: str(tmp_path / shard)}
        result = build_microscale_inverse_weights_map(shard, weight_map, model_files)
        assert len(result) == 1
        assert str(tmp_path / shard) in result

    def test_cross_shard_partners_found(self, tmp_path):
        """q_proj on shard1, k/v on shard2 — shard1 should fetch from shard2."""
        weight_map = {
            "model.layers.0.self_attn.q_proj.weight": "shard-00001.safetensors",
            "model.layers.0.self_attn.k_proj.weight": "shard-00002.safetensors",
            "model.layers.0.self_attn.v_proj.weight": "shard-00002.safetensors",
        }
        model_files = {
            "shard-00001.safetensors": str(tmp_path / "shard-00001.safetensors"),
            "shard-00002.safetensors": str(tmp_path / "shard-00002.safetensors"),
        }
        result = build_microscale_inverse_weights_map(
            "shard-00001.safetensors", weight_map, model_files
        )
        # Should include both shards
        assert len(result) == 2
        shard2_path = str(tmp_path / "shard-00002.safetensors")
        assert shard2_path in result
        assert "model.layers.0.self_attn.k_proj.weight" in result[shard2_path]
        assert "model.layers.0.self_attn.v_proj.weight" in result[shard2_path]


class TestProcessFileMicroscaleSchemeColocated:
    """Tests for co-located fused weights — standard case, no cross-shard needed."""

    @pytest.fixture
    def qkv_tensors(self):
        return {
            "model.layers.0.self_attn.q_proj.weight": _rand_weight(32, 32),
            "model.layers.0.self_attn.k_proj.weight": _rand_weight(32, 32),
            "model.layers.0.self_attn.v_proj.weight": _rand_weight(32, 32),
            "model.layers.0.mlp.down_proj.weight": _rand_weight(32, 32),
        }

    def test_colocated_fused_weights(self, qkv_tensors, tmp_path):
        """Standard case: all fused weights in one shard."""
        shard_name = "model.safetensors"
        shard_path = tmp_path / shard_name
        save_path = tmp_path / "out.safetensors"
        save_file(qkv_tensors, shard_path)

        # Build inverse_weights_map: just the one file with all tensors
        inverse_weights_map = {str(shard_path): list(qkv_tensors.keys())}

        total_size, weight_map = process_file_microscale_scheme(
            inverse_weights_map=inverse_weights_map,
            save_path=save_path,
            scheme=_make_nvfp4_scheme(),
            ignore=[],
            device="cpu",
        )
        assert save_path.exists()
        assert total_size > 0
        assert len(weight_map) > 0


class TestProcessFileMicroscaleSchemeCrossShardInverseMap:
    """Tests for cross-shard fused weights using precomputed inverse_weights_map."""

    @pytest.fixture
    def split_shards(self, tmp_path):
        """q_proj on shard-1, k_proj + v_proj + down_proj on shard-2."""
        shard1_tensors = {
            "model.layers.0.self_attn.q_proj.weight": _rand_weight(32, 32),
        }
        shard2_tensors = {
            "model.layers.0.self_attn.k_proj.weight": _rand_weight(32, 32),
            "model.layers.0.self_attn.v_proj.weight": _rand_weight(32, 32),
            "model.layers.0.mlp.down_proj.weight": _rand_weight(32, 32),
        }
        shard1_path = tmp_path / "shard-00001.safetensors"
        shard2_path = tmp_path / "shard-00002.safetensors"
        save_file(shard1_tensors, shard1_path)
        save_file(shard2_tensors, shard2_path)

        weight_map = {
            "model.layers.0.self_attn.q_proj.weight": "shard-00001.safetensors",
            "model.layers.0.self_attn.k_proj.weight": "shard-00002.safetensors",
            "model.layers.0.self_attn.v_proj.weight": "shard-00002.safetensors",
            "model.layers.0.mlp.down_proj.weight": "shard-00002.safetensors",
        }
        model_files = {
            "shard-00001.safetensors": str(shard1_path),
            "shard-00002.safetensors": str(shard2_path),
        }
        # Precompute inverse_weights_map for each shard
        iwm1 = build_microscale_inverse_weights_map(
            "shard-00001.safetensors", weight_map, model_files
        )
        iwm2 = build_microscale_inverse_weights_map(
            "shard-00002.safetensors", weight_map, model_files
        )
        return shard1_path, shard2_path, iwm1, iwm2

    def test_shard1_produces_output(self, split_shards, tmp_path):
        """Shard-1 (q_proj only) processes correctly using precomputed inverse map."""
        shard1_path, _, iwm1, _ = split_shards
        save_path = tmp_path / "out-00001.safetensors"

        total_size, weight_map = process_file_microscale_scheme(
            inverse_weights_map=iwm1,
            save_path=save_path,
            scheme=_make_nvfp4_scheme(),
            ignore=[],
            device="cpu",
        )
        assert save_path.exists()
        assert total_size > 0
        assert len(weight_map) > 0

    def test_shard2_produces_output(self, split_shards, tmp_path):
        """Shard-2 (k/v/down) processes correctly using precomputed inverse map."""
        _, shard2_path, _, iwm2 = split_shards
        save_path = tmp_path / "out-00002.safetensors"

        total_size, weight_map = process_file_microscale_scheme(
            inverse_weights_map=iwm2,
            save_path=save_path,
            scheme=_make_nvfp4_scheme(),
            ignore=[],
            device="cpu",
        )
        assert save_path.exists()
        assert total_size > 0

    def test_both_shards_produce_same_keys_as_merged(self, split_shards, tmp_path):
        """Combined output keys from both shards
        should match merged single-shard keys."""
        shard1_path, shard2_path, iwm1, iwm2 = split_shards

        out1 = tmp_path / "out-00001.safetensors"
        out2 = tmp_path / "out-00002.safetensors"
        _, wm1 = process_file_microscale_scheme(
            iwm1, out1, _make_nvfp4_scheme(), [], "cpu"
        )
        _, wm2 = process_file_microscale_scheme(
            iwm2, out2, _make_nvfp4_scheme(), [], "cpu"
        )
        combined_keys = set(wm1.keys()) | set(wm2.keys())

        # Process merged shard as reference
        from safetensors.torch import load_file

        merged = {**load_file(shard1_path), **load_file(shard2_path)}
        merged_path = tmp_path / "merged.safetensors"
        merged_out = tmp_path / "merged_out.safetensors"
        save_file(merged, merged_path)
        merged_iwm = {str(merged_path): list(merged.keys())}
        _, wm_merged = process_file_microscale_scheme(
            merged_iwm, merged_out, _make_nvfp4_scheme(), [], "cpu"
        )

        assert combined_keys == set(wm_merged.keys()), (
            f"Key mismatch:\n"
            f"  split only: {sorted(combined_keys - set(wm_merged.keys()))}\n"
            f"  merged only: {sorted(set(wm_merged.keys()) - combined_keys)}"
        )
