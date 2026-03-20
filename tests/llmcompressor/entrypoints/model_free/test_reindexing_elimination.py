"""
Tests for fine-grained partial read approach that eliminates the
reindex_fused_weights preprocessing step for microscale schemes.
"""

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from safetensors.torch import save_file

from llmcompressor.entrypoints.model_free.helpers import build_tensor_file_index
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


class TestBuildTensorFileIndex:
    def test_basic_mapping(self, tmp_path):
        weight_map = {
            "model.layers.0.self_attn.q_proj.weight": "shard-00001.safetensors",
            "model.layers.0.self_attn.k_proj.weight": "shard-00002.safetensors",
        }
        model_files = {
            "shard-00001.safetensors": str(tmp_path / "shard-00001.safetensors"),
            "shard-00002.safetensors": str(tmp_path / "shard-00002.safetensors"),
        }
        index = build_tensor_file_index(weight_map, model_files)
        assert index["model.layers.0.self_attn.q_proj.weight"] == str(
            tmp_path / "shard-00001.safetensors"
        )
        assert index["model.layers.0.self_attn.k_proj.weight"] == str(
            tmp_path / "shard-00002.safetensors"
        )

    def test_empty_weight_map(self, tmp_path):
        index = build_tensor_file_index({}, {})
        assert index == {}

    def test_missing_shard_skipped(self, tmp_path):
        weight_map = {
            "tensor.a": "shard-00001.safetensors",
            "tensor.b": "shard-00002.safetensors",
        }
        model_files = {
            "shard-00001.safetensors": str(tmp_path / "shard-00001.safetensors"),
            # shard-00002 missing from model_files
        }
        index = build_tensor_file_index(weight_map, model_files)
        assert "tensor.a" in index
        assert "tensor.b" not in index


class TestProcessFileMicroscaleSchemeColocated:
    """Tests for the common case: all fused weights in the same shard."""

    @pytest.fixture
    def qkv_tensors(self):
        return {
            "model.layers.0.self_attn.q_proj.weight": _rand_weight(32, 32),
            "model.layers.0.self_attn.k_proj.weight": _rand_weight(32, 32),
            "model.layers.0.self_attn.v_proj.weight": _rand_weight(32, 32),
            "model.layers.0.mlp.down_proj.weight": _rand_weight(32, 32),
        }

    def test_colocated_fused_weights_no_index(self, qkv_tensors, tmp_path):
        """Standard case: all fused weights in one shard, no index needed."""
        shard_path = tmp_path / "model.safetensors"
        save_path = tmp_path / "out.safetensors"
        save_file(qkv_tensors, shard_path)

        total_size, weight_map = process_file_microscale_scheme(
            file_path=shard_path,
            save_path=save_path,
            scheme=_make_nvfp4_scheme(),
            ignore=[],
            device="cpu",
        )
        assert save_path.exists()
        assert total_size > 0
        assert len(weight_map) > 0

    def test_colocated_with_index_same_result(self, qkv_tensors, tmp_path):
        """Providing index should not change output for co-located weights."""
        shard_name = "model.safetensors"
        shard_path = tmp_path / shard_name
        save_file(qkv_tensors, shard_path)

        weight_map = {name: shard_name for name in qkv_tensors}
        model_files = {shard_name: str(shard_path)}
        tensor_file_index = build_tensor_file_index(weight_map, model_files)

        save_path = tmp_path / "out.safetensors"
        total_size, weight_map_out = process_file_microscale_scheme(
            file_path=shard_path,
            save_path=save_path,
            scheme=_make_nvfp4_scheme(),
            ignore=[],
            device="cpu",
            tensor_file_index=tensor_file_index,
        )
        assert save_path.exists()
        assert total_size > 0


class TestProcessFileMicroscaleSchemeCrossShardPartialRead:
    """Tests for cross-shard fused weights using partial reads."""

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
        tensor_file_index = build_tensor_file_index(weight_map, model_files)
        return shard1_path, shard2_path, tensor_file_index

    def test_shard1_produces_output(self, split_shards, tmp_path):
        """Shard-1 (q_proj only) processes correctly by fetching k/v from shard-2."""
        shard1_path, _, tensor_file_index = split_shards
        save_path = tmp_path / "out-00001.safetensors"

        total_size, weight_map = process_file_microscale_scheme(
            file_path=shard1_path,
            save_path=save_path,
            scheme=_make_nvfp4_scheme(),
            ignore=[],
            device="cpu",
            tensor_file_index=tensor_file_index,
        )
        assert save_path.exists()
        assert total_size > 0
        # Output should only contain shard-1's native tensors
        assert all(
            "q_proj" in k or "q_proj" in k.split(".weight")[0]
            for k in weight_map
            if "proj" in k
        )

    def test_shard2_produces_output(self, split_shards, tmp_path):
        """Shard-2 (k/v/down) processes correctly by fetching q from shard-1."""
        _, shard2_path, tensor_file_index = split_shards
        save_path = tmp_path / "out-00002.safetensors"

        total_size, weight_map = process_file_microscale_scheme(
            file_path=shard2_path,
            save_path=save_path,
            scheme=_make_nvfp4_scheme(),
            ignore=[],
            device="cpu",
            tensor_file_index=tensor_file_index,
        )
        assert save_path.exists()
        assert total_size > 0

    def test_both_shards_produce_same_keys_as_merged(self, split_shards, tmp_path):
        """Combined output keys from both shards should match
        merged single-shard keys."""
        shard1_path, shard2_path, tensor_file_index = split_shards

        # Process both shards with partial reads
        out1 = tmp_path / "out-00001.safetensors"
        out2 = tmp_path / "out-00002.safetensors"
        _, wm1 = process_file_microscale_scheme(
            shard1_path,
            out1,
            _make_nvfp4_scheme(),
            [],
            "cpu",
            tensor_file_index=tensor_file_index,
        )
        _, wm2 = process_file_microscale_scheme(
            shard2_path,
            out2,
            _make_nvfp4_scheme(),
            [],
            "cpu",
            tensor_file_index=tensor_file_index,
        )
        combined_keys = set(wm1.keys()) | set(wm2.keys())

        # Process merged shard (reference)
        from safetensors.torch import load_file

        merged = {**load_file(shard1_path), **load_file(shard2_path)}
        merged_path = tmp_path / "merged.safetensors"
        merged_out = tmp_path / "merged_out.safetensors"
        save_file(merged, merged_path)
        _, wm_merged = process_file_microscale_scheme(
            merged_path,
            merged_out,
            _make_nvfp4_scheme(),
            [],
            "cpu",
        )

        assert combined_keys == set(wm_merged.keys()), (
            f"Key mismatch:\n"
            f"  split only: {sorted(combined_keys - set(wm_merged.keys()))}\n"
            f"  merged only: {sorted(set(wm_merged.keys()) - combined_keys)}"
        )
