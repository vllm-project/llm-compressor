"""
Unit tests for the model-free PTQ scheduler.

_pick_device, _free_bytes, and exec_jobs_dynamic now live in
compressed_tensors.entrypoints.convert.memory and are tested there.
This file covers the llm-compressor-specific estimate_job_memory only.
"""

from llmcompressor.entrypoints.model_free.scheduler import estimate_job_memory


def test_estimate_sums_file_sizes(tmp_path):
    f1 = tmp_path / "a.safetensors"
    f2 = tmp_path / "b.safetensors"
    f1.write_bytes(b"\x00" * 1000)
    f2.write_bytes(b"\x00" * 2000)
    assert estimate_job_memory({str(f1): None, str(f2): None}) == int(3000 * 3.0)


def test_estimate_single_file(tmp_path):
    f = tmp_path / "shard.safetensors"
    f.write_bytes(b"\x00" * 500)
    assert estimate_job_memory({str(f): None}) == int(500 * 3.0)
