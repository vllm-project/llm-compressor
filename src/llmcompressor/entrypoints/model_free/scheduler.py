import os

from compressed_tensors.utils.safetensors_load import InverseWeightMap

__all__ = ["estimate_job_memory"]

# 3x covers the weight tensors themselves plus quantization intermediates
# (scales, zero-points, compressed output buffers). Conservative on purpose:
# a slight overestimate just means we serialize a bit more; an underestimate
# means OOM.
_MEMORY_MULTIPLIER = 3.0


def estimate_job_memory(inverse_weight_map: InverseWeightMap) -> int:
    """Rough memory estimate for a shard job, based on source file sizes."""
    total = sum(os.path.getsize(p) for p in inverse_weight_map)
    return int(total * _MEMORY_MULTIPLIER)
