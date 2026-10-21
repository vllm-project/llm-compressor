# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Observe actual CT compression calls without replacing compression math."""

import hashlib
from importlib import import_module

import torch


def state_signature(model):
    """Compare packed bytes too: torch's float8 comparison kernels vary by device."""
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(
                value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
        for name, value in model.state_dict().items()
    }


class CompressionTrace:
    def __init__(self, patch, model):
        self.real, self.meta = [], []
        self.parallel_calls, self.writes = 0, 0
        names = {module: name for name, module in model.named_modules()}
        implementation = import_module(
            "compressed_tensors.compressors.model_compressors.model_compressor"
        )
        compress = implementation.compress_module
        parallel = implementation.replace_module_parallel
        save = type(model).save_pretrained

        def record_compress(module, *args, **kwargs):
            calls = self.meta if module.weight.is_meta else self.real
            calls.append(names[module])
            result = compress(module, *args, **kwargs)
            assert module.quantization_status == "compressed"
            return result

        def record_parallel(*args, **kwargs):
            self.parallel_calls += 1
            return parallel(*args, **kwargs)

        def record_save(model, *args, **kwargs):
            self.writes += 1
            return save(model, *args, **kwargs)

        patch.setattr(implementation, "compress_module", record_compress)
        patch.setattr(implementation, "replace_module_parallel", record_parallel)
        patch.setattr(type(model), "save_pretrained", record_save)

    def report(self):
        return {
            "replace_module_parallel_calls": self.parallel_calls,
            "real_count": len(self.real),
            "real_modules": self.real,
            "meta_count": len(self.meta),
            "meta_modules": self.meta,
            "checkpoint_writes": self.writes,
        }
