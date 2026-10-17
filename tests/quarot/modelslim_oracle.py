# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small, explicitly bounded adapter around original ModelSlim QuaRot functions."""

import logging
import math
import os
import random
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from types import MethodType, SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from typing_extensions import Self, Type

from .source_loader import source_definitions


@contextmanager
def force_cpu_reference():
    """Prevent the source-level ModelSlim oracle from opportunistically using NPU.

    The oracle validates algorithm semantics on CPU. ModelSlim's original
    QuaRotInterface checks torch.npu.is_available() and otherwise moves newly
    created rotation matrices to the current NPU. On Ascend hosts this makes a
    CPU reference test depend on runtime device availability.

    Temporarily override only that availability probe while executing original
    ModelSlim source definitions. The original function is restored afterwards.
    """
    npu = getattr(torch, "npu", None)
    is_available = getattr(npu, "is_available", None) if npu is not None else None

    if not callable(is_available):
        yield
        return

    original = npu.is_available
    npu.is_available = lambda: False
    try:
        yield
    finally:
        npu.is_available = original


@contextmanager
def preserve_rng():
    random_state, numpy_state = random.getstate(), np.random.get_state()
    keys = ("PYTHONHASHSEED", "HCCL_DETERMINISTIC", "CUBLAS_WORKSPACE_CONFIG")
    environment = {key: os.environ.get(key) for key in keys}
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = {
        key: getattr(torch.backends.cudnn, key, None)
        for key in ("deterministic", "benchmark", "enable")
    }
    with torch.random.fork_rng(devices=[]):
        try:
            yield
        finally:
            random.setstate(random_state)
            np.random.set_state(numpy_state)
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            for key, value in environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            for key, value in cudnn.items():
                if value is not None:
                    setattr(torch.backends.cudnn, key, value)
                elif hasattr(torch.backends.cudnn, key):
                    delattr(torch.backends.cudnn, key)


class ModelSlimOracle:
    def __init__(self, root):
        self.root = root / "msmodelslim"
        self.hashes = {}
        common = {
            "torch": torch,
            "nn": nn,
            "math": math,
            "random": random,
            "Enum": Enum,
            "dataclass": dataclass,
            "abstractmethod": abstractmethod,
            "List": List,
            "Dict": Dict,
            "Tuple": Tuple,
            "Any": Any,
            "Optional": Optional,
            "Callable": Callable,
        }

        def load(relative, names, extra=None, class_name=None):
            module, digest = source_definitions(
                self.root / relative,
                names,
                common | (extra or {}),
                class_name=class_name,
            )
            self.hashes[relative] = digest
            return module

        errors = load(
            "utils/exception.py", ["ModelslimError"], {"Self": Self, "Type": Type}
        )
        error = errors.ModelslimError
        # Exception class behavior is retained; only unused numeric error codes differ.
        common.update(UnsupportedError=error, SchemaValidateError=error)
        seed = load(
            "utils/seed.py",
            ["seed_all"],
            {
                "os": os,
                "np": np,
                "version": version,
                "is_gpu": True,
                "get_logger": logging.getLogger,
            },
        )

        def unsupported_asset(*args, **kwargs):
            raise RuntimeError(
                "CPU oracle currently supports power-of-two Hadamards only"
            )

        hadamard = load(
            "processor/quarot/common/hadamard.py",
            [
                "HADAMARD_TXT_DATA_FILE_NAME",
                "get_had_k",
                "is_pow2",
                "matmul_had_u",
                "random_hadamard_matrix",
            ],
            {
                "load_hadamard_matrix_from_txt": unsupported_asset,
                "get_logger": logging.getLogger,
            },
        )
        self.utils = load(
            "processor/quarot/common/quarot_utils.py",
            [
                "GLOBAL_DTYPE",
                "QuaRotMode",
                "random_hadamard_matrix_block",
                "create_rot",
                "rotate_linear",
                "rotate_weight",
                "fuse_ln_linear",
            ],
            {
                "seed_all": seed.seed_all,
                "random_hadamard_matrix": hadamard.random_hadamard_matrix,
            },
        )
        interface = load(
            "processor/quarot/offline_quarot/quarot_interface.py",
            ["RotatePair", "QuaRotInterface"],
            {
                "QuaRotMode": self.utils.QuaRotMode,
                "create_rot": self.utils.create_rot,
                "GLOBAL_DTYPE": self.utils.GLOBAL_DTYPE,
            },
        )
        experts = load(
            "model/common/utils.py",
            ["resolve_expert_ep_range", "_resolve_expert_num", "_get_expert_range"],
            {"dist": dist},
        )
        glm = load(
            "model/glm_5/quarot.py",
            ["get_ln_fuse_map", "get_rotate_map"],
            {
                "QuaRotInterface": interface.QuaRotInterface,
                "_get_expert_range": experts._get_expert_range,
            },
        )
        indexer = load(
            "model/glm_5_2/model.py",
            ["_normalize_indexer_type", "get_indexer_type", "has_indexer"],
        )
        self.adapter = load(
            "model/glm_5_2/model_adapter.py",
            ["get_ln_fuse_map", "get_rotate_map", "_layer_has_indexer"],
            {
                "get_ln_fuse_map": glm.get_ln_fuse_map,
                "get_rotate_map": glm.get_rotate_map,
                "has_indexer": indexer.has_indexer,
            },
            class_name="GLM52ModelAdapter",
        )
        # Extracted methods have the same names as the global helpers they call.
        # Bind methods now, then restore those globals to the original GLM helpers.
        self.methods = {
            name: getattr(self.adapter, name)
            for name in ("get_ln_fuse_map", "get_rotate_map", "_layer_has_indexer")
        }
        self.adapter.get_ln_fuse_map = glm.get_ln_fuse_map
        self.adapter.get_rotate_map = glm.get_rotate_map

    def plan(self, model, block_size):
        adapter = SimpleNamespace(config=model.config)
        for name, method in self.methods.items():
            setattr(adapter, name, MethodType(method, adapter))
        with preserve_rng(), force_cpu_reference():
            pre_fusions, fusions = adapter.get_ln_fuse_map()
            pre, pairs = adapter.get_rotate_map(block_size)
        assert pre_fusions == {}
        # GLM reference assumes its last layer is MTP even with num_hidden_layers
        # explicitly supplied. Remove only these exact MTP-only targets.
        last = f"model.layers.{model.config.num_hidden_layers - 1}"
        excluded = {
            f"{last}.embed_tokens",
            f"{last}.eh_proj",
            f"{last}.shared_head.head",
        }
        excluded_norms = {
            (f"{last}.enorm", f"{last}.hnorm"),
            f"{last}.shared_head.norm",
        }
        fusions = {
            name: targets
            for name, targets in fusions.items()
            if name not in excluded_norms
        }
        for pair in [*pre, *pairs]:
            for mapping in (pair.left_rot, pair.right_rot):
                for name in excluded:
                    mapping.pop(name, None)
                for name in mapping:
                    model.get_submodule(name)  # Any other missing target is a failure.
        for norm, consumers in fusions.items():
            model.get_submodule(norm)
            for name in consumers:
                model.get_submodule(name)
        names = ("rot", "rot_b_proj", "rot_uv", "rot_kv_b_proj")
        stages = dict(zip(names, pairs, strict=True))
        matrices = {
            "rot": pre[0].right_rot["model.embed_tokens"],
            "rot_b_proj": stages["rot_b_proj"].left_rot[
                "model.layers.0.self_attn.q_a_proj"
            ],
            "rot_uv": stages["rot_uv"].left_rot["model.layers.0.self_attn.kv_b_proj"][
                1
            ],
            "rot_kv_b_proj": stages["rot_kv_b_proj"].left_rot[
                "model.layers.0.self_attn.kv_a_proj_with_mqa"
            ][0],
        }
        return fusions, pre[0], stages, matrices

    def rotate(self, model, pair):
        for right, mapping in ((False, pair.left_rot), (True, pair.right_rot)):
            for name, matrix in mapping.items():
                self.utils.rotate_linear(
                    model.get_submodule(name), matrix, right_rotate=right
                )
