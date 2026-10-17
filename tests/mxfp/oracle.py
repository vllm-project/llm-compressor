# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Execute external ModelSlim MX definitions without copying their algorithms."""

from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import torch
from pydantic import BaseModel

from tests.quarot.source_loader import source_definitions


class _Registry:
    # Registration only: dispatch below selects the exact MX4/MX8 definitions.
    @staticmethod
    def register(**kwargs):
        return lambda function: function


class _InputError(ValueError):
    def __init__(self, message, **kwargs):
        super().__init__(message)


class MxOracle:
    def __init__(self, root):
        self.hashes = {}
        namespace = dict(
            torch=torch,
            nn=torch.nn,
            F=torch.nn.functional,
            dist=torch.distributed,
            dataclass=dataclass,
            Enum=Enum,
            namedtuple=namedtuple,
            contextmanager=contextmanager,
            Any=Any,
            Optional=Optional,
            Tuple=Tuple,
            BaseModel=BaseModel,
            SchemaValidateError=_InputError,
            SpecError=_InputError,
            QFuncRegistry=_Registry,
        )

        def load(relative, names, class_name=None):
            module, digest = source_definitions(
                root / relative, names, namespace, class_name=class_name
            )
            self.hashes[relative] = digest
            namespace.update({name: getattr(module, name) for name in names})
            return module

        self.types = load(
            "msmodelslim/ir/qal/qbase.py",
            [
                "_get_format_params",
                "QDType",
                "QScope",
                "QScheme",
                "QParam",
                "QStorage",
                "_TORCH_FLOAT_TYPE",
            ],
        )
        self.blocks = load(
            "msmodelslim/ir/utils.py",
            [
                "reshape_to_blocks",
                "undo_reshape_to_blocks",
            ],
        )
        self.observer = load(
            "msmodelslim/core/observer/minmax.py",
            [
                "MinMaxBlockObserverConfig",
                "MsMinMaxBlockObserver",
            ],
        )
        self.math = load(
            "msmodelslim/ir/api/impl/mx_quantization.py",
            [
                "FP32_EXPONENT_BIAS",
                "FP32_MIN_NORMAL",
                "calculate_mx_qparam",
                "calculate_mxfp4_qparam",
                "mxfp_per_block_quantize",
                "mxfp4_quantize",
                "mxfp_per_block_dequantize",
                "_quant",
            ],
        )
        namespace.update(
            calculate_qparam=self.calculate,
            quantize=self.quantize,
            dequantize=self.math.mxfp_per_block_dequantize,
            fake_quantize=self.fake_quantize,
        )
        self.weight_driver = load(
            "msmodelslim/core/quantizer/impl/minmax.py",
            ["_quantize"],
            "MXWeightPerBlockMinmax",
        )._quantize
        self.linear_forwards = {}
        for bits in (4, 8):
            self.linear_forwards[bits] = load(
                f"msmodelslim/ir/w{bits}a{bits}_mx_dynamic.py",
                ["forward"],
                f"W{bits}A{bits}MXDynamicPerBlockFakeQuantLinear",
            ).forward

    def calculate(self, min_val, max_val, q_dtype, q_scope, symmetric, **kwargs):
        function = (
            self.math.calculate_mxfp4_qparam
            if q_dtype == self.types.QDType.MXFP4
            else self.math.calculate_mx_qparam
        )
        return function(min_val, max_val, q_dtype, q_scope, symmetric, **kwargs)

    def quantize(self, storage, params):
        function = (
            self.math.mxfp4_quantize
            if params.scheme.dtype == self.types.QDType.MXFP4
            else self.math.mxfp_per_block_quantize
        )
        return function(storage, params)

    def fake_quantize(self, storage, params):
        return self.math.mxfp_per_block_dequantize(
            self.quantize(storage, params), params
        )

    def weight(self, tensor, bits):
        dtype = self.types.QDType(f"mxfp{bits}")
        config = SimpleNamespace(
            dtype=dtype,
            scope=self.types.QScope.PER_BLOCK,
            symmetric=True,
            ext={"axes": -1},
        )
        state = SimpleNamespace(
            weight=self.types.QStorage(self.types.QDType.FLOAT, tensor),
            axes=-1,
            block_size=32,
            config=config,
        )
        self.weight_driver(state)
        return state

    def tensor(self, tensor, bits):
        state = self.weight(tensor, bits)
        dq = self.math.mxfp_per_block_dequantize(
            state.w_q_storage, state.w_q_param
        ).value
        dq = self.blocks.undo_reshape_to_blocks(
            dq, state._padded_shape, state._orig_shape, state._axes
        )
        exponent = state.w_q_param.ext["scale"].squeeze(-1)
        return {
            "exponent": exponent,
            "scale": torch.exp2(exponent.float()),
            "e8m0": (exponent.float() + 127).to(torch.uint8),
            "quantized": state.w_q_storage_orig.value,
            "dequantized": dq,
        }

    def linear(self, x, weight, bits):
        state = self.weight(weight, bits)
        scheme = state.w_q_param.scheme
        block_observer = self.observer.MsMinMaxBlockObserver(
            self.observer.MinMaxBlockObserverConfig(axes=-1)
        )
        module = SimpleNamespace(
            x_axes=-1,
            w_axes=-1,
            x_mx_finfo=scheme.dtype.mx_finfo,
            w_mx_finfo=scheme.dtype.mx_finfo,
            x_scheme=scheme,
            w_scheme=scheme,
            x_minmax_block_observer=block_observer,
            weight=state.w_q_storage_orig.value,
            weight_scale=state.w_q_param.ext["scale"],
            bias=None,
        )
        return self.linear_forwards[bits](module, x)
