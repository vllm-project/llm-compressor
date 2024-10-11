# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from enum import Enum
from typing import Any, Dict, Optional, Union

import torch
from pydantic import BaseModel, Field, field_validator, model_validator


__all__ = [
    "FP8_DTYPE",
    "QuantizationType",
    "QuantizationStrategy",
    "QuantizationArgs",
    "round_to_quantized_type",
    "ActivationOrdering",
]

FP8_DTYPE = torch.float8_e4m3fn


class QuantizationType(str, Enum):
    """
    Enum storing quantization type options
    """

    INT = "int"
    FLOAT = "float"


class QuantizationStrategy(str, Enum):
    """
    Enum storing quantization strategy options
    """

    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"
    TOKEN = "token"


class ActivationOrdering(str, Enum):
    """
    Enum storing strategies for activation ordering

    Group: reorder groups and weight\n
    Weight: only reorder weight, not groups. Slightly lower latency and
    accuracy compared to group actorder\n
    """

    GROUP = "group"
    WEIGHT = "weight"


class QuantizationArgs(BaseModel, use_enum_values=True):
    """
    User facing arguments used to define a quantization config for weights or
    activations

    :param num_bits: quantization bit depth
    :param type: dtype to quantized to, either int or float
    :param symmetric: whether or not quantization scale is symmetric about zero-point
    :param strategy: string id determining the scope of scale/zero-point to apply
    :param group_size: group length to use for the group strategy
    :param block_structure: 2d block structure to use for the block strategy, must be
    of the format "2x4", "8x16", etc.
    :param dynamic: set True to perform dynamic quantization - values will not be
        calibrated during calibration phase, instead during inference new quantization
        ranges will be observed with every sample. Defaults to False for static
        quantization. Note that enabling dynamic quantization will change the default
        observer to a memoryless one
    :param actorder: whether to apply group quantization in decreasing order of
        activation. Defaults to None for arbitrary ordering
    """

    num_bits: int = 8
    type: QuantizationType = QuantizationType.INT
    symmetric: bool = True
    group_size: Optional[int] = None
    strategy: Optional[QuantizationStrategy] = None
    block_structure: Optional[str] = None
    dynamic: bool = False
    actorder: Union[ActivationOrdering, bool, None] = None
    observer: Optional[str] = Field(
        default="minmax",
        description=(
            "The class to use to compute the quantization param - "
            "scale and zero-point'"
        ),
    )
    observer_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "optional dict of kwargs to be passed directly to torch quantization "
            "Observers constructor excluding quantization range or symmetry"
        ),
    )

    def get_observer(self):
        """
        :return: torch quantization FakeQuantize built based on these QuantizationArgs
        """
        from compressed_tensors.quantization.observers.base import Observer

        # No observer required for the dynamic case
        if self.dynamic:
            self.observer = None
            return self.observer

        return Observer.load_from_registry(self.observer, quantization_args=self)

    def get_kv_cache(self):
        """Get the singleton KV Cache"""
        from compressed_tensors.quantization.cache import QuantizedKVParameterCache

        return QuantizedKVParameterCache(self)

    @field_validator("type", mode="before")
    def validate_type(cls, value) -> QuantizationType:
        if isinstance(value, str):
            return QuantizationType(value.lower())

        return value

    @field_validator("group_size", mode="before")
    def validate_group(cls, value) -> Union[int, None]:
        if value is None:
            return value

        if value < -1:
            raise ValueError(
                f"Invalid group size {value}. Use group_size > 0 for "
                "strategy='group' and group_size = -1 for 'channel'"
            )

        return value

    @field_validator("strategy", mode="before")
    def validate_strategy(cls, value) -> Union[QuantizationStrategy, None]:
        if isinstance(value, str):
            return QuantizationStrategy(value.lower())

        return value

    @field_validator("actorder", mode="before")
    def validate_actorder(cls, value) -> Optional[ActivationOrdering]:
        if isinstance(value, bool):
            return ActivationOrdering.GROUP if value else None

        if isinstance(value, str):
            return ActivationOrdering(value.lower())

        return value

    @model_validator(mode="after")
    def validate_model_after(model: "QuantizationArgs") -> Dict[str, Any]:
        # extract user-passed values from dictionary
        strategy = model.strategy
        group_size = model.group_size
        actorder = model.actorder
        dynamic = model.dynamic
        observer = model.observer

        # infer strategy
        if strategy is None:
            if group_size is None:
                strategy = QuantizationStrategy.TENSOR
            elif group_size > 0:
                strategy = QuantizationStrategy.GROUP
            elif group_size == -1:
                strategy = QuantizationStrategy.CHANNEL
            else:
                raise ValueError(
                    f"Invalid group size {group_size}. Use group_size > 0 for "
                    "strategy='group' and group_size = -1 for 'channel'"
                )

        # validate strategy and group
        if strategy == QuantizationStrategy.GROUP:
            if group_size is None or group_size <= 0:
                raise ValueError(
                    f"strategy {strategy} requires group_size to be "
                    "set to a positive value"
                )
        if (
            group_size is not None
            and group_size > 0
            and strategy != QuantizationStrategy.GROUP
        ):
            raise ValueError("group_size requires strategy to be set to 'group'")

        # validate activation ordering and strategy
        if actorder is not None and strategy != QuantizationStrategy.GROUP:
            raise ValueError(
                "Must use group quantization strategy in order to apply "
                "activation ordering"
            )

        if dynamic:
            if strategy not in (
                QuantizationStrategy.TOKEN,
                QuantizationStrategy.TENSOR,
            ):
                raise ValueError(
                    f"One of {QuantizationStrategy.TOKEN} or "
                    f"{QuantizationStrategy.TENSOR} must be used for dynamic ",
                    "quantization",
                )
            if observer is not None:
                warnings.warn(
                    "No observer is used for dynamic quantization, setting to None"
                )
                model.observer = None

        # if we have not set an observer and we
        # are running static quantization, use minmax
        if not observer and not dynamic:
            model.observer = "minmax"

        # write back modified values
        model.strategy = strategy
        return model

    def pytorch_dtype(self) -> torch.dtype:
        if self.type == QuantizationType.FLOAT:
            return FP8_DTYPE
        elif self.type == QuantizationType.INT:
            if self.num_bits <= 8:
                return torch.int8
            elif self.num_bits <= 16:
                return torch.int16
            else:
                return torch.int32
        else:
            raise ValueError(f"Invalid quantization type {self.type}")


def round_to_quantized_type(
    tensor: torch.Tensor, args: QuantizationArgs
) -> torch.Tensor:
    """
    Rounds each element of the input tensor to the nearest quantized representation,
    keeping to original dtype

    :param tensor: tensor to round
    :param args: QuantizationArgs to pull appropriate dtype from
    :return: rounded tensor
    """
    original_dtype = tensor.dtype
    if args.type == QuantizationType.FLOAT:
        rounded = tensor.to(FP8_DTYPE)
    elif args.type == QuantizationType.INT:
        rounded = torch.round(tensor)
    else:
        raise ValueError(f"Invalid quantization type {args.type}")

    return rounded.to(original_dtype)
