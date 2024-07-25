import re
from typing import Callable, Dict, Sequence, Tuple, Union

import torch
import torch.nn.functional as TF
from torch import Tensor
from torch.nn import Module

from llmcompressor.core import State

__all__ = [
    "TensorOrCollectionType",
    "ProjectionFuncType",
    "CreateProjectionFuncType",
    "TransformFuncType",
    "CreateTransformFuncType",
    "ComparisonFuncType",
    "CreateComparisonFuncType",
    "KDFactory",
    "recursive_apply",
    "recursive_combine",
    "identity_transform",
    "softmax_transform",
    "log_softmax_transform",
    "normalize_transform",
    "l1_comparison",
    "l2_comparison",
    "inner_product_comparison",
    "cosine_similarity_comparison",
    "kl_divergence_comparison",
    "cross_entropy_comparison",
]


TensorOrCollectionType = Union[Tensor, Sequence[Tensor], Dict[str, Tensor]]
ProjectionFuncType = Callable[
    [TensorOrCollectionType, TensorOrCollectionType], TensorOrCollectionType
]
CreateProjectionFuncType = Callable[
    [str, Module, Module, State], Tuple[ProjectionFuncType, ProjectionFuncType]
]
TransformFuncType = Callable[[TensorOrCollectionType], TensorOrCollectionType]
CreateTransformFuncType = Callable[[str, Module, Module, State], TransformFuncType]
ComparisonFuncType = Callable[
    [TensorOrCollectionType, TensorOrCollectionType], TensorOrCollectionType
]
CreateComparisonFuncType = Callable[[str, Module, Module, State], ComparisonFuncType]


class KDFactory:
    registry_projections: Dict[str, CreateProjectionFuncType] = {}
    registry_transforms: Dict[str, CreateTransformFuncType] = {}
    registry_comparisons: Dict[str, CreateComparisonFuncType] = {}

    @staticmethod
    def register_projection(name: str, func: CreateProjectionFuncType):
        KDFactory.registry_projections[name] = func

    @staticmethod
    def register_projection_decorator(name: str):
        def inner(func: CreateProjectionFuncType):
            KDFactory.registry_projections[name] = func
            return func

        return inner

    @staticmethod
    def create_projection(
        name: str, student_layer: Module, teacher_layer: Module, state: State, **kwargs
    ) -> Tuple[ProjectionFuncType, ProjectionFuncType]:
        for pattern, creator in KDFactory.registry_projections:
            match = pattern == name

            if not match:
                try:
                    match = re.match(pattern, name)
                except Exception:
                    pass

            if match:
                return creator(
                    name=name,
                    student_layer=student_layer,
                    teacher_layer=teacher_layer,
                    state=state,
                    **kwargs,
                )

        raise ValueError(f"Invalid projection name: {name}")

    @staticmethod
    def register_transform(name: str, func: CreateTransformFuncType):
        KDFactory.registry_transforms[name] = func

    @staticmethod
    def register_transform_decorator(name: str):
        def inner(func: CreateTransformFuncType):
            KDFactory.registry_transforms[name] = func
            return func

        return inner

    @staticmethod
    def create_transform(
        name: str,
        layer: Module,
        state: State,
        **kwargs,
    ) -> TransformFuncType:
        for pattern, creator in KDFactory.registry_transforms.items():
            match = pattern == name

            if not match:
                try:
                    match = re.match(pattern, name)
                except Exception:
                    pass

            if match:
                return creator(
                    name=name,
                    layer=layer,
                    state=state,
                    **kwargs,
                )

        raise ValueError(f"Invalid transform name: {name}")

    @staticmethod
    def register_comparison(name: str, func):
        KDFactory.registry_comparisons[name] = func

    @staticmethod
    def register_comparison_decorator(name: str):
        def inner(func):
            KDFactory.registry_comparisons[name] = func
            return func

        return inner

    @staticmethod
    def create_comparison(
        name: str, student_layer: Module, teacher_layer: Module, state: State, **kwargs
    ) -> ComparisonFuncType:
        for pattern, creator in KDFactory.registry_comparisons.items():
            match = pattern == name

            if not match:
                try:
                    match = re.match(pattern, name)
                except Exception:
                    pass

            if match:
                return creator(
                    name=name,
                    student_layer=student_layer,
                    teacher_layer=teacher_layer,
                    state=state,
                    **kwargs,
                )

        raise ValueError(f"Invalid comparison name: {name}")


def recursive_apply(
    val: TensorOrCollectionType,
    func: Callable[[Tensor], Tensor],
) -> TensorOrCollectionType:
    if isinstance(val, Tensor):
        return func(val)

    if isinstance(val, Sequence):
        return [recursive_apply(item, func) for item in val]

    if isinstance(val, dict):
        return {key: recursive_apply(item, func) for key, item in val.items()}

    raise ValueError(f"Unsupported type for recursive_apply: {type(val)}")


def recursive_combine(
    val_one: TensorOrCollectionType,
    val_two: TensorOrCollectionType,
    func: Callable[[Tensor, Tensor], Tensor],
):
    if not isinstance(val_one, type(val_two)):
        raise ValueError(
            f"val_one type of {type(val_one)} must match "
            f"val_two type of {type(val_two)}"
        )

    if isinstance(val_one, Tensor):
        return func(val_one, val_two)

    if isinstance(val_one, Sequence):
        return [
            recursive_combine(item_one, item_two, func)
            for item_one, item_two in zip(val_one, val_two)
        ]

    if isinstance(val_one, dict):
        return {
            key: recursive_combine(val_one[key], val_two[key], func)
            for key in val_one.keys()
        }

    raise ValueError(f"Unsupported type for recursive_combine: {type(val_one)}")


@KDFactory.register_transform_decorator("identity")
def identity_transform(name: str, **kwargs):
    if name != "identity":
        raise ValueError(f"Invalid transform name: {name}")

    def _create_transform(val: TensorOrCollectionType) -> TensorOrCollectionType:
        return val

    return _create_transform


@KDFactory.register_transform_decorator("softmax")
def softmax_transform(name: str, temperature: float = 1.0, dim: int = -1, **kwargs):
    if name != "softmax":
        raise ValueError(f"Invalid transform name: {name}")

    def _softmax(val: Tensor) -> Tensor:
        val = val / temperature

        return torch.softmax(val, dim=dim)

    def _create_transform(val: TensorOrCollectionType) -> TensorOrCollectionType:
        return recursive_apply(val, _softmax)

    return _create_transform


@KDFactory.register_transform_decorator("log_softmax")
def log_softmax_transform(name: str, temperature: float = 1.0, dim: int = -1, **kwargs):
    if name != "log_softmax":
        raise ValueError(f"Invalid transform name: {name}")

    def _log_softmax(val: Tensor) -> Tensor:
        val = val / temperature

        return torch.log_softmax(val, dim=dim)

    def _create_transform(val: TensorOrCollectionType) -> TensorOrCollectionType:
        return recursive_apply(val, _log_softmax)

    return _create_transform


@KDFactory.register_transform_decorator("normalize")
def normalize_transform(
    name: str,
    p: float = 1,
    dim: int = -1,
    eps: float = 1e-12,
    mean: bool = False,
    std: bool = False,
    **kwargs,
):
    if name != "normalize":
        raise ValueError(f"Invalid transform name: {name}")

    def _normalize(val: Tensor) -> Tensor:
        out = TF.normalize(val, p=p, dim=dim, eps=eps)

        if mean:
            out = out - out.mean(dim=dim, keepdim=True)

        if std:
            out = out / out.std(dim=dim, keepdim=True)

        return out

    def _create_transform(val: TensorOrCollectionType) -> TensorOrCollectionType:
        return recursive_apply(val, _normalize)

    return _create_transform


@KDFactory.register_comparison_decorator("l1_distance")
def l1_comparison(name: str, dim: int = -1, **kwargs):
    if name != "l1_distance":
        raise ValueError(f"Invalid comparison name: {name}")

    def _l1(val_one: Tensor, val_two: Tensor) -> Tensor:
        return torch.sum(torch.abs(val_one - val_two), dim=dim)

    def _create_comparison(
        val_one: TensorOrCollectionType, val_two: TensorOrCollectionType
    ) -> TensorOrCollectionType:
        return recursive_combine(val_one, val_two, _l1)

    return _create_comparison


@KDFactory.register_comparison_decorator("l2_distance")
def l2_comparison(name: str, dim: int = -1, **kwargs):
    if name != "l2_distance":
        raise ValueError(f"Invalid comparison name: {name}")

    def _l2(val_one: Tensor, val_two: Tensor) -> Tensor:
        return torch.sum((val_one - val_two) ** 2, dim=dim)

    def _create_comparison(
        val_one: TensorOrCollectionType, val_two: TensorOrCollectionType
    ) -> TensorOrCollectionType:
        return recursive_combine(val_one, val_two, _l2)

    return _create_comparison


@KDFactory.register_comparison_decorator("inner_product")
def inner_product_comparison(name: str, dim: int = -1, **kwargs):
    if name != "inner_product":
        raise ValueError(f"Invalid comparison name: {name}")

    def _inner_product(val_one: Tensor, val_two: Tensor) -> Tensor:
        return torch.sum(val_one * val_two, dim=dim)

    def _create_comparison(
        val_one: TensorOrCollectionType, val_two: TensorOrCollectionType
    ) -> TensorOrCollectionType:
        return recursive_combine(val_one, val_two, _inner_product)

    return _create_comparison


@KDFactory.register_comparison_decorator("cosine_similarity")
def cosine_similarity_comparison(name: str, dim: int = -1, **kwargs):
    if name != "cosine_similarity":
        raise ValueError(f"Invalid comparison name: {name}")

    def _cosine_similarity(val_one: Tensor, val_two: Tensor) -> Tensor:
        return torch.sum(val_one * val_two, dim=dim) / (
            torch.norm(val_one, dim=dim) * torch.norm(val_two, dim=dim)
        )

    def _create_comparison(
        val_one: TensorOrCollectionType, val_two: TensorOrCollectionType
    ) -> TensorOrCollectionType:
        return recursive_combine(val_one, val_two, _cosine_similarity)

    return _create_comparison


@KDFactory.register_comparison_decorator("kl_divergence")
def kl_divergence_comparison(
    name: str, dim: int = -1, temperature: float = 1.0, **kwargs
):
    if name != "kl_divergence":
        raise ValueError(f"Invalid comparison name: {name}")

    def _kl_divergence(val_one: Tensor, val_two: Tensor) -> Tensor:
        val_one = val_one / temperature
        val_two = val_two / temperature

        return torch.sum(val_one * torch.log(val_one / val_two), dim=dim)

    def _create_comparison(
        val_one: TensorOrCollectionType, val_two: TensorOrCollectionType
    ) -> TensorOrCollectionType:
        return recursive_combine(val_one, val_two, _kl_divergence)

    return _create_comparison


@KDFactory.register_comparison_decorator("cross_entropy")
def cross_entropy_comparison(
    name: str, temperature: float = 1.0, reduction: str = "none", **kwargs
):
    if name != "cross_entropy":
        raise ValueError(f"Invalid projection name: {name}")

    def _cross_entropy(val_one: Tensor, val_two: Tensor) -> Tensor:
        val_one = val_one / temperature
        val_two = val_two / temperature

        return TF.cross_entropy(val_one, val_two, reduction=reduction)

    def _create_comparison(
        val_one: TensorOrCollectionType, val_two: TensorOrCollectionType
    ) -> TensorOrCollectionType:
        return recursive_combine(val_one, val_two, _cross_entropy)

    return _create_comparison


@KDFactory.register_comparison_decorator("square_head")
def square_head_comparison(name: str, **kwargs):
    if name != "square_head":
        raise ValueError(f"Invalid projection name: {name}")

    def _square_head(val_one: Tensor, val_two: Tensor) -> Tensor:
        numerator = torch.sum(torch.square(val_two - val_one))
        denominator = torch.sum(torch.square(val_two))

        return numerator / denominator

    def _create_comparison(
        val_one: TensorOrCollectionType, val_two: TensorOrCollectionType
    ) -> TensorOrCollectionType:
        return recursive_combine(val_one, val_two, _square_head)

    return _create_comparison
