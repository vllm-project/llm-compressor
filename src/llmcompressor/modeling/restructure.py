"""Composable model restructuring operations."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from compressed_tensors import match_modules_set, match_named_modules
from compressed_tensors.compressors import decompress_module
from compressed_tensors.quantization.utils import is_module_quantized
from loguru import logger
from torch.nn import Module
from torch.utils._pytree import tree_leaves
from tqdm import tqdm

from llmcompressor.modeling.fuse import center_embeddings, fuse_norm_linears
from llmcompressor.modeling.norm_mappings import infer_norm_mapping_from_model
from llmcompressor.modeling.offset_norm import (
    NormCalibrationModule,
    get_norm_calibration_targets,
)
from llmcompressor.utils import untie_word_embeddings
from llmcompressor.utils.transformers import get_embeddings

__all__ = [
    "Restructure",
    "LinearizeMoERestructure",
    "OffsetNormRestructure",
    "DecompressionRestructure",
    "UntieEmbeddingsRestructure",
    "CenterEmbeddingsRestructure",
    "FuseNormsRestructure",
    "ModelRestructures",
    "pipeline_restructures",
    "requires_rotation_restructures",
]


class Restructure(ABC):
    """Apply a model restructure and optionally restore its original convention."""

    @abstractmethod
    def apply(self, model: Module) -> None:
        """Restructure the model in place."""

    def restore(self, model: Module) -> None:
        """Restore a temporary restructure; permanent restructures stay applied."""


class LinearizeMoERestructure(Restructure):
    """Replace fused MoE experts with per-expert linear modules."""

    def apply(self, model: Module) -> None:
        from llmcompressor.modeling.moe.linearize import (
            get_non_linearized_moes,
            linearize_moe,
        )

        if get_non_linearized_moes(model):
            logger.warning(
                "Detected an MoE model which has not been linearized. First load "
                "model `with llmcompressor.modeling.moe.linearize.load_quantizable_moe`"
                " before passing to `oneshot`. Falling back to post-load linearization."
            )
            linearize_moe(model)


@dataclass
class OffsetNormRestructure(Restructure):
    """Temporarily replace offset norms with their calibration equivalents."""

    replaced: dict[str, tuple[Module, NormCalibrationModule]] = field(
        default_factory=dict, init=False
    )

    def apply(self, model: Module) -> None:
        targets = get_norm_calibration_targets(model)
        if targets:
            logger.info(f"Found {len(targets)} offset-norm modules to convert")
        for name, original, class_name in targets:
            replacement = NormCalibrationModule.load_from_registry(
                class_name, original=original, config=model.config
            )
            model.set_submodule(name, replacement)
            self.replaced[name] = (original, replacement)

    def restore(self, model: Module) -> None:
        if self.replaced:
            logger.info(
                f"Restoring {len(self.replaced)} norm modules to offset convention"
            )
        for name, (original, replacement) in self.replaced.items():
            model.set_submodule(name, replacement.restore(original))
        self.replaced.clear()


class DecompressionRestructure(Restructure):
    """Decompress quantized modules."""

    def apply(self, model: Module) -> None:
        self.apply_modules(model.modules())

    def apply_modules(self, modules: Iterable[Module]) -> None:
        compressed = [module for module in modules if is_module_quantized(module)]
        for module in tqdm(compressed, desc="Decompressing modules"):
            decompress_module(module, leave_decompressed=False)


class UntieEmbeddingsRestructure(Restructure):
    """Untie input and output embeddings before modifying them."""

    def apply(self, model: Module) -> None:
        untie_word_embeddings(model)


@dataclass
class CenterEmbeddingsRestructure(Restructure):
    """Center selected embedding weights."""

    targets: list[str] = field(default_factory=list)

    @torch.no_grad()
    def apply(self, model: Module) -> None:
        for _, embedding in match_named_modules(model, self.targets, warn_on_fail=True):
            center_embeddings(embedding)


@dataclass
class FuseNormsRestructure(Restructure):
    """Fuse norm scales into their linear consumers."""

    mappings: list[tuple[str, list[str]]] = field(default_factory=list)

    @torch.no_grad()
    def apply(self, model: Module) -> None:
        for norm_target, linear_targets in self.mappings:
            for norms, *linears in match_modules_set(
                model, (norm_target, *linear_targets)
            ):
                if len(norms) != 1:
                    raise ValueError("Expected exactly one norm per fusion group")
                fuse_norm_linears(norms[0], tree_leaves(linears))


def requires_rotation_restructures(modifiers: Iterable) -> bool:
    """Whether the recipe requires the restructures used by rotations."""
    from llmcompressor.modifiers.transform.spinquant import SpinQuantModifier

    return any(isinstance(modifier, SpinQuantModifier) for modifier in modifiers)


def _rotation_restructures(model: Module) -> list[Restructure]:
    """Build the embedding and norm operations needed before rotations."""

    embedding, _ = get_embeddings(model)
    targets = (
        [name for name, module in model.named_modules() if module is embedding]
        if embedding is not None
        else ["re:.*embed_tokens$"]
    )
    mappings = infer_norm_mapping_from_model(model)
    return [
        UntieEmbeddingsRestructure(),
        CenterEmbeddingsRestructure(targets),
        FuseNormsRestructure([(mapping.norm, mapping.linears) for mapping in mappings]),
    ]


@dataclass
class ModelRestructures:
    """Coordinate restructures and restore temporary ones after calibration.

    Decompression is applied separately to each onloaded subgraph so it does not
    materialize the whole model at once.
    """

    rotation_restructures: bool = False
    decompression: DecompressionRestructure = field(
        default_factory=DecompressionRestructure, init=False
    )
    applied: list[Restructure] = field(default_factory=list, init=False)

    def apply(self, model: Module) -> None:
        restructures: list[Restructure] = [
            LinearizeMoERestructure(),
            OffsetNormRestructure(),
        ]
        for restructure in restructures:
            self.applied.append(restructure)
            restructure.apply(model)

        if self.rotation_restructures:
            self.apply_rotation_preparation(model)

    def apply_rotation_preparation(self, model: Module) -> None:
        """Apply the embedding and norm restructures needed by rotations."""
        for restructure in _rotation_restructures(model):
            self.applied.append(restructure)
            restructure.apply(model)

    def decompress_modules(self, modules: Iterable[Module]) -> None:
        self.decompression.apply_modules(modules)

    def restore(self, model: Module) -> None:
        for restructure in reversed(self.applied):
            restructure.restore(model)
        self.applied.clear()

    @contextmanager
    def context(self, model: Module):
        try:
            self.apply(model)
            yield
        finally:
            self.restore(model)


def pipeline_restructures(model: Module, session) -> ModelRestructures:
    """Use the oneshot coordinator, or prepare rotations for a standalone pipeline."""
    if session.state.restructures is not None:
        return session.state.restructures

    session.state.rotation_restructures = requires_rotation_restructures(
        session.lifecycle.recipe.modifiers
    )
    restructures = ModelRestructures(session.state.rotation_restructures)
    if restructures.rotation_restructures:
        restructures.apply_rotation_preparation(model)
    return restructures
