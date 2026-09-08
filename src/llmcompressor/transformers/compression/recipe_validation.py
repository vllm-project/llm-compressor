"""
Report recipe entries which matched no module in the model, at export time.

``modifiers/quantization/group_size_validation.py`` validates a recipe against the
model it will be applied to, at initialize. This does the same kind of check at save,
for a different reason: ``save_pretrained_wrapper`` writes the resolved quantization
config and, through ``update_and_save_recipe``, the recipe itself. Nothing currently
checks that the recipe's entries correspond to anything in the model at the moment
they are written, and after the write the checkpoint is what a consumer reads.

Only ``ignore`` is checked here. Unmatched ``targets`` already surface through
``match_named_modules(..., warn_on_fail=True)`` in
``compressed_tensors.quantization.lifecycle.apply.apply_quantization_config``;
``ignore`` has no equivalent, so an entry matching nothing is silent.

An unmatched entry is not an error and is not treated as one. The configuration is
applied exactly as written, and a recipe may legitimately carry an entry for a module
a given model does not have. It is reported so the divergence between what the author
wrote and what was resolved is visible while the checkpoint can still be rebuilt.
"""

from __future__ import annotations

import torch
from compressed_tensors.utils import is_match
from loguru import logger

from llmcompressor.modifiers import Modifier

__all__ = [
    "get_unmatched_ignore_entries",
    "warn_on_unmatched_ignore_entries",
]


def _unmatched_entries(model: torch.nn.Module, entries: list[str]) -> list[str]:
    """
    Return the entries which match no module in the model, in their original order.

    An entry stops being tested once it has matched, so each is compared against
    modules only until its first match. Only module names and classes are examined;
    no parameter is touched, so this does not onload an offloaded model.

    :param model: model to match against
    :param entries: entry strings, potentially containing "re:" prefixes
    :return: the subset of entries matching no module, order preserved
    """
    unmatched = set(entries)
    for name, module in model.named_modules():
        if not unmatched:
            break
        unmatched -= {entry for entry in unmatched if is_match(name, module, entry)}

    return [entry for entry in entries if entry in unmatched]


def get_unmatched_ignore_entries(
    model: torch.nn.Module,
    modifiers: list[Modifier],
) -> list[tuple[str, str]]:
    """
    Find recipe ``ignore`` entries which match no module in the model.

    :param model: model about to be saved
    :param modifiers: recipe modifiers, e.g.
        ``active_session().lifecycle.recipe.modifiers``
    :return: list of (modifier class name, ignore entry) for each unmatched entry
    """
    unmatched: list[tuple[str, str]] = []

    for modifier in modifiers:
        entries = getattr(modifier, "ignore", None)
        if not entries:
            continue

        modifier_name = type(modifier).__name__
        for entry in _unmatched_entries(model, list(entries)):
            unmatched.append((modifier_name, entry))

    return unmatched


def warn_on_unmatched_ignore_entries(
    model: torch.nn.Module,
    modifiers: list[Modifier],
) -> None:
    """
    Warn once per recipe ``ignore`` entry which matches no module in the model.

    Warns only. Nothing here raises and the checkpoint is written either way.

    :param model: model about to be saved
    :param modifiers: recipe modifiers, e.g.
        ``active_session().lifecycle.recipe.modifiers``
    """
    for modifier_name, entry in get_unmatched_ignore_entries(model, modifiers):
        logger.warning(
            f"Recipe entry `ignore: {entry}` on {modifier_name} matched no module in "
            f"{model.__class__.__name__}, so it had no effect. The checkpoint about "
            "to be written reflects the configuration that was resolved, not this "
            "entry."
        )
