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

import logging
import re
from collections.abc import Generator
from typing import Iterable, Tuple

import torch


_LOGGER: logging.Logger = logging.getLogger(__name__)


__all__ = [
    "match_named_modules",
    "match_named_parameters",
    "match_modules_set",
    "is_match",
    "match_name",
    "match_class",
]


def match_named_modules(
    model: torch.nn.Module,
    targets: Iterable[str],
    ignore: Iterable[str] = tuple(),
    warn_on_fail: bool = False,
) -> Generator[Tuple[str, torch.nn.Module]]:
    """
    Yields names and modules which match `targets` but do not match `ignore`.
    Values are returned in order of `model.named_modules()`

    :param model: model containing submodules to match against
    :param targets: target strings, potentially containing "re:" prefixes
    :param ignore: targets to ignore, potentially containing "re:" prefixes
    :param warn_on_fail: if True, warns if any targets do not match any modules in model
    :return: generator of module names and modules
    """
    unmatched_targets = set(targets)
    for name, module in model.named_modules():
        for target in targets:
            if is_match(name, module, target):
                unmatched_targets -= {target}

                if not any(is_match(name, module, ign) for ign in ignore):
                    yield name, module

    if warn_on_fail:
        for target in unmatched_targets:
            _LOGGER.warning(
                f"Could not match `{target}` in instance of {model.__class__.__name__}"
            )


def match_named_parameters(
    model: torch.nn.Module,
    targets: Iterable[str],
    ignore: Iterable[str] = tuple(),
    warn_on_fail: bool = False,
) -> Generator[Tuple[str, torch.nn.Module, torch.nn.Parameter]]:
    """
    Yields parameters which match `targets` but do not match `ignore`.
    Values are returned in order of `model.named_modules()`

    :param model: model containing params to match against
    :param targets: target strings, potentially containing "re:" prefixes
    :param ignore: targets to ignore, potentially containing "re:" prefixes
    :param warn_on_fail: if True, warns if any targets do not match any params in model
    :return: generator of fully-qualified param names, parent modules, and params
    """
    unmatched_targets = set(targets)
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            param_fqn = f"{module_name}.{param_name}"
            for target in targets:
                if match_name(param_fqn, target):
                    unmatched_targets -= {target}

                    if not any(match_name(param_fqn, ign) for ign in ignore):
                        yield param_fqn, module, param

    if warn_on_fail:
        for target in unmatched_targets:
            _LOGGER.warning(
                f"Could not match `{target}` in instance of {model.__class__.__name__}"
            )


def match_modules_set(
    model: torch.nn.Module,
    targets: Iterable[str],
    ignore: Iterable[str] = tuple(),
) -> Generator[Iterable[torch.nn.Module]]:
    """
    Yields modules grouped with the same order and size as `targets`.
    Values are returned in order of `model.named_modules()`

    For example, the following targets would yield module belonging to the following layers:
    ```python3
    match_modules_set(model, ["q_proj", "k_proj", "v_proj"]) == (
        (
            `model.layers.0.self_attn.q_proj`,
            `model.layers.0.self_attn.k_proj`,
            `model.layers.0.self_attn.v_proj`,
        ),
        (
            `model.layers.1.self_attn.q_proj`,
            `model.layers.1.self_attn.k_proj`,
            `model.layers.1.self_attn.v_proj`,
        ),
        ...
        (
            `model.layers.32.self_attn.q_proj`,
            `model.layers.32.self_attn.k_proj`,
            `model.layers.32.self_attn.v_proj`,
        ),
    )
    ```

    This can be used to match layers to their corresponding downstream counterparts.
    For example, matching layer norms to their subsequent linear layers
    ```python3
    for norm, q, k, v in match_modules_set(model, (norm_tgt, q_tgt, k_tgt, v_tgt)):
        fuse_norm_linears(norm, [q, k, v])

    :param model: model containing modules to match against
    :param targets: target strings, potentially containing "re:" prefixes
    :param ignore: targets to ignore, potentially containing "re:" prefixes
    """
    matches = dict.fromkeys(targets, None)
    for name, module in model.named_modules():
        # match until we get a full set
        for target in targets:
            if is_match(name, module, target) and not any(
                is_match(name, module, ign) for ign in ignore
            ):
                if matches[target] is not None:
                    raise ValueError(f"Matched a {target} twice before completing set")
                matches[target] = module

        # once we have a full set, yield and reset
        if targets and all((matches[target] is not None for target in targets)):
            yield [matches[target] for target in targets]  # ensure correct ordering
            matches = dict.fromkeys(targets, None)

    # check that none are left over
    unmatched_keys = [match for match, value in matches.items() if value is not None]
    if len(unmatched_keys):
        raise ValueError(f"Unable to match targets into set: {unmatched_keys}")


def is_match(name: str, module: torch.nn.Module, target: str) -> bool:
    """
    Returns true if either module name or module parent classes match against target
    """
    return match_name(name, target) or match_class(module, target)


def match_name(name: str, target: str) -> bool:
    """
    Returns true if target string begins with "re:" and
    regex matches or if target string exactly matches name
    """
    if target.startswith("re:"):
        return re.match(target.removeprefix("re:"), name) is not None
    else:
        return target == name


def match_class(module: torch.nn.Module, target: str) -> bool:
    """
    Returns true if any torch parent class names match the target string exactly
    """
    # will never match against a regex pattern since `:` is not allowed in class names
    return any(
        issubclass(cls, torch.nn.Module) and cls.__name__ == target
        for cls in module.__class__.__mro__
    )
