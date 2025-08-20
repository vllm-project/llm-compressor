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
from typing import Iterable, List, Mapping, Optional, Tuple, Union

import torch
from compressed_tensors.utils.internal import InternalModule


_LOGGER: logging.Logger = logging.getLogger(__name__)


__all__ = [
    "match_named_modules",
    "match_named_parameters",
    "match_targets",
    "match_modules_set",
    "is_match",
]


FusedMappping = Mapping[str, Iterable[str]]


def match_named_modules(
    model: torch.nn.Module,
    targets: Optional[Iterable[str]],
    ignore: Optional[Iterable[str]] = None,
    fused: Optional[FusedMappping] = None,
    warn_on_fail: bool = False,
) -> Generator[Tuple[str, torch.nn.Module]]:
    """
    Yields names and modules which match `targets` but do not match `ignore`.
    Values are returned in order of `model.named_modules()`

    :param model: model containing submodules to match against
    :param targets: target strings, potentially containing "re:" prefixes
    :param ignore: targets to ignore, potentially containing "re:" prefixes
    :fused: optional mapping from suffixes of fused modules to the suffixes of their
        corresponding shards. See `compressed_tensors.utils.match.is_match`
    :param warn_on_fail: if True, warns if any targets do not match any modules in model
    :return: generator of module names and modules
    """
    targets = targets or []
    ignore = ignore or []

    unmatched_targets = set(targets)

    for name, module in model.named_modules():
        for target in targets:
            if is_match(name, module, target, fused=fused):
                unmatched_targets -= {target}
                if not is_match(name, module, ignore, fused=fused):
                    yield name, module
                break

    if warn_on_fail:
        for target in unmatched_targets:
            _LOGGER.warning(
                f"Could not match `{target}` in instance of {model.__class__.__name__}"
            )


def match_named_parameters(
    model: torch.nn.Module,
    targets: Optional[Iterable[str]],
    ignore: Optional[Iterable[str]] = None,
    fused: Optional[FusedMappping] = None,
    warn_on_fail: bool = False,
) -> Generator[Tuple[str, torch.nn.Module, torch.nn.Parameter]]:
    """
    Yields parameters which match `targets` but do not match `ignore`.
    Values are returned in order of `model.named_modules()`

    :param model: model containing params to match against
    :param targets: target strings, potentially containing "re:" prefixes
    :param ignore: targets to ignore, potentially containing "re:" prefixes
    :fused: optional mapping from suffixes of fused modules to the suffixes of their
        corresponding shards. See `compressed_tensors.utils.match.is_match`
    :param warn_on_fail: if True, warns if any targets do not match any params in model
    :return: generator of fully-qualified param names, parent modules, and params
    """
    targets = targets or []
    ignore = ignore or []

    unmatched_targets = set(targets)
    for module_name, module in model.named_modules():
        if isinstance(module, InternalModule):
            continue

        for param_name, param in module.named_parameters(recurse=False):
            param_fqn = f"{module_name}.{param_name}"
            for target in targets:
                if _match_name(param_fqn, target, fused):
                    unmatched_targets -= {target}

                    if not any(_match_name(param_fqn, ign, fused) for ign in ignore):
                        yield param_fqn, module, param

    if warn_on_fail:
        for target in unmatched_targets:
            _LOGGER.warning(
                f"Could not match `{target}` in instance of {model.__class__.__name__}"
            )


def match_targets(
    name: str, module: torch.nn.Module, targets: Optional[Iterable[str]]
) -> List[str]:
    """
    Returns the targets that match the given name and module.

    :param name: the name of the module
    :param module: the module to match
    :param targets: the target strings, potentially containing "re:" prefixes
    :return: the targets that match the given name and module

    Outputs are ordered by type: exact name match, regex name match, class name match
    """
    targets = targets or []

    if isinstance(module, InternalModule):
        return []

    # The order of the output `matches` list matters, the are arranged from most
    # specific to least specific, and this order will be used when merging configs.
    # The entries are sorted in the following order:
    #     1. matches on exact strings
    #     2. matches on regex patterns
    #     3. matches on module names

    targets = sorted(targets, key=lambda x: ("re:" in x, x))
    matched_targets = []
    for target in targets:
        if _match_name(name, target):
            matched_targets.append(target)

    for target in targets:
        if _match_class(module, target) and target not in matched_targets:
            matched_targets.append(target)

    return matched_targets


def match_modules_set(
    model: torch.nn.Module,
    targets: Optional[Iterable[str]],
    ignore: Optional[Iterable[str]] = None,
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
    targets = targets or []
    ignore = ignore or []

    matches = dict.fromkeys(targets, None)
    for name, module in model.named_modules():
        # match until we get a full set
        for target in targets:
            if is_match(name, module, target, ignore):
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


def is_match(
    name: str,
    module: torch.nn.Module,
    targets: Union[str, Iterable[str]],
    ignore: Union[str, Iterable[str]] = tuple(),
    fused: Optional[FusedMappping] = None,
) -> bool:
    """
    Returns true if either module name or module parent classes match against target
    and the module is not an internal module. The name and module may refer to a fused
    module defined by vLLM. In these cases, a `fused` mapping must be provided.

    For example, in `vllm/model_executor/models/llama.py`:
    ```python
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }
    ```

    :param name: name of module
    :param module: module to match
    :param target: target which matches name or module, potentially contains regex
    :fused: optional mapping from suffixes of fused modules to the suffixes of their
        corresponding shards
    """
    targets = [targets] if isinstance(targets, str) else targets
    ignore = [ignore] if isinstance(ignore, str) else ignore

    return not isinstance(module, InternalModule) and (
        any(
            _match_name(name, target, fused) or _match_class(module, target)
            for target in targets
        )
        and not any(
            _match_name(name, ign, fused) or _match_class(module, ign) for ign in ignore
        )
    )


def _match_name(name: str, target: str, fused: Optional[FusedMappping] = None) -> bool:
    """
    Returns true if target string begins with "re:" and regex matches or if target
    string exactly matches name. If the name refers to a fused module defined by vLLM,
    a `fused` mapping must be provided.

    :param name: name of module
    :param target: target name, potentially contains regex
    :fused: optional mapping from suffixes of fused modules to the suffixes of their
        corresponding shards
    """
    if fused is not None:
        for fused_suffix in fused:
            if name.endswith(fused_suffix):
                name_stripped = name.removesuffix(fused_suffix)
                return any(
                    _match_name(name_stripped + shard_suffix, target)
                    for shard_suffix in fused[fused_suffix]
                )

    if target.startswith("re:"):
        return re.match(target.removeprefix("re:"), name) is not None
    else:
        return target == name


def _match_class(module: torch.nn.Module, target: str) -> bool:
    """
    Returns true if any torch parent class names match the target string exactly.
    A special exception is made for vllm's `LinearBase` class which matches `Linear`

    :param module: module to match
    :param target: target which matches name or module
    """
    # will never match against a regex pattern since `:` is not allowed in class names
    return any(
        (
            issubclass(cls, torch.nn.Module)
            and (
                cls.__name__ == target
                or (cls.__name__ == "LinearBase" and target == "Linear")
            )
        )
        for cls in module.__class__.__mro__
    )
