# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Device emulation utilities for ``--emulate-xpu`` testing.

Provides :class:`DeviceRemapMode`, a :class:`~torch.overrides.TorchFunctionMode`
subclass that transparently remaps device type strings in all ``torch.*`` function
calls. This allows the full test suite to run on CUDA hardware while the code
paths use XPU (or any other) device strings.
"""

import re

import torch
from torch.overrides import TorchFunctionMode


class DeviceRemapMode(TorchFunctionMode):
    """Transparently remap device type strings in all torch operations.

    When activated, any torch function receiving a device argument with
    ``fake_type`` will have it silently replaced with ``real_type`` before
    the call reaches the C++ backend.

    Uses a strict regex pattern to avoid false-positive replacements on
    strings that happen to contain the fake device type as a substring.
    """

    def __init__(self, fake_type: str, real_type: str):
        self.fake_type = fake_type
        self.real_type = real_type
        self._device_pat = re.compile(rf"^{re.escape(fake_type)}(?::\d+)?$")

    def _remap(self, arg):
        if isinstance(arg, torch.device):
            if arg.type == self.fake_type:
                return torch.device(self.real_type, arg.index)
        elif isinstance(arg, str) and self._device_pat.match(arg):
            return arg.replace(self.fake_type, self.real_type, 1)
        return arg

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        new_args = tuple(self._remap(a) for a in args)
        new_kwargs = {k: self._remap(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)
