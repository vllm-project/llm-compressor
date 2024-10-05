from typing import List, Optional, Set, Tuple

import torch
from torch.nn import Module

from llmcompressor.modifiers.distillation.utils.pytorch.kd_factory import (
    TransformFuncType,
)

__all__ = ["KDModuleWrapper"]


class KDModuleWrapper(Module):
    KD_TRANSFORMED_BUFFER = "kd_last_transformed"

    def __init__(
        self,
        layer: Module,
        hidden_size: Tuple,
        transforms: Optional[List[TransformFuncType]],
        fsdp_active: bool,
        offload_output: bool,
    ):
        super(KDModuleWrapper, self).__init__()

        self.layer = layer
        self._save_active = False
        self._fsdp_active = fsdp_active
        self.offload_output = offload_output
        self.kd_transforms = transforms
        self.kd_enabled = False
        self.register_buffer(
            self.KD_TRANSFORMED_BUFFER, torch.zeros(hidden_size, device="cpu")
        )
        self._init_called = True  # make sure this is last property to be set

        def _clear_missing_keys(module, incompatible_keys):
            incompatible_keys.missing_keys.clear()

        self.register_load_state_dict_post_hook(_clear_missing_keys)

    def forward(self, *args, **kwargs):
        if not self.kd_enabled:
            return self.layer(*args, **kwargs)

        org_output = self.layer(*args, **kwargs)
        output = org_output if isinstance(org_output, torch.Tensor) else org_output[0]

        if self.kd_transforms is not None:
            for transform in self.kd_transforms:
                output = transform(output)

        if self.offload_output:
            output = output.to("cpu")
        setattr(self, self.KD_TRANSFORMED_BUFFER, output)
        return org_output

    def state_dict(self, destination=None, prefix="", keep_vars=False, **kwargs):
        return self.layer.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars, **kwargs
        )

    def load_state_dict(self, state_dict, strict=True):
        return self.layer.load_state_dict(state_dict, strict=strict)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.layer._load_from_state_dict(
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=strict,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )

    def named_modules(
        self,
        memo: Optional[Set["Module"]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        # outside of saving, we want the full names of modules in two cases:
        # 1. trainer initialization, so teacher is moved to the correct device. This is
        # caught by the kd_enabled flag, which is set when the modifier is started
        # 2. running in DataParallel (non-FSDP) mode so the replicate function can pick
        # up the teacher.
        if self._save_active or (self.kd_enabled and self._fsdp_active):
            return self.layer.named_modules(
                memo=memo, prefix=prefix, remove_duplicate=remove_duplicate
            )

        return super().named_modules(
            memo=memo, prefix=prefix, remove_duplicate=remove_duplicate
        )

    def prepare_for_save(self):
        """
        Prepare model structure to be saved, specifically `self.named_modules`
        """
        self._save_active = True

    def finish_save(self):
        """
        Finish saving model
        """
        self._save_active = False
