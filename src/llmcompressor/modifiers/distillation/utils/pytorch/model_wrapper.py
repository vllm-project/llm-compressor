from typing import Any, Dict, Optional, Set

import torch
from torch.nn import Module

__all__ = ["KDModelWrapper"]


class KDModelWrapper(Module):
    KD_LAST_COMPARISON = "kd_last_comparison"

    def __init__(
        self,
        student_model: Module,
        teacher_model: Module,
        wrappers: Dict[str, Any],
        comparison,
        fsdp_active: bool,
    ):
        super(KDModelWrapper, self).__init__()

        self.student_model = student_model
        self.teacher_model = teacher_model
        self.wrappers = wrappers
        self.kd_comparison = comparison
        self._save_active = False
        self._fsdp_active = fsdp_active
        self.kd_enabled = False
        self.register_buffer(self.KD_LAST_COMPARISON, torch.zeros(1, device="cpu"))
        self._init_called = True  # make sure this is last property to be set

        def _clear_missing_keys(module, incompatible_keys):
            incompatible_keys.missing_keys.clear()

        self.register_load_state_dict_post_hook(_clear_missing_keys)

    def forward(self, *args, **kwargs):
        if not self.kd_enabled:
            return self.student_model(*args, **kwargs)

        org_output = self.student_model(*args, **kwargs)
        with torch.no_grad():
            self.teacher_model(*args, **kwargs)

        layerwise_comps = []
        nonpad_tokens = kwargs["attention_mask"] == 1
        device = nonpad_tokens.device
        for key, (student_wrapper, teacher_wrapper) in self.wrappers.items():
            student_out = student_wrapper.kd_last_transformed.to(device)[nonpad_tokens]
            teacher_out = teacher_wrapper.kd_last_transformed.to(device)[nonpad_tokens]
            comp = self.kd_comparison(student_out, teacher_out)
            layerwise_comps.append(comp)

        setattr(self, self.KD_LAST_COMPARISON, torch.stack(layerwise_comps).mean())

        return org_output

    def state_dict(self, destination=None, prefix="", keep_vars=False, **kwargs):
        return self.student_model.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars, **kwargs
        )

    def load_state_dict(self, state_dict, strict=True):
        return self.student_model.load_state_dict(state_dict, strict=strict)

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
        self.student_model._load_from_state_dict(
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
            return self.student_model.named_modules(
                memo=memo, prefix=prefix, remove_duplicate=remove_duplicate
            )

        return super().named_modules(
            memo=memo, prefix=prefix, remove_duplicate=remove_duplicate
        )

    def named_children(self):
        return self.student_model.named_children()

    def train(self, mode: bool = True):
        self.student_model.train(mode)
        return self

    def prepare_for_save(self):
        """
        Prepare model structure to be saved, specifically `self.named_modules`
        """
        self._save_active = True
        for student_wrapper, teacher_wrapper in self.wrappers.values():
            student_wrapper.prepare_for_save()
            teacher_wrapper.prepare_for_save()

    def finish_save(self):
        """
        Finish saving model
        """
        self._save_active = False
        for student_wrapper, teacher_wrapper in self.wrappers.values():
            student_wrapper.finish_save()
            teacher_wrapper.finish_save()

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.student_model, name)
