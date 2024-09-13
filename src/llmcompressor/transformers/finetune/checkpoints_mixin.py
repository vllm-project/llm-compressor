from typing import Optional


class SafeCheckpointsMixin:
    """
    Assumes that all loading and saving of checkpoints is done within the train call
    and that all model saving after finalization happens outisdeof the train call

    | Checkpoints     | Normal         | Distillation            |
    | --------------- | -------------- | ----------------------- |
    | save_compressed | Precision loss | Precision loss          |
    | save_safetensors| OK             | Error, shared tensors   |


    """

    def train(self, *args, stage: Optional[str] = None, **kwargs):
        # capture original args
        original_save_compressed = self.args.save_compressed
        original_save_safetensors = self.args.save_safetensors

        # use safe args
        # distillation training checkpoints contain fused state dicts with
        # shared tensors which are not compatible with safe_tensors
        self.args.save_compressed = False
        if self.teacher is not None and self.teacher not in ("disable", "self"):
            self.args.save_safetensors = False

        output = super().train(*args, stage=stage, **kwargs)

        # restore original args
        self.args.save_compressed = original_save_compressed
        self.args.save_safetensors = original_save_safetensors

        return output
