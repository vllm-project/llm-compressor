from dataclasses import dataclass, field
from typing import Optional

import torch

from llmcompressor.pipelines.registry import PIPELINES


@dataclass
class PostTrainArguments:
    pipeline: Optional[str] = field(
        default=None,
        metadata={
            "help": "Calibration pipeline used to calibrate model. "
            f"Options: {PIPELINES.keys()}"
        },
    )

    oneshot_device: Optional[torch.device] = field(default=None)

    save_path: Optional[str] = field(default=None)
