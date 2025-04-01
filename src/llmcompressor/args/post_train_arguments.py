from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class PostTrainArguments:
    pipeline: Optional[str] = field(
        default=None,
        metadata={"help": "Calibration pipeline used to calibrate model."},
    )

    oneshot_device: Optional[torch.device] = field(default=None)

    output_dir: Optional[str] = field(default=None)
