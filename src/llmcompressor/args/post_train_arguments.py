from dataclasses import dataclass, field
from typing import Optional

from llmcompressor.pipelines.registry import PIPELINES


@dataclass
class PostTrainArguments:
    pipeline: Optional[str] = field(
        default="independent",
        metadata={
            "help": "Calibration pipeline used to calibrate model. "
            f"Options: {PIPELINES.keys()}"
        },
    )

    save_path: Optional[str] = field(
        default=None
    )