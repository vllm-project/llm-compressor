"""Deprecated alias for backward compatibility.

Use autoround_example.py as the single maintained entrypoint.
"""

import runpy
import sys
from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("autoround_example.py")
    print(
        "[DEPRECATED] qwen3_example.py is deprecated. "
        "Running autoround_example.py instead.",
        file=sys.stderr,
    )
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
