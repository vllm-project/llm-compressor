#!/usr/bin/env python3
"""
Simple script: run AWQ with smooth_layer_quantization on Modal.

Usage:
  # From repo root: modal run examples/awq/modal_awq_with_smooth.py
  # From examples/awq: modal run modal_awq_with_smooth.py  (no path prefix)
  # Add --skip-lm-eval to skip evaluation
"""

import sys
from pathlib import Path

_awq_dir = Path(__file__).resolve().parent
if str(_awq_dir) not in sys.path:
    sys.path.insert(0, str(_awq_dir))

from modal_awq_runners import app, run_with_smooth  # noqa: E402


@app.local_entrypoint()
def main(skip_lm_eval: bool = False):
    """Run with-smooth on Modal and print result."""
    result = run_with_smooth.remote(skip_lm_eval=skip_lm_eval)
    print("With-smooth result:", result)
