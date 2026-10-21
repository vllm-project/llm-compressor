import sys

import pytest

from llmcompressor.transformers.tracing.debug import parse_args

BASE_ARGV = [
    "llmcompressor.trace",
    "--model_id",
    "dummy/model",
    "--model_class",
    "AutoModelForCausalLM",
]


@pytest.mark.parametrize(
    "extra_argv,trust_remote_code,skip_weights",
    [
        ([], False, True),
        (["--trust_remote_code"], True, True),
        (["--no-trust_remote_code"], False, True),
        (["--skip_weights"], False, True),
        (["--no-skip_weights"], False, False),
        (["--trust_remote_code", "--no-skip_weights"], True, False),
    ],
)
def test_boolean_flags_are_toggleable(
    monkeypatch, extra_argv, trust_remote_code, skip_weights
):
    """Both boolean flags must be settable in either direction.

    ``type=bool`` would route every non-empty value through ``bool(str)``, so
    ``--skip_weights False`` evaluated to ``True`` and neither flag could be
    turned off from the command line.
    """
    monkeypatch.setattr(sys, "argv", BASE_ARGV + extra_argv)

    args = parse_args()

    assert args.trust_remote_code is trust_remote_code
    assert args.skip_weights is skip_weights
