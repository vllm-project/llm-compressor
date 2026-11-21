---
name: cleanup-tests
description: >-
  Review tests in the tests folder and clean up the suite: remove tests of
  trivial properties, mark long-running but useful tests with nightly or
  weekly cadence marks, and refactor tests that are overly verbose or hard
  to understand.
---

# Clean Up Tests

Review the test suite under [tests/](../../../tests/) and apply the three
kinds of cleanup below. Work in small batches, verify each batch, and never
reduce real coverage: when in doubt whether a test is useful, keep it.

## 1. Remove trivial tests

Delete a test only when all of the following hold:

- It tests a trivial property rather than llmcompressor behavior: the
  assertion merely restates the test's own arrangement (a tautology), it
  checks the language or a third-party library rather than our code (dict
  ordering, tensor dtype defaults, `isinstance` on a literal), or it
  re-asserts exactly what another test already covers with no added value.
- Deleting it loses no meaningful regression coverage.
- Nothing references it: grep the repository for the test name before
  removing it (fixtures, scripts, `.buildkite/`, `docs/`, CI configs).

Record every removal with a one-line justification for the final report.

## 2. Mark long-running but useful tests

Slow tests that still catch real regressions should move to a slower cadence
instead of running on every commit:

- Pytest tests: add `@pytest.mark.nightly` or `@pytest.mark.weekly`. Both
  markers are registered in `pyproject.toml` and `pytest-xpu.ini`. They are
  consumed through the `CADENCE` variable (see the Makefile): the default
  commit cadence skips marked tests, `CADENCE=nightly` runs nightly-marked
  tests, `CADENCE=weekly` runs weekly-marked tests, and `CADENCE=release`
  runs everything. This matches the semantics of the existing
  `requires_cadence` decorator in `tests/testing_utils.py`, which is the
  older equivalent mechanism: leave existing `requires_cadence` uses alone,
  but prefer the markers for new marks. Keep existing markers such as `unit`
  or `multi_gpu`: a cadence mark stacks with them, it does not replace them.
- Config-driven e2e and lm-eval tests (YAML files under
  `tests/e2e/configs/` and `tests/lmeval/configs/`): set or extend the
  existing `cadence:` field in the YAML config instead of adding a pytest
  mark.

Choose the cadence from measured runtime when possible
(`pytest --durations=25`). As a guideline, mark `nightly` when a test runs
on the order of minutes, and `weekly` when it runs tens of minutes or more,
downloads very large models, or requires multiple GPUs. Do not mark tests
that finish in seconds; they belong on the commit cadence.

## 3. Refactor verbose or hard-to-understand tests

For tests that are hard to follow:

- Collapse copy-pasted variants into `pytest.mark.parametrize`.
- Move repeated setup or model construction into fixtures or shared helpers.
- Replace magic numbers and repeated literals with named constants.
- Rename tests and variables so the intent and expected behavior are clear
  at a glance.
- Split tests that assert many unrelated things into focused tests.

A refactor must not change what a test verifies: keep assertions and
expected values identical.

## Verification

After each batch of changes:

1. Run every test file you touched, following
   [.agents/skills/test.md](../test.md): use `canhazgpu` when it is
   available, since most tests require a GPU.
2. Run `make style` and `make quality` and fix any reported issues, as
   described in [.agents/skills/style.md](../style.md).

Finish with a report listing removed tests (with reasons), marked tests
(with the mark chosen and rough runtime), and refactored tests (with a
summary of each change). When running unattended, for example on a schedule,
commit the changes and use the report as the commit message.
