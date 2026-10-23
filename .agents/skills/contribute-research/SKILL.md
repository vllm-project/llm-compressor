---
name: contribute-research
description: >-
  Create, update, or review LLM Compressor Open Research Hub entries in
  docs/research from experiment results, logs, or issue discussions.
---

# Contribute to the Open Research Hub

Read [docs/research/index.md](../../../docs/research/index.md) before editing.
It is the single source of truth for submission, formatting, and attribution.
Follow it if this skill, its template, or an existing entry differs.

1. Read the supplied evidence and search `docs/research/` for related experiments.
   If the contributor has already posted an experiment and has related results
   to add, update their existing post. Otherwise, create a new entry.
2. For new entries, copy the Markdown block in the
   [entry template](templates/research-entry.md), keeping its block-style tags, into
   `docs/research/<short-title>-<username>.md`. Start the filename with the
   algorithm or scheme and end with the contributor's GitHub handle.
3. Fill the template from the evidence. Use the experiment date and the index's
   byline and author-consent rules. State hardware and calibration setup.
   Resolve missing information without inventing details;
   distinguish unmeasured values from measured results. Negative results are
   valuable and should be included.
4. Add reproduction details or source links when useful and available; these
   do not require additional experiments. For updates, retain original credits
   and experiment dates, explain corrections or new findings, and apply the
   index's updated-date guidance.
5. Check the entry against the index and source evidence, including numbers,
   links, and unfilled placeholders. New entries appear automatically; do not
   edit navigation files.

For rendering checks, follow [docs/DEVELOPMENT.md](../../../docs/DEVELOPMENT.md).
Keep generated build output out of the contribution. Report checks performed
and unresolved gaps; a docs build does not reproduce an experiment.

When asked to submit, follow [CONTRIBUTING.md](../../../CONTRIBUTING.md) and
[.github/PULL_REQUEST_TEMPLATE.md](../../../.github/PULL_REQUEST_TEMPLATE.md).
Otherwise, hand off the draft and its remaining gaps.
