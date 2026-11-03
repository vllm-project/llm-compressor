---
title: Open Research Hub
---

# Open Research Hub

A community log of experimental compression results. If you have tried an
algorithm, scheme, or model combination that is not yet covered by the official
guides, post your findings here so others can learn from them.

Entries are not official benchmarks. They are field reports: what you ran, what
hardware you ran it on, and what you measured. Negative results are welcome and
are often more useful than positive ones.

## Finding entries

Every entry carries a list of tags. Tags are indexed by the site search, so
typing an algorithm, scheme, or model name into the search box at the top of
the page will surface the entries that use it. Searching for `AWQ` or `NVFP4`
is the fastest way to find prior work before you start your own run.

## How to submit

Copy the template below into `docs/research/<short-title>-<username>.md`, fill
it in, and open a pull request. For example,
`awq-hadamard-qwen3-soyr-redhat.md`. You do not need to edit any navigation
file; new entries are picked up automatically.

Lead the filename with the algorithm or scheme so that related entries sit
next to each other in the sidebar, and end it with your GitHub handle so that
two people writing about the same combination do not collide.

If you've already posted an experiment and have related results to add,
please update your existing post.

Coding agents can use the
[`contribute-research` skill](https://github.com/vllm-project/llm-compressor/blob/main/.agents/skills/contribute-research/SKILL.md)
to draft or review entries using the template below.

```markdown
---
title: "Short description of what you tried"
tags:
  - AWQ
  - NVFP4
  - Qwen3.5
---

<div class="entry-byline" markdown>
[![USERNAME](https://github.com/USERNAME.png?size=64){ .entry-avatar }](https://github.com/USERNAME)
**[@USERNAME](https://github.com/USERNAME)** posted on **YYYY-MM-DD**
</div>

# Short description of what you tried

## Summary

One or two sentences on the headline result.

## Setup

- Model: `org/model-id`
- Scheme: NVFP4
- Calibration: dataset, number of sequences, sequence length
- Hardware: 4x H100 80GB

## Results

| Configuration | Metric | Metric |
|---------------|--------|--------|
| BF16 baseline | | |
| Your run | | |

## Observations

What surprised you, what you would try next, what you did not test.
```

Replace `USERNAME` with your GitHub handle and `YYYY-MM-DD` with the date you
ran the experiment. The experiment date is more useful to readers than the date
the pull request merged, since results age with the library and the model.

### Crediting more than one author

An entry can carry any number of authors. Add one avatar line per person, in
the order you want them credited, and list the handles in the sentence below
them. The avatars stack automatically, so the byline stays compact no matter
how many you add.

```markdown
<div class="entry-byline" markdown>
[![ALICE](https://github.com/ALICE.png?size=64){ .entry-avatar }](https://github.com/ALICE)
[![BOB](https://github.com/BOB.png?size=64){ .entry-avatar }](https://github.com/BOB)
[![CAROL](https://github.com/CAROL.png?size=64){ .entry-avatar }](https://github.com/CAROL)
**[@ALICE](https://github.com/ALICE)**, **[@BOB](https://github.com/BOB)**, and
**[@CAROL](https://github.com/CAROL)** posted on **YYYY-MM-DD**
</div>
```

If people contributed at different times, add an updated date, for example
`posted on **2026-08-25**, updated **2026-10-09**`.

Credit anyone whose results or analysis you incorporated, and ask them before
you add their name. If someone's work is referenced rather than incorporated,
link to it in the body instead of adding them to the byline.

## Conventions

- Use block-style YAML lists for `tags`. Inline lists such as `[AWQ, GPTQ]`
  are parsed as a Markdown link by the documentation link checker and will
  produce a build warning.
- Keep `tags` to the algorithms, schemes, and model families a reader would
  plausibly search for.
- State your hardware and calibration setup. A result without them is hard to
  compare against.
