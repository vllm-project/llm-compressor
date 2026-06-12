---
name: validate-pr
description: Validate an llm-compressor PR by running AWQ mapping tests on GPU and posting results to the PR.
user-invocable: true
args:
  pr_number:
    type: int
    description: The PR number on github (e.g., 2800)
    required: true
  ct_ref:
    type: string
    description: Optionally install compressed-tensors from this git ref (e.g. "main")
    required: false
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Agent
---

# Validate PR Skill

Validate an llm-compressor PR by checking out the code, identifying new AWQ mappings, running test scripts on GPU, and posting a pass/fail report as a PR comment.

## Arguments

- `pr_number` (required): The LLM Compressor PR number to validate.
- `ct_ref` (optional): A git ref for `compressed-tensors` to install (e.g. "main"). If unset, the version pinned by `llm-compressor` is used.

## Usage

Trigger examples (all resolve to `pr_number=2800`):
- "Validate PR #2800"
- "Validate https://github.com/vllm-project/llm-compressor/pull/2800"

## Implementation

### Step 1: Verify prerequisites

Run each of these and abort with a clear error if any fails:
```
gh auth status
chg status
huggingface-cli whoami
uv --version
```

### Step 2: Clone and check out the PR

Clone into a subdirectory of the current working directory:
```bash
gh repo clone vllm-project/llm-compressor ./validate-pr-<pr_number>
cd ./validate-pr-<pr_number>
gh pr checkout <pr_number>
```

Then diff against `main` to understand what changed:
```bash
git diff main...HEAD --stat
git diff main...HEAD
```

### Step 3: Identify new AWQ mappings

Inspect the diff from step 2. Look for new or modified AWQ mapping files (typically under `src/llmcompressor/modifiers/transform/awq/`). If the PR contains **no new AWQ mappings**, skip to step 8 and post a comment noting that no AWQ validation was needed.

### Step 4: Determine model IDs

For each new mapping, choose a model ID whose architecture matches the mapping's target architecture. Prefer small variants to keep runtime reasonable.

### Step 5: Set up the virtual environment

From inside `./validate-pr-<pr_number>/`:
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

If `ct_ref` was provided, also install that ref of compressed-tensors:
```bash
uv pip install "compressed-tensors @ git+https://github.com/vllm-project/compressed-tensors.git@<ct_ref>"
```

### Step 6: Run AWQ template on GPU for each mapping

Copy the trusted template from this skill into the workspace, then run it once per mapping with the appropriate model ID:
```bash
cp .claude/skills/validate-pr/scripts/awq_template.py ./validate-pr-<pr_number>/awq_template.py
chg run -- .venv/bin/python ./awq_template.py --model-id <model_id>
```

Run each invocation **sequentially** (each needs a GPU). If a run fails, capture the error output and continue to the next mapping.

### Step 7: Analyze results

For each script that completed:
1. Parse stdout for `best_error` values reported per mapping.
2. Track whether `best_error` improved (decreased) relative to the first iteration's value.
3. A mapping **passes** if `best_error` improved over the first iteration.

### Step 8: Post report to PR

Build a markdown report and post it as a PR comment:

```bash
gh pr comment <pr_number> --body "<report>"
```

The report should include:
- **Summary**: PASS if >= 80% of mappings improved, FAIL otherwise (or N/A if no AWQ mappings were found).
- **Per-script results table**: script name, status (success/error), number of mappings tested, number that improved, first and final `best_error` values.
- **Error details**: for any script that failed, include the last 50 lines of output.

### Step 9: Clean up

```bash
cd ..
rm -rf ./validate-pr-<pr_number>
```