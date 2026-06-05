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

### Step 2: Create workspace

Create a temporary workspace at `/tmp/validate-pr-<pr_number>` and `cd` into it. All subsequent work happens inside this directory.

### Step 3: Copy the trusted AWQ template

Before cloning the PR, copy the AWQ example from the **current repository** (the one this skill is defined in) into the workspace so the PR cannot tamper with it:
```bash
cp <current_repo_root>/examples/awq/llama_example.py /tmp/validate-pr-<pr_number>/llama_template.py
```
`<current_repo_root>` is the root of the repository that contains this skill (the directory that holds `.claude/skills/validate-pr/`).

### Step 4: Clone and check out the PR

```bash
gh repo clone vllm-project/llm-compressor ./pr-repo
cd ./pr-repo
gh pr checkout <pr_number>
```

Then diff against `main` to understand what changed:
```bash
git diff main...HEAD --stat
git diff main...HEAD
```

### Step 5: Identify new AWQ mappings

Inspect the diff from step 4. Look for new or modified AWQ mapping files (typically under `src/llmcompressor/modifiers/transform/awq/`). If the PR contains **no new AWQ mappings**, skip to step 10 and post a comment noting that no AWQ validation was needed.

### Step 6: Create test scripts

For each new mapping, create a test script in the workspace based on `llama_template.py` (copied in step 3). Adapt each script so that:
- `MODEL_ID` matches the architecture the mapping targets (choose a small variant of that architecture to keep runtime reasonable).
- The script name reflects the mapping being tested (e.g., `test_awq_<arch>.py`).

### Step 7: Set up the virtual environment

From the workspace root (`/tmp/validate-pr-<pr_number>`), install the PR's version of llm-compressor:
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ./pr-repo
```

If `ct_ref` was provided, also install that ref of compressed-tensors:
```bash
uv pip install "compressed-tensors @ git+https://github.com/vllm-project/compressed-tensors.git@<ct_ref>"
```

### Step 8: Run test scripts on GPU

Run each script using `chg` to reserve a GPU and avoid collisions:
```bash
chg run -- .venv/bin/python ./test_awq_<arch>.py
```

Run scripts **sequentially** (each needs a GPU). If a script fails, capture the error output and continue to the next script.

### Step 9: Analyze results

For each script that completed:
1. Parse stdout for `best_error` values reported per mapping.
2. Track whether `best_error` improved (decreased) relative to the first iteration's value.
3. A mapping **passes** if `best_error` improved over the first iteration.

### Step 10: Post report to PR

Build a markdown report and post it as a PR comment:

```bash
gh pr comment <pr_number> --body "<report>"
```

The report should include:
- **Summary**: PASS if >= 80% of mappings improved, FAIL otherwise (or N/A if no AWQ mappings were found).
- **Per-script results table**: script name, status (success/error), number of mappings tested, number that improved, first and final `best_error` values.
- **Error details**: for any script that failed, include the last 50 lines of output.

### Step 11: Clean up

```bash
cd /tmp
rm -rf /tmp/validate-pr-<pr_number>
```