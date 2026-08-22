# Summarize PR Failures Skill

Automates the analysis of failed Buildkite CI jobs for GitHub PRs. Downloads logs, summarizes failures, posts comments, and suggests fixes.

## Quick Start

```bash
# 1. Set up Buildkite token
export BUILDKITE_TOKEN="your_buildkite_api_token"

# 2. Use the skill
/summarize-pr-failures https://github.com/vllm-project/llm-compressor/pull/2899
```

## Setup

### 1. Buildkite API Token

Get a token from: https://buildkite.com/user/api-access-tokens

Required scopes:
- `read_builds`
- `read_job_logs`

Set the environment variable:
```bash
export BUILDKITE_TOKEN="bkua_..."
```

To persist, add to your shell profile (`~/.zshrc`, `~/.bashrc`, etc.):
```bash
echo 'export BUILDKITE_TOKEN="bkua_..."' >> ~/.zshrc
```

### 2. GitHub CLI

The `gh` CLI should already be configured. If not:
```bash
gh auth login
```

### 3. Buildkite Organization (Optional)

If your Buildkite organization slug differs from your GitHub org:
```bash
export BUILDKITE_ORG="your-buildkite-org-slug"
```

## How It Works

1. **Downloads logs** from failed Buildkite jobs using the Buildkite API
2. **Sanitizes logs** to remove ANSI codes, timestamps, and noise
3. **Analyzes failures** using the `summarize-log-failures` skill
4. **Gets PR diff** to understand code changes
5. **Suggests fixes** for obvious issues based on the diff
6. **Posts comment** to the PR with summary and suggestions

## Scripts

### download_buildkite_logs.sh

Downloads logs from failed jobs for a PR.

```bash
./scripts/download_buildkite_logs.sh <pr_url> <output_dir>
```

Example:
```bash
./scripts/download_buildkite_logs.sh \
  https://github.com/vllm-project/llm-compressor/pull/2899 \
  /tmp/pr-2899-logs
```

### sanitize_buildkite_logs.py

Cleans log files for analysis.

```bash
python scripts/sanitize_buildkite_logs.py <input_dir> [output_dir] [--aggressive]
```

Flags:
- `--aggressive`: Only keep error-related content (recommended)

Example:
```bash
python scripts/sanitize_buildkite_logs.py \
  /tmp/pr-2899-logs \
  /tmp/pr-2899-logs/sanitized \
  --aggressive
```

### post_pr_comment.sh

Posts a comment to a PR from a markdown file.

```bash
./scripts/post_pr_comment.sh <pr_number> <comment_file>
```

Example:
```bash
./scripts/post_pr_comment.sh 2899 /tmp/comment.md
```

## Manual Usage

If you want to use the scripts independently:

```bash
# Download logs
./scripts/download_buildkite_logs.sh \
  https://github.com/vllm-project/llm-compressor/pull/2899 \
  /tmp/logs

# Sanitize
python scripts/sanitize_buildkite_logs.py /tmp/logs /tmp/logs/clean --aggressive

# Analyze (using the other skill)
/summarize-log-failures /tmp/logs/clean

# Get diff for context
gh pr diff 2899 > /tmp/pr-2899.diff

# Post comment manually
gh pr comment 2899 --body "Your comment here"
```

## Troubleshooting

### "BUILDKITE_TOKEN environment variable is not set"

Set your token:
```bash
export BUILDKITE_TOKEN="bkua_..."
```

### "No Buildkite builds found for commit"

- Check that the PR has Buildkite checks configured
- Verify the Buildkite org slug matches (set `BUILDKITE_ORG` if different)
- Ensure the token has correct permissions

### "Could not extract PR number from URL"

Ensure the URL format is correct:
```
https://github.com/org/repo/pull/NUMBER
```

### "No failed jobs found"

If all jobs passed, the skill will report this and exit early. This is expected behavior.

## Related Skills

- `/summarize-log-failures` - Analyze log files and generate summaries (used internally)

## Examples

### Basic usage:
```bash
/summarize-pr-failures https://github.com/vllm-project/llm-compressor/pull/2899
```

### With custom Buildkite org:
```bash
export BUILDKITE_ORG="my-org-slug"
/summarize-pr-failures https://github.com/vllm-project/llm-compressor/pull/2899
```
