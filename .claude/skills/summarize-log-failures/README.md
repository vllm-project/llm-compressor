# Summarize Log Failures Skill

Analyzes Buildkite log files and generates structured summaries of test failures.

## Quick Start

```bash
/summarize-log-failures /path/to/logs
```

## Usage

This skill is primarily used as a component of the `summarize-pr-failures` skill, but can also be used independently to analyze any directory of `.log` files.

### Prerequisites

- Directory containing `.log` files from failed CI jobs
- Logs should ideally be sanitized first (see below)

### Invocation

```bash
# Analyze logs in a directory
/summarize-log-failures /tmp/buildkite-logs

# Or with sanitized logs
/summarize-log-failures /tmp/buildkite-logs/sanitized
```

## What It Does

1. **Reads all `.log` files** in the specified directory
2. **Extracts test failures** including:
   - Test names and paths
   - Error messages
   - Tracebacks
   - Assertion failures
3. **Categorizes failures** by:
   - Error type
   - Affected modules
   - Common patterns
4. **Generates summary** using the template at `templates/failure_summary.md`

## Output Format

The skill outputs a markdown-formatted summary with:

- **Failed Tests**: List of all failed tests with error descriptions
- **Error Analysis**: Common patterns and groupings
- **Next Steps**: Suggested actions for investigation
- **Full Details**: Complete failure information (in collapsible section)

## Template

The summary uses `templates/failure_summary.md` with these placeholders:

- `{build_url}` - Link to CI build
- `{commit_sha}` - Git commit SHA
- `{failed_job_count}` - Number of failed jobs
- `{failed_tests_section}` - Formatted list of failures
- `{error_analysis_section}` - Pattern analysis
- `{next_steps_section}` - Remediation suggestions
- `{full_details_section}` - Complete details

## Sanitizing Logs

For best results, sanitize logs first using the script from `summarize-pr-failures`:

```bash
python .claude/skills/summarize-pr-failures/scripts/sanitize_buildkite_logs.py \
  /tmp/logs \
  /tmp/logs/sanitized \
  --aggressive
```

This removes:
- ANSI escape codes
- Timestamps
- Progress indicators
- Decorative elements
- Non-error content (with `--aggressive`)

## Example Workflow

```bash
# 1. Download logs (using summarize-pr-failures script)
.claude/skills/summarize-pr-failures/scripts/download_buildkite_logs.sh \
  https://github.com/org/repo/pull/123 \
  /tmp/logs

# 2. Sanitize
python .claude/skills/summarize-pr-failures/scripts/sanitize_buildkite_logs.py \
  /tmp/logs \
  /tmp/logs/sanitized \
  --aggressive

# 3. Summarize
/summarize-log-failures /tmp/logs/sanitized
```

## Common Test Failure Patterns

The skill recognizes various test failure formats:

### pytest
```
FAILED tests/test_example.py::test_function - AssertionError: ...
```

### Python exceptions
```
Traceback (most recent call last):
  File "...", line X, in test_function
    ...
AssertionError: expected X but got Y
```

### Test summaries
```
===== 10 failed, 50 passed, 5 skipped in 30.00s =====
```

## Integration

This skill is designed to be called by `summarize-pr-failures` but can be used standalone for:

- Manual log analysis
- Local CI debugging  
- One-off failure investigation
- Custom automation workflows

## Related Skills

- `/summarize-pr-failures` - Complete PR failure analysis workflow (downloads logs, analyzes, posts comments)
