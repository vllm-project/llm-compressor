# Summarize Log Failures Skill

Analyzes test log files and generates structured summaries of test failures.

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
/summarize-log-failures /tmp/test-logs
```

## What It Does

1. **Sanitizes logs** using the `sanitize_logs.py` script to remove noise (ANSI codes, timestamps, progress indicators, CI metadata)
2. **Reads all `.log` files** in the specified directory
3. **Extracts test failures** including:
   - Test names and paths
   - Error messages
   - Tracebacks
   - Assertion failures
4. **Categorizes failures** by:
   - Error type
   - Affected modules
   - Common patterns
5. **Generates summary** using the template at `templates/failure_summary.md`

## Output Format

The skill outputs a markdown-formatted summary with:

- **Failed Tests**: List of all failed tests with error descriptions
- **Error Analysis**: Common patterns and groupings
- **Next Steps**: Suggested actions for investigation
- **Full Details**: Complete failure information (in collapsible section)

## Template

The summary uses `templates/failure_summary.md` with these placeholders:

- `{build_url}` - Link to CI job (buildkite job or github job)
- `{commit_sha}` - Git commit SHA
- `{failed_job_count}` - Number of failed jobs
- `{failed_tests_section}` - Formatted list of failures
- `{error_analysis_section}` - Pattern analysis
- `{next_steps_section}` - Remediation suggestions
- `{full_details_section}` - Complete details

## Example Workflow

```bash
# Analyze test logs from any source
# Logs are automatically sanitized before analysis
/summarize-log-failures /tmp/test-logs
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
