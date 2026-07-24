---
name: summarize-log-failures
description: Summarize test failures from log files
args:
  log_dir:
    type: string
    description: Path to directory containing .log files to analyze
    required: true
---

# Summarize Log Failures Skill

This skill analyzes test log files and generates a structured summary of test failures.

## Arguments

- `log_dir` (required): Path to directory containing `.log` files from failed test jobs

## Available Scripts

Scripts are located in `.claude/skills/summarize-log-failures/scripts/`:

- `sanitize_logs.py` - Sanitizes log files by removing ANSI codes, timestamps, progress indicators, and CI metadata. Use `--aggressive` flag to extract only error-related content.

## Templates

Templates are located in `.claude/skills/summarize-log-failures/templates/`:

- `failure_summary.md` - Template for formatting failure summaries with placeholders:
  - `{build_url}` - URL to the CI job (buildkite job or github job)
  - `{commit_sha}` - Git commit SHA
  - `{failed_job_count}` - Number of failed jobs
  - `{failed_tests_section}` - List of failed tests with details
  - `{error_analysis_section}` - Analysis of common errors and patterns
  - `{next_steps_section}` - Suggested remediation steps
  - `{full_details_section}` - Complete failure details for reference

## Steps to Summarize Log Failures

When this skill is invoked, the following steps will be completed:

1. **Sanitize log files**: Use the `sanitize_logs.py` script to clean all `.log` files in the specified directory, removing ANSI codes, timestamps, progress indicators, and CI metadata

2. **Read log files**: Read all sanitized `.log` files

3. **Parse test failures**: Extract failed test names, error messages, and tracebacks from each log file. Look for:
   - pytest failure patterns: `FAILED tests/...::test_name`
   - Error messages and assertions
   - Tracebacks with file/line information
   - Test result summaries (X passed, Y failed, Z skipped)

3. **Categorize failures**: Group failures by:
   - Common error types (AssertionError, RuntimeError, etc.)
   - Affected test modules/files
   - Similar error messages or patterns

4. **Generate summary**: Using the `failure_summary.md` template, create a summary that includes:
   - List of all failed tests with brief error descriptions
   - Analysis of common failure patterns
   - Suggested next steps for investigation
   - Full details in a collapsible section for reference

5. **Return summary**: Output the formatted summary as markdown

## Usage Example

```bash
# Invoke the skill with a directory of log files
# Logs will be automatically sanitized before analysis
/summarize-log-failures /path/to/log-files
```

## Output Format

The skill returns a markdown-formatted summary following the template structure:
- Clear test failure list
- Pattern analysis
- Actionable next steps
- Full details available on expansion
