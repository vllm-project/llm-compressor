#!/usr/bin/env bash
# Download Buildkite logs for failed jobs in a PR
# Usage: ./download_buildkite_logs.sh <pr_url> <output_dir>

set -euo pipefail

PR_URL="${1:-}"
OUTPUT_DIR="${2:-buildkite-logs}"

if [ -z "$PR_URL" ]; then
    echo "Error: PR URL is required"
    echo "Usage: $0 <pr_url> [output_dir]"
    exit 1
fi

# Check if bk CLI is available
if ! command -v bk &> /dev/null; then
    echo "Error: 'bk' CLI is not installed or not in PATH"
    echo "Please install it from: https://github.com/buildkite/cli"
    exit 1
fi

# Extract PR number from URL
PR_NUMBER=$(echo "$PR_URL" | grep -oE '[0-9]+$' || echo "")
if [ -z "$PR_NUMBER" ]; then
    echo "Error: Could not extract PR number from URL: $PR_URL"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Fetching PR #$PR_NUMBER details..."

# Get PR details including the head SHA
PR_JSON=$(gh pr view "$PR_NUMBER" --json headRefOid,number,headRefName)
HEAD_SHA=$(echo "$PR_JSON" | jq -r '.headRefOid')

if [ -z "$HEAD_SHA" ] || [ "$HEAD_SHA" = "null" ]; then
    echo "Error: Could not determine HEAD SHA for PR #$PR_NUMBER"
    exit 1
fi

echo "HEAD SHA: $HEAD_SHA"

# Get the repository name (org/repo format)
REPO=$(gh repo view --json nameWithOwner -q '.nameWithOwner')
echo "Repository: $REPO"

echo "Fetching Buildkite builds for commit $HEAD_SHA..."

# List builds for this commit using bk CLI
# Format: build list with commit filter and JSON output
BUILDS_JSON=$(bk build list --commit "$HEAD_SHA" --format json 2>/dev/null || echo "[]")

if [ "$BUILDS_JSON" = "[]" ] || [ "$(echo "$BUILDS_JSON" | jq 'length')" -eq 0 ]; then
    echo "No Buildkite builds found for commit $HEAD_SHA"
    exit 0
fi

# Get the most recent build number
BUILD_NUMBER=$(echo "$BUILDS_JSON" | jq -r '.[0].number')
BUILD_URL=$(echo "$BUILDS_JSON" | jq -r '.[0].web_url')

echo "Found build #$BUILD_NUMBER: $BUILD_URL"

# Get build details with jobs using bk CLI
BUILD_DETAILS=$(bk build view "$BUILD_NUMBER" --format json)

# Extract failed job IDs
FAILED_JOBS=$(echo "$BUILD_DETAILS" | jq -r '.jobs[] | select(.state == "failed") | "\(.id)|\(.name // "unknown")"')

if [ -z "$FAILED_JOBS" ]; then
    echo "No failed jobs found in build #$BUILD_NUMBER"
    exit 0
fi

echo "Downloading logs for failed jobs..."

LOG_COUNT=0
while IFS='|' read -r JOB_ID JOB_NAME; do
    # Sanitize job name for filename
    SAFE_JOB_NAME=$(echo "$JOB_NAME" | tr '/' '_' | tr ' ' '_' | tr ':' '_')
    LOG_FILE="${OUTPUT_DIR}/${BUILD_NUMBER}_${JOB_ID}_${SAFE_JOB_NAME}.log"

    echo "  - Downloading: $JOB_NAME (Job ID: $JOB_ID)"

    # Get job log using bk CLI
    bk job log "$JOB_ID" > "$LOG_FILE" 2>/dev/null || {
        echo "    Warning: Could not download log for job $JOB_ID"
        continue
    }

    LOG_COUNT=$((LOG_COUNT + 1))
done <<< "$FAILED_JOBS"

echo ""
echo "Downloaded $LOG_COUNT log file(s) to: $OUTPUT_DIR"
echo "Build URL: $BUILD_URL"
