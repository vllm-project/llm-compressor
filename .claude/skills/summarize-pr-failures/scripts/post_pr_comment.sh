#!/usr/bin/env bash
# Post a comment to a GitHub PR
# Usage: ./post_pr_comment.sh <pr_number> <comment_file>

set -euo pipefail

PR_NUMBER="${1:-}"
COMMENT_FILE="${2:-}"

if [ -z "$PR_NUMBER" ] || [ -z "$COMMENT_FILE" ]; then
    echo "Error: PR number and comment file are required"
    echo "Usage: $0 <pr_number> <comment_file>"
    exit 1
fi

if [ ! -f "$COMMENT_FILE" ]; then
    echo "Error: Comment file does not exist: $COMMENT_FILE"
    exit 1
fi

echo "Posting comment to PR #$PR_NUMBER..."

gh pr comment "$PR_NUMBER" --body-file "$COMMENT_FILE"

echo "Comment posted successfully!"
