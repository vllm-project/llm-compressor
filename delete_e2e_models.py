#!/usr/bin/env python3
"""Delete all nm-testing HuggingFace models ending in -e2e."""

import sys
from huggingface_hub import HfApi


ORG = "nm-testing"
SUFFIX = "-e2e"


def main():
    dry_run = "--delete" not in sys.argv

    api = HfApi()

    print(f"Fetching models for org: {ORG} ...")
    all_models = list(api.list_models(author=ORG))
    e2e_models = [m for m in all_models if m.id.endswith(SUFFIX)]

    print(f"\nFound {len(e2e_models)} models ending in '{SUFFIX}':\n")
    for m in e2e_models:
        print(f"  {m.id}")

    if dry_run:
        print(f"\nDry run — {len(e2e_models)} models would be deleted.")
        print("Run with --delete to actually delete them.")
        return

    confirm = input(f"\nDelete all {len(e2e_models)} models? Type 'yes' to confirm: ")
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        return

    failed = []
    for m in e2e_models:
        try:
            api.delete_repo(repo_id=m.id, repo_type="model")
            print(f"Deleted: {m.id}")
        except Exception as e:
            print(f"FAILED: {m.id} — {e}")
            failed.append(m.id)

    print(f"\nDone. Deleted {len(e2e_models) - len(failed)}/{len(e2e_models)} models.")
    if failed:
        print("Failed:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
