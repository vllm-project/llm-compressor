#!/usr/bin/env python3
"""
Sync config.json fields from BF16 model to quantized model,
preserving the quantization_config in the target.
"""
import json
import argparse
from pathlib import Path


def sync_configs(source_path: str, target_path: str, dry_run: bool = False):
    """
    Update target config with fields from source config, except quantization_config.

    Args:
        source_path: Path to the source config.json (BF16 model)
        target_path: Path to the target config.json (quantized model)
        dry_run: If True, print changes without writing
    """
    # Load both configs
    print(f"Loading source config from: {source_path}")
    with open(source_path, 'r') as f:
        source_config = json.load(f)

    print(f"Loading target config from: {target_path}")
    with open(target_path, 'r') as f:
        target_config = json.load(f)

    # Preserve the quantization_config from target
    quantization_config = target_config.get('quantization_config')

    if quantization_config is None:
        print("WARNING: No quantization_config found in target!")
    else:
        print(f"Preserving quantization_config (size: {len(json.dumps(quantization_config))} chars)")

    # Track changes
    added_keys = []
    modified_keys = []
    removed_keys = []

    # Find keys that will be added or modified
    for key, value in source_config.items():
        if key == 'quantization_config':
            continue  # Skip this key entirely

        if key not in target_config:
            added_keys.append(key)
        elif target_config[key] != value:
            modified_keys.append(key)

    # Find keys that will be removed (exist in target but not source)
    for key in target_config.keys():
        if key == 'quantization_config':
            continue  # Don't remove this
        if key not in source_config:
            removed_keys.append(key)

    # Print summary
    print("\n=== Change Summary ===")
    print(f"Keys to add: {len(added_keys)}")
    if added_keys:
        for key in added_keys:
            print(f"  + {key}")

    print(f"\nKeys to modify: {len(modified_keys)}")
    if modified_keys:
        for key in modified_keys:
            print(f"  ~ {key}")
            print(f"    Old: {target_config[key]}")
            print(f"    New: {source_config[key]}")

    print(f"\nKeys to remove: {len(removed_keys)}")
    if removed_keys:
        for key in removed_keys:
            print(f"  - {key}")

    if quantization_config:
        print("\nPreserved key:")
        print("  = quantization_config (unchanged)")

    # Create new config
    new_config = source_config.copy()

    # Restore quantization_config
    if quantization_config is not None:
        new_config['quantization_config'] = quantization_config

    # Write or show result
    if dry_run:
        print("\n=== DRY RUN - No changes written ===")
        print("\nTo apply changes, run without --dry-run")
    else:
        # Backup original
        backup_path = f"{target_path}.backup"
        print(f"\nBacking up original to: {backup_path}")
        with open(backup_path, 'w') as f:
            json.dump(target_config, f, indent=2)

        # Write new config
        print(f"Writing updated config to: {target_path}")
        with open(target_path, 'w') as f:
            json.dump(new_config, f, indent=2)

        print("\n=== SUCCESS ===")
        print(f"Config updated successfully!")
        print(f"Backup saved to: {backup_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sync config.json fields while preserving quantization_config"
    )
    parser.add_argument(
        '--source',
        default='/mnt/nvme-data/engine/kylesayrs/DeepSeek-V4-Pro-BF16/config.json',
        help='Path to source config.json (default: BF16 model)'
    )
    parser.add_argument(
        '--target',
        default='/mnt/nvme-data/engine/kylesayrs/DeepSeek-V4-Pro-NVFP4-FP8-BLOCK/config.json',
        help='Path to target config.json (default: quantized model)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would change without modifying files'
    )

    args = parser.parse_args()

    # Verify files exist
    if not Path(args.source).exists():
        print(f"ERROR: Source file not found: {args.source}")
        return 1

    if not Path(args.target).exists():
        print(f"ERROR: Target file not found: {args.target}")
        return 1

    sync_configs(args.source, args.target, args.dry_run)
    return 0


if __name__ == '__main__':
    exit(main())
