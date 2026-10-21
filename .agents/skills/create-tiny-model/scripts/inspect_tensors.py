"""Inspect safetensors model metadata: name, shape, dtype for each tensor."""

import argparse
import json
from pathlib import Path

from safetensors import safe_open


def inspect_file(path: Path):
    with safe_open(str(path), framework="pt") as f:
        for key in sorted(f.keys()):
            shape = list(f.get_slice(key).get_shape())
            dtype = str(f.get_slice(key).get_dtype())
            yield key, shape, dtype


def main():
    parser = argparse.ArgumentParser(description="Inspect safetensors model metadata")
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()

    index_path = args.model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            shard_names = sorted(set(json.load(f)["weight_map"].values()))
        files = [args.model_dir / s for s in shard_names]
    else:
        files = sorted(args.model_dir.glob("*.safetensors"))

    if not files:
        print("No safetensors files found.")
        return

    rows = []
    for f in files:
        rows.extend(inspect_file(f))

    name_w = max(len(r[0]) for r in rows)
    shape_w = max(len(str(r[1])) for r in rows)
    dtype_w = max(len(r[2]) for r in rows)

    header = f"{'Name':<{name_w}}  {'Shape':<{shape_w}}  {'Dtype':<{dtype_w}}"
    print(header)
    print("-" * len(header))
    for name, shape, dtype in rows:
        print(f"{name:<{name_w}}  {str(shape):<{shape_w}}  {dtype:<{dtype_w}}")

    print(f"\nTotal: {len(rows)} tensors")


if __name__ == "__main__":
    main()
