#!/usr/bin/env python3
"""Prove the skill references only REAL, current llmcompressor / compressed-tensors
APIs and schemes -- a one-command trust check for reviewers.

It imports every symbol the skill's generated scripts use, and validates every
quantization scheme string the skill can emit against the live
``compressed_tensors`` preset registry. Nothing is downloaded and no model is
loaded; this only introspects the installed packages. Exit code is non-zero if
anything the skill claims does not exist in the installed environment.

    python verify_apis.py            # human-readable PASS/FAIL table
    python verify_apis.py --json     # machine-readable
"""

import argparse
import importlib
import json
import sys

# (module, symbol) pairs the skill's generated scripts and tools rely on.
REQUIRED_IMPORTS = [
    ("llmcompressor", "oneshot"),
    ("llmcompressor", "model_free_ptq"),
    ("llmcompressor.modifiers.quantization", "QuantizationModifier"),
    ("llmcompressor.modifiers.gptq", "GPTQModifier"),
    ("llmcompressor.modifiers.transform.awq", "AWQModifier"),
    ("llmcompressor.modifiers.transform.smoothquant", "SmoothQuantModifier"),
    ("compressed_tensors.offload", "dispatch_model"),
    ("transformers", "AutoModelForCausalLM"),
    ("transformers", "AutoModelForImageTextToText"),
    ("transformers", "AutoProcessor"),
    ("transformers", "AutoTokenizer"),
]

# Every scheme string the skill names in code or docs.
REQUIRED_SCHEMES = [
    "FP8",
    "FP8_DYNAMIC",
    "FP8_BLOCK",
    "W8A8",
    "INT8",
    "W8A16",
    "W4A16",
    "W4A16_ASYM",
    "NVFP4",
    "NVFP4A16",
    "MXFP4",
    "MXFP4A16",
    "MXFP8",
    "MXFP8A16",
]


def check_imports():
    results = []
    for module, symbol in REQUIRED_IMPORTS:
        try:
            mod = importlib.import_module(module)
            ok = hasattr(mod, symbol)
            detail = "" if ok else f"module has no attribute '{symbol}'"
        except Exception as exc:  # noqa: BLE001 - report any import failure
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        results.append((f"{module}.{symbol}", ok, detail))
    return results


def check_schemes():
    try:
        from compressed_tensors.quantization.quant_scheme import PRESET_SCHEMES

        known = set(PRESET_SCHEMES.keys())
    except Exception as exc:  # noqa: BLE001
        return [
            (s, False, f"registry import failed: {exc}") for s in REQUIRED_SCHEMES
        ], None
    results = []
    for s in REQUIRED_SCHEMES:
        ok = s in known
        results.append((s, ok, "" if ok else "not in PRESET_SCHEMES"))
    return results, sorted(known)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    versions = {}
    for pkg in ("llmcompressor", "compressed_tensors", "transformers"):
        try:
            versions[pkg] = importlib.import_module(pkg).__version__
        except Exception:  # noqa: BLE001
            versions[pkg] = "NOT INSTALLED"

    imports = check_imports()
    schemes, known = check_schemes()
    all_ok = all(ok for _, ok, _ in imports) and all(ok for _, ok, _ in schemes)

    if args.json:
        json.dump(
            {
                "versions": versions,
                "imports": [{"item": i, "ok": ok, "detail": d} for i, ok, d in imports],
                "schemes": [
                    {"scheme": s, "ok": ok, "detail": d} for s, ok, d in schemes
                ],
                "registry_schemes": known,
                "all_ok": all_ok,
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        print("=" * 64)
        print("verify_apis -- skill API/scheme reality check")
        print("=" * 64)
        for pkg, ver in versions.items():
            print(f"  {pkg:<20} {ver}")
        print("\nImports the skill relies on:")
        for item, ok, detail in imports:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {item}" + (f"  <- {detail}" if detail else ""))
        print("\nScheme strings the skill emits (vs live PRESET_SCHEMES):")
        for s, ok, detail in schemes:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {s}" + (f"  <- {detail}" if detail else ""))
        print("\n" + ("ALL CHECKS PASSED" if all_ok else "FAILURES DETECTED"))
        if not all_ok:
            print(
                "Some APIs/schemes are missing in the installed packages. "
                "Upgrade llmcompressor/compressed-tensors (install from "
                "source for the latest) and re-run."
            )

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
