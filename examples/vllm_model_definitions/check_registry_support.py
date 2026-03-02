"""Check whether one or more architecture names are supported by vLLM.

Usage:
  python3 check_registry_support.py --architecture LlamaForCausalLM
  python3 check_registry_support.py --architecture Qwen3ForCausalLM --architecture MistralForCausalLM
"""

import argparse

from vllm.model_executor.models import ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        action="append",
        required=True,
        help="Architecture name (repeat flag for multiple values).",
    )
    args = parser.parse_args()

    supported_archs = set(ModelRegistry.get_supported_archs())

    for arch in args.architecture:
        status = "supported" if arch in supported_archs else "not_supported"
        print(f"{arch}: {status}")


if __name__ == "__main__":
    main()
