"""Load a model with vLLM runtime and run a generation.

This is vLLM-native model loading.

Usage:
  python3 load_with_vllm_runtime.py --model meta-llama/Meta-Llama-3-8B-Instruct
"""

import argparse

from vllm import LLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model id or local path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="My name is",
        help="Prompt to generate from.",
    )
    args = parser.parse_args()

    llm = LLM(model=args.model)
    outputs = llm.generate(args.prompt)
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
