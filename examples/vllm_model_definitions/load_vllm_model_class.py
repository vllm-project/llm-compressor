"""Load a model class from the vLLM model registry.

This loads the model *definition* directly from vLLM's model folder via
`ModelRegistry`, not via `transformers.AutoModel*`.

Usage:
  python3 load_vllm_model_class.py --architecture LlamaForCausalLM
"""

import argparse

from vllm.model_executor.models import ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        type=str,
        default="LlamaForCausalLM",
        help="Architecture name from vLLM's model registry.",
    )
    args = parser.parse_args()

    if args.architecture not in ModelRegistry.models:
        raise ValueError(
            f"Architecture {args.architecture!r} is not in vLLM ModelRegistry."
        )

    model_cls = ModelRegistry.models[args.architecture].load_model_cls()

    print("requested_architecture:", args.architecture)
    print("loaded_model_class:", model_cls.__name__)
    print("loaded_from_module:", model_cls.__module__)


if __name__ == "__main__":
    main()
