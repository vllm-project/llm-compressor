import json
import sys

from vllm import LLM, SamplingParams


def parse_args():
    """Parse JSON arguments passed via command line."""
    if len(sys.argv) < 5:
        msg = (
            "Usage: python script.py '<scheme>' '<llm_kwargs_json>' "
            "'<prompts_json>' '<sampling_params_json>'"
        )
        raise ValueError(msg)

    try:
        llm_kwargs = json.loads(sys.argv[2])
        prompts = json.loads(sys.argv[3])
        sampling_params = json.loads(sys.argv[4])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON input: {e}")

    return llm_kwargs, prompts, sampling_params


def run_vllm(llm_kwargs: dict, prompts: list[str], sampling_params: dict) -> None:
    """Run vLLM with given kwargs and prompts, then print outputs.

    Args:
        llm_kwargs: Keyword arguments for LLM initialization
        prompts: List of prompts to generate from
        sampling_params: vLLM sampling parameters (temperature, top_p, top_k, etc.)
                        Defaults to {'temperature': 0.0} for deterministic output
                        See: https://docs.vllm.ai/en/v0.4.1/dev/sampling_params.html
    """
    vllm_sampling_params = SamplingParams(**sampling_params)

    llm = LLM(**llm_kwargs)
    outputs = llm.generate(prompts, vllm_sampling_params)

    print("================= vLLM GENERATION =================")
    for output in outputs:
        if not output or not output.outputs:
            print("[Warning] Empty output for prompt:", output.prompt)
            continue

        print(f"\nPROMPT:\n{output.prompt}")
        print(f"GENERATED TEXT:\n{output.outputs[0].text}")


def main():
    llm_kwargs, prompts, sampling_params = parse_args()
    run_vllm(llm_kwargs, prompts, sampling_params)


if __name__ == "__main__":
    main()
