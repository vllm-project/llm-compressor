import json
import sys
from vllm import LLM, SamplingParams


def parse_args():
    """Parse JSON arguments passed via command line."""
    if len(sys.argv) < 3:
        raise ValueError("Usage: python script.py '<llm_kwargs_json>' '<prompts_json>'")

    try:
        llm_kwargs = json.loads(sys.argv[1])
        prompts = json.loads(sys.argv[2])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON input: {e}")

    return llm_kwargs, prompts


def run_vllm(llm_kwargs: dict, prompts: list[str]) -> None:
    """Run vLLM with given kwargs and prompts, then print outputs."""
    sampling_params = SamplingParams(temperature=0.80, top_p=0.95)

    llm = LLM(**llm_kwargs)
    outputs = llm.generate(prompts, sampling_params)

    print("================= vLLM GENERATION =================")
    for output in outputs:
        if not output or not output.outputs:
            print("[Warning] Empty output for prompt:", output.prompt)
            continue

        print(f"\nPROMPT:\n{output.prompt}")
        print(f"GENERATED TEXT:\n{output.outputs[0].text}")


def main():
    llm_kwargs, prompts = parse_args()
    run_vllm(llm_kwargs, prompts)


if __name__ == "__main__":
    main()
