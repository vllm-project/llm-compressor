import json
import sys

from vllm import LLM, SamplingParams


def parse_args():
    """Parse JSON arguments passed via command line."""
    if len(sys.argv) < 4:
        msg = "Usage: python script.py '<scheme>' '<llm_kwargs_json>' '<prompts_json>'"
        raise ValueError(msg)

    try:
        llm_kwargs = json.loads(sys.argv[2])
        prompts = json.loads(sys.argv[3])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON input: {e}")

    return llm_kwargs, prompts


def run_vllm(llm_kwargs: dict, prompts: list[str]) -> None:
    """Run vLLM with given kwargs and prompts, then print outputs."""
    # Use greedy decoding for deterministic output
    sampling_params = SamplingParams(temperature=0.0)

    llm = LLM(**llm_kwargs)
    outputs = llm.generate(prompts, sampling_params)

    if llm_kwargs.get("speculative_config", {}).get("method") == "mtp":
        assert len(outputs) == len(prompts)
        assert all(output.outputs and output.outputs[0].token_ids for output in outputs)
        assert any(
            metric.name == "vllm:spec_decode_num_draft_tokens" and metric.value > 0
            for metric in llm.get_metrics()
        ), "MTP did not draft any tokens"

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
