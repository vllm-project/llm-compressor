import json
import sys
from vllm import LLM, SamplingParams


llm_kwargs = json.loads(sys.argv[1])
prompts = json.loads(sys.argv[2])

def main():
    sampling_params = SamplingParams(temperature=0.80, top_p=0.95)

    llm = LLM(**llm_kwargs)
    outputs = llm.generate(prompts, sampling_params)

    print("================= vLLM GENERATION =================")
    for output in outputs:
        assert output
        print("PROMPT")
        print(output.prompt)
        print("GENERATED TEXT")
        print(output.outputs[0].text)

if __name__ == "__main__":
    main()
