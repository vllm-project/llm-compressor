import gc
import json
import sys
import torch
from vllm import LLM, SamplingParams


def main():
    sampling_params = SamplingParams(temperature=0.80, top_p=0.95)

    llm = LLM(**llm_kwargs)
    outputs = llm.generate(prompts, sampling_params)

    json_outputs = {}
    for output in outputs:
        assert output
        prompt = output.prompt
        generated_text = output.outputs[0].text
        json_outputs[prompt] = generated_text

    del llm
    gc.collect()

    print("VLLM OUTPUT:"+str(json_outputs))

if __name__ == "__main__":
    llm_kwargs = json.loads(sys.argv[1])
    prompts = json.loads(sys.argv[2])

    main()
