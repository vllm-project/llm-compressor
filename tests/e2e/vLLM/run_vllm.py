import json
import sys
import torch
from vllm import LLM, SamplingParams


llm_kwargs = json.loads(sys.argv[1])
prompts = json.loads(sys.argv[2])

sampling_params = SamplingParams(temperature=0.80, top_p=0.95)

llm = LLM(**llm_kwargs)
outputs = llm.generate(prompts, sampling_params)
json_outputs = json.dumps(outputs)

return json_outputs
