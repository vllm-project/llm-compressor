import json
import sys
import torch
from vllm import LLM, SamplingParams


llm_kwargs = json.loads(sys.argv[1])
prompts = json.loads(sys.argv[2])
logger = json.loads(sys.argv[3])

sampling_params = SamplingParams(temperature=0.80, top_p=0.95)

llm = LLM(**llm_kwargs)
outputs = llm.generate(prompts, sampling_params)


logger.info("================= vLLM GENERATION ======================")
for output in outputs:
    assert output
    prompt = output.prompt
    generated_text = output.outputs[0].text

    logger.info("PROMPT")
    logger.info(prompt)
    logger.info("GENERATED TEXT")
    logger.info(generated_text)
