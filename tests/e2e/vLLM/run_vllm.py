import json
import sys
import torch
from vllm import LLM, SamplingParams


opt_model = json.loads(sys.argv[1])
logger = json.loads(sys.argv[2])

sampling_params = SamplingParams(temperature=0.80, top_p=0.95)
llm_kwargs = {"model": opt_model.save_dir}

if "W4A16_2of4" in opt_model.scheme:
    # required by the kernel
    llm_kwargs["dtype"] = torch.float16

if opt_model.gpu_memory_utilization is not None:
    llm_kwargs["gpu_memory_utilization"] = opt_model.gpu_memory_utilization

llm = LLM(**llm_kwargs)
outputs = llm.generate(opt_model.prompts, sampling_params)


logger.info("================= vLLM GENERATION ======================")
for output in outputs:
    assert output
    prompt = output.prompt
    generated_text = output.outputs[0].text

    logger.info("PROMPT")
    logger.info(prompt)
    logger.info("GENERATED TEXT")
    logger.info(generated_text)
