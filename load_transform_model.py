from pathlib import Path

from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import CompressedTensorsConfig

MODEL_ID = "Llama-3.2-1B-Instruct-W4A16-uncompressed-hadamard-random-debug"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
    quantization_config=CompressedTensorsConfig(run_compressed=False),
)
breakpoint()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))

import lm_eval

results = lm_eval.simple_evaluate(
    model="hf",
    model_args={
        "pretrained": MODEL_ID,
        "add_bos_token": True,
        "quantization_config": CompressedTensorsConfig(run_compressed=False),
    },
    tasks=["gsm8k"],
    num_fewshot=8,
    limit=1000,
    device="cuda:0",
    batch_size=100,
)
print(results["results"])
"""
For: Llama-3.2-1B-Instruct

Dense:
{'gsm8k': {'alias': 'gsm8k', 'exact_match,strict-match': 0.379,
 'exact_match_stderr,strict-match': 0.015349091002225352, 
 'exact_match,flexible-extract': 0.381, 
 'exact_match_stderr,flexible-extract': 0.015364734787007436}}

----------------------------MINMAX ---------------------------:

QantModifier - NO TRANSFORMS 
{'gsm8k': {'alias': 'gsm8k', 'exact_match,strict-match': 0.177, 
'exact_match_stderr,strict-match': 0.011743632866916145, 
'exact_match,flexible-extract': 0.179, 
'exact_match_stderr,flexible-extract': 0.0117721103708122}}

QuantModifier - TRANSFORMS (random)
{'gsm8k': {'alias': 'gsm8k', 'exact_match,strict-match': 0.231, 
'exact_match_stderr,strict-match': 0.012997843819031815, 
'exact_match,flexible-extract': 0.236, 
'exact_match_stderr,flexible-extract': 0.01301973553930782}}

GPTQ
{'gsm8k': {'alias': 'gsm8k', 'exact_match,strict-match': 0.243, 
'exact_match_stderr,strict-match': 0.013569640199177434, 
'exact_match,flexible-extract': 0.244, 
'exact_match_stderr,flexible-extract': 0.013588548437881431}}


---------------------------MSE-----------------------------------:
QuantModifier - No Transforms
{'gsm8k': {'alias': 'gsm8k', 'exact_match,strict-match': 0.195,
'exact_match_stderr,strict-match': 0.012535235623319334, 
'exact_match,flexible-extract': 0.195,
 'exact_match_stderr,flexible-extract': 0.012535235623319334}}

QuantModifier - With Transforms (random)
{'gsm8k': {'alias': 'gsm8k', 'exact_match,strict-match': 0.243, 
'exact_match_stderr,strict-match': 0.013569640199177457, 
'exact_match,flexible-extract': 0.244,
 'exact_match_stderr,flexible-extract': 0.013588548437881412}}

QuantModifier - With Transforms (not random, not normalized )
{'gsm8k': {'alias': 'gsm8k', 'exact_match,strict-match': 0.261, 
'exact_match_stderr,strict-match': 0.013895037677965126, 
'exact_match,flexible-extract': 0.262, 
'exact_match_stderr,flexible-extract': 0.013912208651021352}}

QuantModifier - With Transforms (not random, normalized)
{'gsm8k': {'alias': 'gsm8k', 
'exact_match,strict-match': 0.27, 
'exact_match_stderr,strict-match': 0.014046255632633915, 
'exact_match,flexible-extract': 0.27,
 'exact_match_stderr,flexible-extract': 0.014046255632633915}}

GPTQ:
{'gsm8k': {'alias': 'gsm8k', 'exact_match,strict-match': 0.285, 
'exact_match_stderr,strict-match': 0.014282120955200484, 
'exact_match,flexible-extract': 0.286, 
'exact_match_stderr,flexible-extract': 0.01429714686251791}}

---------------------8bit----------------------------------:
QuantModifier - with Transforms (not random, normalized)
{'gsm8k': {'alias': 'gsm8k', 'exact_match,strict-match': 0.371, 
'exact_match_stderr,strict-match': 0.015283736211823187,
'exact_match,flexible-extract': 0.372,
'exact_match_stderr,flexible-extract': 0.015292149942040577}}

GPTQ
{'gsm8k': {'alias': 'gsm8k', 'exact_match,strict-match': 0.364, 
'exact_match_stderr,strict-match': 0.01522286884052202,
 'exact_match,flexible-extract': 0.365,
  'exact_match_stderr,flexible-extract': 0.015231776226264903}}
"""
