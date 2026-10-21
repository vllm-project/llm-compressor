from compressed_tensors.quantization.quant_scheme import (
    MXFP4,
    MXFP8,
    QuantizationScheme,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import model_free_ptq
from llmcompressor.utils import load_context

MODEL_ID = "inference-optimization/GLM-5.2-0.8B-A0.8B"  # zai-org/GLM-5.2
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP4xMXFP8"

# use `model_free_ptq` to apply quantization
# NOTE: MXFP4xMXFP8 is experimentally supported in vllm
model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme=QuantizationScheme(
        targets=["Linear"],
        weights=MXFP4["weights"],
        input_activations=MXFP8["input_activations"],
    ),
    ignore=[
        r"model.embed_tokens",
        r"re:^model\.layers\.[0-2]\..*",
        r"re:.*self_attn.*",
        r"re:.*mlp\.gate.*",
        r"lm_head",
    ],
    max_workers=50,
)

# sample generation: skip for large models
model_id = SAVE_DIR
with load_context():
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

input_ids = tokenizer("According to all known laws", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
