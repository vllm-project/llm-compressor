import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modeling.moe.linearize import linearize_moe_model

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V4-Flash",
    torch_dtype="auto",
    device_map="cpu",
)
delattr(model, "_weight_conversions")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V4-Flash")

save_dir = "DeepSeek-V4-Flash-bf16"
#model.dequantize(torch.bfloat16)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
