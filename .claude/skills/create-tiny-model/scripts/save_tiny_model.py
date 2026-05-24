from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.utils.dev import skip_weights_download

model_id = "Qwen/Qwen3-30B-A3B"

with skip_weights_download(AutoModelForCausalLM):
    model = AutoModelForCausalLM.from_pretrained(model_id, num_hidden_layers=1, hidden_size=1024)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

num_parameters = model.num_parameters()
num_parameters_b = num_parameters / 1e9
assert num_parameters_b <= 1.0  # ideally the model is ~1b parameters. It's okay to use more parameters if they are required by the model architecture

save_path = f"Qwen3-{num_parameters_b:.1f}B-{num_parameters_b/10:.1f}B"  # for moe models, an estimate of the number of activated parameters should be included in the name
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
