from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from llmcompressor.transformers import oneshot

# model to quantize
model = "Isotonic/TinyMixtral-4x248M-MoE"

# set dataset config parameters
dataset = "open_platypus"
max_seq_length = 128
num_calibration_samples = 512

# set save location of quantized model
output_dir = "tinymixtral-4-248-moe_quantized_fp8"
save_compressed = True

# define a fp8 GPTQ quantization recipe
recipe = GPTQModifier(scheme="FP8", targets="Linear", ignore=["lm_head"])


oneshot(
    model=model,
    dataset=dataset,
    recipe=recipe,
    save_compressed=save_compressed,
    output_dir=output_dir,
    overwrite_output_dir=True,
    max_seq_length=max_seq_length,
    num_calibration_samples=num_calibration_samples,
)
