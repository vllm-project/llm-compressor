import torch
from loguru import logger
from transformers import AutoModelForCausalLM

from llmcompressor import oneshot, train

# load the model in as bfloat16 to save on memory and compute
model_stub = "neuralmagic/Llama-2-7b-ultrachat200k"
model = AutoModelForCausalLM.from_pretrained(
    model_stub, torch_dtype=torch.bfloat16, device_map="auto"
)

# uses LLM Compressor's built-in preprocessing for ultra chat
dataset = "ultrachat-200k"

# Select the recipe for 2 of 4 sparsity and 4-bit activation quantization
recipe = "2of4_w4a16_recipe.yaml"

# save location of quantized model
output_dir = "output_llama7b_2of4_w4a16_channel"

# set dataset config parameters
splits = {"calibration": "train_gen[:5%]", "train": "train_gen"}
max_seq_length = 512
num_calibration_samples = 512

# set training parameters for finetuning
num_train_epochs = 0.01
logging_steps = 500
save_steps = 5000
gradient_checkpointing = True  # saves memory during training
learning_rate = 0.0001
bf16 = False  # using full precision for training
lr_scheduler_type = "cosine"
warmup_ratio = 0.1
preprocessing_num_workers = 64


oneshot_kwargs = dict(
    dataset=dataset,
    recipe=recipe,
    num_calibration_samples=num_calibration_samples,
    preprocessing_num_workers=preprocessing_num_workers,
    splits=splits,
    output_dir=output_dir,
)

training_kwargs = dict(
    bf16=bf16,
    max_seq_length=max_seq_length,
    num_train_epochs=num_train_epochs,
    logging_steps=logging_steps,
    save_steps=save_steps,
    gradient_checkpointing=gradient_checkpointing,
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    warmup_ratio=warmup_ratio,
)

# This will run the targeted stage of the recipe
# oneshot sparsification -> finetuning -> oneshot quantization

# Models are automatically saved in
# ./output_llama7b_2of4_w4a16_channel/ + (finetuning/sparsity/quantization)_stage

# Oneshot sparsification
oneshot_applied_model = oneshot(
    model=model,
    **oneshot_kwargs,
    stage="sparsity_stage",
)

# Sparse finetune
finetune_applied_model = train(
    model=oneshot_applied_model,
    **oneshot_kwargs,
    **training_kwargs,
    stage="finetuning_stage",
)

# Oneshot quantization
model = oneshot(
    model=finetune_applied_model,
    **oneshot_kwargs,
    stage="quantization_stage",
)

logger.info(
    "llmcompressor does not currently support running ",
    "compressed models in the marlin24 format. "
    "The model produced from this example can be ",
    "run on vLLM with dtype=torch.float16.",
)
