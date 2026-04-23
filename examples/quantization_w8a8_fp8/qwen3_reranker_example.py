from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3-Reranker-8B"

# Load model.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with per channel via ptq
#   * quantize the activations to fp8 with dynamic per token
# Note: FP8 Dynamic quantization does not require calibration data
recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

# Test the quantized model with a reranking example.
# Reranker models score query-document pairs
print("========== SAMPLE RERANKING TEST ==============")

query = "What is the capital of France?"
documents = [
    "Paris is the capital and most populous city of France.",
    "London is the capital of England and the United Kingdom.",
    "Berlin is the capital and largest city of Germany.",
]

# Format inputs for reranking
# The model expects query and document pairs. Process them in batch
pairs = [[query, doc] for doc in documents]
inputs = tokenizer(
    pairs,
    padding=True,
    truncation=True,
    return_tensors="pt",
    max_length=512,
).to(model.device)

dispatch_model(model)

# Get the model output
# The model processes the query-document pairs through transformer layers
outputs = model(**inputs)

# Extract relevance score from logits
# outputs.logits shape: [batch_size, sequence_length, vocab_size]
# [:, -1, :] means: all batches, last token, all vocabulary logits
# We use the maximum logit value as the relevance score
# Higher score indicates the document is more relevant to the query
scores = outputs.logits[:, -1, :].max(dim=-1).values

for i, (doc, score) in enumerate(zip(documents, scores)):
    print(f"Document {i+1} score: {score.item():.4f}")
    print(f"  Content: {doc[:80]}...")

print("==========================================")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
