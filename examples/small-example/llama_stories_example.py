from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot

# Constants
MODEL_ID = "Xenova/llama2.c-stories110M"
DATASET_ID = "open_platypus"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512  # Good starting point; increase for better accuracy
MAX_SEQUENCE_LENGTH = 2048

def load_model_and_tokenizer():
    """Initialize the pretrained model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",     # Automatically map model to available devices
        torch_dtype="auto",    # Use appropriate precision based on hardware
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer

def preprocess_dataset(tokenizer):
    """Load and preprocess the calibration dataset."""
    # Load and sample dataset
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

    # Apply chat template to messages
    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }
    
    # Tokenize the preprocessed text
    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(preprocess)
    ds = ds.map(tokenize, remove_columns=ds.column_names)
    return ds

def gptq_recipe():
    """
    Define the configuration for model optimization.
    - Targets Linear layers for quantization
    - Uses 4-bit weights with 16-bit activations (W4A16)
    - Groups weights in sets of 128 for better efficiency
    """
    return GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"]  # Skip language model head from quantization
    )

def apply_oneshot(model, recipe):
    """Apply the quantization process to the model using the oneshot method."""
    oneshot(
        model=model,
        dataset=DATASET_ID,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

def test_model_output(model, tokenizer):
    """Generate and print a sample output to verify model sanity."""
    print("\n\n========== SAMPLE GENERATION ==============")
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids, max_new_tokens=100)
    print(tokenizer.decode(output[0]))
    print("==========================================\n\n")

def save_compressed_model(model, tokenizer):
    """Save the quantized model and tokenizer to disk."""
    save_dir = MODEL_ID.split("/")[1] + "-W4A16-G128"
    # model.save_pretrained delegates to compressed_tensors for efficient storage
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)
    return save_dir

def main():
    """Main execution flow for model quantization and compression."""
    # Step 1: Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Step 2: Prepare dataset (commented out as oneshot handles it internally)
    # dataset = preprocess_dataset(tokenizer)
    
    # Step 3: Set up quantization configuration
    recipe = gptq_recipe()
    
    # Step 4: Apply quantization to the model
    apply_oneshot(model, recipe)
    
    # Step 5: Verify the quantized model works
    test_model_output(model, tokenizer)
    
    # Step 6: Save the compressed model
    save_dir = save_compressed_model(model, tokenizer)
    print(f"Model saved to: {save_dir}")

if __name__ == "__main__":
    main()