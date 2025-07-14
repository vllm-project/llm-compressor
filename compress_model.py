# python3 compress_model.py --model_id meta-llama/Llama-3.2-1B-Instruct --transform_type random-hadamard
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import SpinQuantModifier
from llmcompressor.utils import dispatch_for_generation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, help="Model stub to compress")
    parser.add_argument("--transform_type", type=str, default=None, help="Type of transform used in SpinQuantModifier")
    parser.add_argument("--scheme", type=str, default=None, help="Quantization scheme (e.g. W4A16)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Select model and load it.
    MODEL_ID = args.model_id
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Select number of samples. 512 samples is a good place to start.
    # Increasing the number of samples can improve accuracy.
    NUM_CALIBRATION_SAMPLES = 512
    MAX_SEQUENCE_LENGTH = 2048

    # Configure the quantization algorithm to run.
    recipe = []
    if args.transform_type:
        recipe.append(SpinQuantModifier(rotations=["R1", "R2"], transform_type=args.transform_type))

    if args.scheme:
        recipe.append(QuantizationModifier(targets="Linear", scheme=args.scheme, ignore=["lm_head"]))

    # Apply algorithms.
    oneshot(
        model=model,
        recipe=recipe,
        dataset="ultrachat_200k",
        splits={"calibration": f"train_sft[:{NUM_CALIBRATION_SAMPLES}]"},
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    # Confirm generations of the quantized model look sane.
    print("\n\n")
    print("========== SAMPLE GENERATION ==============")
    dispatch_for_generation(model)
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids, max_new_tokens=100)
    print(tokenizer.decode(output[0]))
    print("==========================================\n\n")

    # Save to disk compressed.
    SAVE_DIR = MODEL_ID.split("/")[1] + f"-{args.transform_type}-{args.scheme}"
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
