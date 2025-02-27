import argparse
from vllm import LLM, SamplingParams


def run_inference(model_path, tensor_parallel_size, prompt="Hello my name is:"):
    """
    Loads a model and performs inference using LLM.
    """
    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
    )
    # Load the model
    model = LLM(
        model=model_path,
        enforce_eager=True,
        dtype="auto",
        tensor_parallel_size=tensor_parallel_size,
    )

    # Generate inference
    outputs = model.generate(prompt, sampling_params=sampling_params)
    return outputs[0].outputs[0].text


def main():
    """Main function to handle CLI and process the model."""
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Run inference on a single model and print results."
    )
    parser.add_argument(
        "model_path", type=str, help="Path to the model to perform inference."
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=2,
        help="Tensor parallel size for the model. Default is 2.",
    )

    args = parser.parse_args()
    model_path = args.model_path
    tensor_parallel_size = args.tensor_parallel_size

    prompt = "Hello my name is:"

    # Load model and perform inference
    inference_result = run_inference(model_path, tensor_parallel_size)
    print("=" * 20)
    print("Model:", model_path)
    print(prompt, inference_result)


if __name__ == "__main__":
    main()
