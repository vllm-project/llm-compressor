"""
Extract hidden states from a model using vLLM's hidden state extraction API.

This script runs a model through vLLM and extracts post-norm hidden states
(at layer index num_hidden_layers, i.e. after the final layer norm) and saves
them as safetensors files. These hidden states can be passed directly to lm_head
by compute_kl.py to efficiently calculate KL divergence without full-vocab logprobs.

Requires vllm >= 0.18.0.

Usage:
    python tools/kl_divergence/extract_hidden_states.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --output-dir ./hidden_states/base \
        --dataset Salesforce/wikitext \
        --dataset-config wikitext-2-raw-v1 \
        --split test \
        --max-seq-length 2048 \
        --num-samples 128
"""

import argparse
import json
import os
import tempfile

import torch
from datasets import load_dataset
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract hidden states from a model using vLLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID or path to local model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save extracted hidden states",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Salesforce/wikitext",
        help="HuggingFace dataset name (default: Salesforce/wikitext)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset configuration name (default: wikitext-2-raw-v1)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length for chunking (default: 2048)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Max number of sequence chunks to process (default: all)",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=None,
        help="Layer index to extract hidden states from "
        "(default: num_hidden_layers, i.e. post-norm output)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM (default: 0.9)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM (default: 1)",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column name containing text in the dataset (default: text)",
    )
    return parser.parse_args()


def prepare_token_chunks(
    model_id: str,
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_seq_length: int = 2048,
    num_samples: int = None,
    text_column: str = "text",
) -> list[list[int]]:
    """
    Tokenize a dataset, concatenate all tokens, and split into fixed-length
    chunks suitable for vLLM prompt input.

    :param model_id: model ID for loading the tokenizer
    :param dataset_name: HuggingFace dataset name
    :param dataset_config: dataset configuration name
    :param split: dataset split
    :param max_seq_length: chunk size in tokens
    :param num_samples: max number of chunks to return
    :param text_column: column containing text data
    :return: list of token ID lists, each of length max_seq_length
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    required_tokens = None if num_samples is None else num_samples * max_seq_length

    all_tokens = []

    if required_tokens is None:
        # Full-dataset path: keep batched tokenization for throughput.
        dataset = dataset.filter(lambda x: x[text_column] and x[text_column].strip())
        tokenized = dataset.map(
            lambda batch: {
                "input_ids": tokenizer(
                    batch[text_column],
                    add_special_tokens=False,
                )["input_ids"]
            },
            batched=True,
            remove_columns=dataset.column_names,
        )
        for row in tokenized:
            all_tokens.extend(row["input_ids"])
    else:
        # Capped path: stop tokenizing once enough tokens are available.
        for row in dataset:
            text = row[text_column]
            if not text or not text.strip():
                continue
            all_tokens.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
            if len(all_tokens) >= required_tokens:
                all_tokens = all_tokens[:required_tokens]
                break

    # Chunk into fixed-length sequences (drop the last incomplete chunk)
    total_length = (len(all_tokens) // max_seq_length) * max_seq_length
    chunks = [
        all_tokens[i : i + max_seq_length]
        for i in range(0, total_length, max_seq_length)
    ]

    print(f"Total tokens: {len(all_tokens)}")
    print(f"Sequence chunks: {len(chunks)} x {max_seq_length} tokens")

    return chunks


def extract_hidden_states(
    model_id: str,
    output_dir: str,
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_seq_length: int = 2048,
    num_samples: int = None,
    layer_index: int = None,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    text_column: str = "text",
) -> str:
    """
    Extract hidden states from a model using vLLM and save to disk.

    :param model_id: HuggingFace model ID or local path
    :param output_dir: directory to save hidden states
    :param dataset_name: HuggingFace dataset name
    :param dataset_config: dataset configuration name
    :param split: dataset split
    :param max_seq_length: sequence length for chunking
    :param num_samples: max number of chunks to process
    :param layer_index: which layer to extract (default: last)
    :param gpu_memory_utilization: vLLM GPU memory fraction
    :param tensor_parallel_size: vLLM tensor parallel size
    :param text_column: column containing text data
    :return: path to output directory
    """
    try:
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
    except ImportError:
        raise ImportError(
            "vllm >= 0.18.0 is required for hidden state extraction. "
            "Install with: pip install vllm>=0.18.0"
        )

    # Resolve relative paths so AutoConfig and vLLM can find local models
    if os.path.exists(model_id):
        model_id = os.path.abspath(model_id)

    # Auto-detect layer index for post-norm hidden state extraction
    if layer_index is None:
        config = AutoConfig.from_pretrained(model_id)
        config = getattr(config, "text_config", config)
        layer_index = config.num_hidden_layers
        print(f"Auto-detected layer index (post-norm): {layer_index}")

    # Prepare token chunks
    chunks = prepare_token_chunks(
        model_id=model_id,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        max_seq_length=max_seq_length,
        num_samples=num_samples,
        text_column=text_column,
    )

    if len(chunks) == 0:
        raise ValueError("No token chunks produced from dataset")

    os.makedirs(output_dir, exist_ok=True)

    # Use a temp dir for vLLM's connector output, then reorganize
    with tempfile.TemporaryDirectory() as vllm_output_dir:
        # Initialize vLLM with hidden state extraction
        print(f"Initializing vLLM for model: {model_id}")
        print(f"Extracting hidden states from layer {layer_index}")

        llm = LLM(
            model=model_id,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            speculative_config={
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {
                        "eagle_aux_hidden_state_layer_ids": [layer_index],
                    }
                },
            },
            kv_transfer_config={
                "kv_connector": "ExampleHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "shared_storage_path": vllm_output_dir,
                },
            },
        )

        sampling_params = SamplingParams(max_tokens=1)
        prompts = [TokensPrompt(prompt_token_ids=chunk) for chunk in chunks]

        print(f"Running extraction on {len(prompts)} sequences...")
        outputs = llm.generate(prompts, sampling_params)

        # Collect and reorganize hidden state files
        hidden_state_files = []
        for idx, output in enumerate(tqdm(outputs, desc="Saving hidden states")):
            kv_params = output.kv_transfer_params
            if kv_params is None or "hidden_states_path" not in kv_params:
                print(f"Warning: No hidden states for sequence {idx}, skipping")
                continue

            src_path = kv_params["hidden_states_path"]

            # Load and re-save with consistent naming
            with safe_open(src_path, framework="pt") as f:
                tensors = {key: f.get_tensor(key) for key in f.keys()}

            dst_filename = f"hidden_states_{idx:06d}.safetensors"
            dst_path = os.path.join(output_dir, dst_filename)
            save_file(tensors, dst_path)
            hidden_state_files.append(dst_filename)

    # Save metadata
    metadata = {
        "model_id": model_id,
        "layer_index": layer_index,
        "max_seq_length": max_seq_length,
        "num_samples": len(hidden_state_files),
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "files": hidden_state_files,
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nExtraction complete:")
    print(f"  Saved {len(hidden_state_files)} hidden state files to {output_dir}")
    print(f"  Metadata saved to {metadata_path}")

    # Clean up vLLM resources
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_dir


def main():
    args = parse_args()
    extract_hidden_states(
        model_id=args.model,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        max_seq_length=args.max_seq_length,
        num_samples=args.num_samples,
        layer_index=args.layer_index,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        text_column=args.text_column,
    )


if __name__ == "__main__":
    main()
