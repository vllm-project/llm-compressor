"""
Distributed SmoothQuant runner for multi-GPU testing.

Usage:
torchrun --nproc_per_node=2 _smoothquant_ddp_runner.py \
         <model_id> <num_samples> <output_path>
"""

import sys

import torch
import torch.distributed as dist
from compressed_tensors.offload import init_dist, load_offloaded_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: torchrun --nproc_per_node=2 _smoothquant_ddp_runner.py \
                    <model_id> <num_samples> <output_path>"
        )
        sys.exit(1)

    model_id = sys.argv[1]
    num_samples = int(sys.argv[2])
    output_path = sys.argv[3]

    # Initialize distributed
    init_dist()

    # Load model with offloading
    with load_offloaded_model():
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32, device_map="auto_offload"
        )

    # Prepare dataset with rank partitioning
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.chat_template is None:
        tok.chat_template = (
            "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        )

    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split=get_rank_partition("train_sft", num_samples),
    )
    ds = ds.map(
        lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}
    )
    ds = ds.map(
        lambda s: tok(s["text"], padding=False, max_length=512, truncation=True),
        remove_columns=ds.column_names,
    )

    # Run SmoothQuant
    oneshot(
        model=model,
        dataset=ds,
        recipe=SmoothQuantModifier(smoothing_strength=0.8),
        num_calibration_samples=num_samples,
        max_seq_length=512,
    )

    # Save weights from rank 0
    if dist.get_rank() == 0:
        weights = {
            name: param.clone().cpu()
            for name, param in model.named_parameters()
            if "input_layernorm" in name or "q_proj" in name
        }
        torch.save(weights, output_path)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
