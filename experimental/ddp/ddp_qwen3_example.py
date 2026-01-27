"""
python ddp_qwen3_example.py --ddp --nsamples 128 --iters 100

"""

from auto_round.calib_dataset import get_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier
from llmcompressor.utils import dispatch_for_generation

# Select model and load it.
model_id = "Qwen/Qwen3-235B-A22B"
model_id = "Qwen/Qwen3-8B"
# model_id = "/data5/yiliu4/Qwen/Qwen2-0.5B"
# model_id = "Qwen/Qwen2-0.5B"

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup_ddp(rank, world_size):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12356")

    # Initialize process group
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def quantize_model(rank, world_size, model_name, scheme, iters=4, nsamples=32):
    """
    Quantize model on a specific GPU rank.

    Args:
        rank: GPU rank for this process
        world_size: Total number of GPUs
        model_name: Model name or path
        scheme: Quantization scheme
        iters: Number of iterations
        nsamples: Number of samples
    """
    print(f"[Rank {rank}/{world_size}] Starting quantization")

    # Setup DDP if using multiple GPUs
    if world_size > 1:
        setup_ddp(rank, world_size)

    # Set device for this process
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Select calibration dataset.
    NUM_CALIBRATION_SAMPLES = nsamples
    MAX_SEQUENCE_LENGTH = 2048
    ITERS = iters
    # Get aligned calibration dataset.

    ds = get_dataset(
        tokenizer=tokenizer,
        seqlen=MAX_SEQUENCE_LENGTH,
        nsamples=NUM_CALIBRATION_SAMPLES,
    )

    # Configure the quantization algorithm to run.
    #   * quantize the weights to 4 bit with AutoRound with a group size 128
    #   * For `Qwen/Qwen3-235B-A22B`, it requires about 300 GB memory
    #     to run tuning with default settings.
    recipe = AutoRoundModifier(
        targets="Linear",
        scheme=scheme,
        ignore=[
            "lm_head",
            "re:.*mlp.gate$",
        ],
        iters=ITERS,
        enable_torch_compile=False,
        # device_ids="0,1,2,3",  # Use 4 A100 GPUs
    )

    # Apply algorithms.
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        shuffle_calibration_samples=False,
    )

    # Synchronize all processes
    if world_size > 1:
        dist.barrier()

    print(f"[Rank {rank}] Quantization completed")
    if rank == 0:
        # Confirm generations of the quantized model look sane.
        print("\n\n")
        print("========== SAMPLE GENERATION ==============")
        dispatch_for_generation(model)
        sample = tokenizer("Hello my name is", return_tensors="pt")
        sample = {key: value.to(model.device) for key, value in sample.items()}
        output = model.generate(**sample, max_new_tokens=100)
        print(tokenizer.decode(output[0]))
        print("==========================================\n\n")

        # Save to disk compressed.
        SAVE_DIR = model_name.rstrip("/").split("/")[-1] + f"-{scheme}-AutoRound" + f"-iters{iters}-nsamples{nsamples}"
        print(f"save to {SAVE_DIR}")
        model.save_pretrained(SAVE_DIR, save_compressed=True)
        tokenizer.save_pretrained(SAVE_DIR)
    else:
        # Other ranks just run quantization without saving
        print(f"[Rank {rank}] Running quantization (not saving)")

    # except Exception as e:
    #     print(f"[Rank {rank}] Error during quantization: {e}")
    #     raise

    # finally:
    #     # Cleanup DDP
    if world_size > 1:
        cleanup_ddp()


def main_spawn(model_name, scheme, iters, nsamples):
    """Main function using mp.spawn for multi-GPU quantization."""
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # if world_size < 2:
    #     print("Warning: Only 1 GPU detected. Running single GPU mode.")
    #     return main_single_gpu(model_name, scheme, iters, nsamples)
    print(f"Starting DDP quantization with {world_size} GPUs")

    mp.spawn(
        quantize_model,
        args=(world_size, model_name, scheme, iters, nsamples),
        nprocs=world_size,
        join=True,
    )

    print("Quantization completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoRound Quantization with DDP support"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=model_id,
        help="Model name or path",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default="W4A16",
        help="Quantization scheme (W4A16, MXFP8, MXFP4, etc.)",
    )
    parser.add_argument("--iters", type=int, default=4, help="Number of iterations")
    parser.add_argument("--nsamples", type=int, default=32, help="Number of samples")
    parser.add_argument("--ddp", action="store_true", help="Enable DDP multi-GPU mode")

    args = parser.parse_args()

    # For backward compatibility with existing hardcoded values
    model_name = args.model_name

    # Parse scheme from string if needed
    from auto_round import schemes as ar_schemes

    scheme_map = {
        "FP8_STATIC": ar_schemes.FP8_STATIC,
        "MXFP8": ar_schemes.MXFP8,
        "MXFP4": ar_schemes.MXFP4,
    }
    # scheme = scheme_map.get(args.scheme, args.scheme)

    # # Check if running with torchrun
    # if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    #     print("Detected torchrun environment")
    #     main_torchrun(model_name, scheme, args.iters, args.nsamples)
    if args.ddp:
        print("Using mp.spawn mode for multi-GPU quantization")
        main_spawn(model_name, args.scheme, args.iters, args.nsamples)


"""
vllm (pretrained=/home/yiliu7/workspace/llm-compressor/examples/Qwen3-8B-W4A16-G128-AutoRound,tensor_parallel_size=1,max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False), gen_kwargs: (None), limit: 1000.0, num_fewshot: None, batch_size: 128
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.897|±  |0.0096|
|     |       |strict-match    |     5|exact_match|↑  |0.897|±  |0.0096|

vllm (pretrained=/home/yiliu7/workspace/llm-compressor/examples/Qwen3-8B-W4A16-G128-AutoRound-iters200-nsamples128,tensor_parallel_size=1,max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False), gen_kwargs: (None), limit: 1000.0, num_fewshot: None, batch_size: 128
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.904|±  |0.0093|
|     |       |strict-match    |     5|exact_match|↑  |0.904|±  |0.0093|

vllm (pretrained=/home/yiliu7/workspace/llm-compressor/examples/Qwen3-8B-W4A16-G128-AutoRound-iters100-nsamples256,tensor_parallel_size=1,max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False), gen_kwargs: (None), limit: 1000.0, num_fewshot: None, batch_size: 128
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |  0.9|±  |0.0095|
|     |       |strict-match    |     5|exact_match|↑  |  0.9|±  |0.0095|

vllm (pretrained=/home/yiliu7/workspace/llm-compressor/examples/Qwen3-8B-W4A16-G128-AutoRound-iters100-nsamples256-index_sampler,tensor_parallel_size=1,max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False), gen_kwargs: (None), limit: 1000.0, num_fewshot: None, batch_size: 128
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.896|±  |0.0097|
|     |       |strict-match    |     5|exact_match|↑  |0.896|±  |0.0097|


vllm (pretrained=/home/yiliu7/workspace/llm-compressor/examples/Qwen3-8B-W4A16-G128-AutoRound-iters100-nsamples256-index_sampler,tensor_parallel_size=1,max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False), gen_kwargs: (None), limit: 1000.0, num_fewshot: None, batch_size: 128
|                 Tasks                 |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|------|---|-----:|---|-----:|
|mmlu                                   |      2|none  |      |acc   |↑  |0.7306|±  |0.0036|
| - humanities                          |      2|none  |      |acc   |↑  |0.6416|±  |0.0069|
|  - formal_logic                       |      1|none  |     0|acc   |↑  |0.6825|±  |0.0416|
|  - high_school_european_history       |      1|none  |     0|acc   |↑  |0.8485|±  |0.0280|
|  - high_school_us_history             |      1|none  |     0|acc   |↑  |0.8480|±  |0.0252|
|  - high_school_world_history          |      1|none  |     0|acc   |↑  |0.8819|±  |0.0210|
|  - international_law                  |      1|none  |     0|acc   |↑  |0.7769|±  |0.0380|
|  - jurisprudence                      |      1|none  |     0|acc   |↑  |0.7963|±  |0.0389|
|  - logical_fallacies                  |      1|none  |     0|acc   |↑  |0.8405|±  |0.0288|
|  - moral_disputes                     |      1|none  |     0|acc   |↑  |0.7543|±  |0.0232|
|  - moral_scenarios                    |      1|none  |     0|acc   |↑  |0.3966|±  |0.0164|
|  - philosophy                         |      1|none  |     0|acc   |↑  |0.7460|±  |0.0247|
|  - prehistory                         |      1|none  |     0|acc   |↑  |0.8241|±  |0.0212|
|  - professional_law                   |      1|none  |     0|acc   |↑  |0.4890|±  |0.0158|
|  - world_religions                    |      1|none  |     0|acc   |↑  |0.8596|±  |0.0266|
| - other                               |      2|none  |      |acc   |↑  |0.7634|±  |0.0074|
|  - business_ethics                    |      1|none  |     0|acc   |↑  |0.7400|±  |0.0441|
|  - clinical_knowledge                 |      1|none  |     0|acc   |↑  |0.7698|±  |0.0259|
|  - college_medicine                   |      1|none  |     0|acc   |↑  |0.7514|±  |0.0330|
|  - global_facts                       |      1|none  |     0|acc   |↑  |0.4300|±  |0.0498|
|  - human_aging                        |      1|none  |     0|acc   |↑  |0.7265|±  |0.0299|
|  - management                         |      1|none  |     0|acc   |↑  |0.8835|±  |0.0318|
|  - marketing                          |      1|none  |     0|acc   |↑  |0.9231|±  |0.0175|
|  - medical_genetics                   |      1|none  |     0|acc   |↑  |0.8100|±  |0.0394|
|  - miscellaneous                      |      1|none  |     0|acc   |↑  |0.8429|±  |0.0130|
|  - nutrition                          |      1|none  |     0|acc   |↑  |0.7712|±  |0.0241|
|  - professional_accounting            |      1|none  |     0|acc   |↑  |0.5745|±  |0.0295|
|  - professional_medicine              |      1|none  |     0|acc   |↑  |0.8051|±  |0.0241|
|  - virology                           |      1|none  |     0|acc   |↑  |0.5663|±  |0.0386|
| - social sciences                     |      2|none  |      |acc   |↑  |0.8268|±  |0.0067|
|  - econometrics                       |      1|none  |     0|acc   |↑  |0.6754|±  |0.0440|
|  - high_school_geography              |      1|none  |     0|acc   |↑  |0.8535|±  |0.0252|
|  - high_school_government_and_politics|      1|none  |     0|acc   |↑  |0.9275|±  |0.0187|
|  - high_school_macroeconomics         |      1|none  |     0|acc   |↑  |0.7923|±  |0.0206|
|  - high_school_microeconomics         |      1|none  |     0|acc   |↑  |0.8950|±  |0.0199|
|  - high_school_psychology             |      1|none  |     0|acc   |↑  |0.9046|±  |0.0126|
|  - human_sexuality                    |      1|none  |     0|acc   |↑  |0.8397|±  |0.0322|
|  - professional_psychology            |      1|none  |     0|acc   |↑  |0.7598|±  |0.0173|
|  - public_relations                   |      1|none  |     0|acc   |↑  |0.7091|±  |0.0435|
|  - security_studies                   |      1|none  |     0|acc   |↑  |0.7837|±  |0.0264|
|  - sociology                          |      1|none  |     0|acc   |↑  |0.8657|±  |0.0241|
|  - us_foreign_policy                  |      1|none  |     0|acc   |↑  |0.8500|±  |0.0359|
| - stem                                |      2|none  |      |acc   |↑  |0.7222|±  |0.0077|
|  - abstract_algebra                   |      1|none  |     0|acc   |↑  |0.5700|±  |0.0498|
|  - anatomy                            |      1|none  |     0|acc   |↑  |0.6889|±  |0.0400|
|  - astronomy                          |      1|none  |     0|acc   |↑  |0.8882|±  |0.0256|
|  - college_biology                    |      1|none  |     0|acc   |↑  |0.8681|±  |0.0283|
|  - college_chemistry                  |      1|none  |     0|acc   |↑  |0.5800|±  |0.0496|
|  - college_computer_science           |      1|none  |     0|acc   |↑  |0.7000|±  |0.0461|
|  - college_mathematics                |      1|none  |     0|acc   |↑  |0.5600|±  |0.0499|
|  - college_physics                    |      1|none  |     0|acc   |↑  |0.5490|±  |0.0495|
|  - computer_security                  |      1|none  |     0|acc   |↑  |0.8500|±  |0.0359|
|  - conceptual_physics                 |      1|none  |     0|acc   |↑  |0.8085|±  |0.0257|
|  - electrical_engineering             |      1|none  |     0|acc   |↑  |0.7724|±  |0.0349|
|  - elementary_mathematics             |      1|none  |     0|acc   |↑  |0.7143|±  |0.0233|
|  - high_school_biology                |      1|none  |     0|acc   |↑  |0.8968|±  |0.0173|
|  - high_school_chemistry              |      1|none  |     0|acc   |↑  |0.6995|±  |0.0323|
|  - high_school_computer_science       |      1|none  |     0|acc   |↑  |0.8600|±  |0.0349|
|  - high_school_mathematics            |      1|none  |     0|acc   |↑  |0.5111|±  |0.0305|
|  - high_school_physics                |      1|none  |     0|acc   |↑  |0.6623|±  |0.0386|
|  - high_school_statistics             |      1|none  |     0|acc   |↑  |0.7407|±  |0.0299|
|  - machine_learning                   |      1|none  |     0|acc   |↑  |0.5893|±  |0.0467|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.7306|±  |0.0036|
| - humanities     |      2|none  |      |acc   |↑  |0.6416|±  |0.0069|
| - other          |      2|none  |      |acc   |↑  |0.7634|±  |0.0074|
| - social sciences|      2|none  |      |acc   |↑  |0.8268|±  |0.0067|
| - stem           |      2|none  |      |acc   |↑  |0.7222|±  |0.0077|

vllm (pretrained=/home/yiliu7/workspace/llm-compressor/examples/Qwen3-8B-W4A16-G128-AutoRound-iters200-nsamples128,tensor_parallel_size=1,max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False), gen_kwargs: (None), limit: 1000.0, num_fewshot: None, batch_size: 128
|                 Tasks                 |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|------|---|-----:|---|-----:|
|mmlu                                   |      2|none  |      |acc   |↑  |0.7279|±  |0.0036|
| - humanities                          |      2|none  |      |acc   |↑  |0.6413|±  |0.0068|
|  - formal_logic                       |      1|none  |     0|acc   |↑  |0.6667|±  |0.0422|
|  - high_school_european_history       |      1|none  |     0|acc   |↑  |0.8727|±  |0.0260|
|  - high_school_us_history             |      1|none  |     0|acc   |↑  |0.8824|±  |0.0226|
|  - high_school_world_history          |      1|none  |     0|acc   |↑  |0.8776|±  |0.0213|
|  - international_law                  |      1|none  |     0|acc   |↑  |0.8264|±  |0.0346|
|  - jurisprudence                      |      1|none  |     0|acc   |↑  |0.7500|±  |0.0419|
|  - logical_fallacies                  |      1|none  |     0|acc   |↑  |0.8282|±  |0.0296|
|  - moral_disputes                     |      1|none  |     0|acc   |↑  |0.7514|±  |0.0233|
|  - moral_scenarios                    |      1|none  |     0|acc   |↑  |0.3866|±  |0.0163|
|  - philosophy                         |      1|none  |     0|acc   |↑  |0.7556|±  |0.0244|
|  - prehistory                         |      1|none  |     0|acc   |↑  |0.8086|±  |0.0219|
|  - professional_law                   |      1|none  |     0|acc   |↑  |0.4980|±  |0.0158|
|  - world_religions                    |      1|none  |     0|acc   |↑  |0.8304|±  |0.0288|
| - other                               |      2|none  |      |acc   |↑  |0.7621|±  |0.0073|
|  - business_ethics                    |      1|none  |     0|acc   |↑  |0.7400|±  |0.0441|
|  - clinical_knowledge                 |      1|none  |     0|acc   |↑  |0.7811|±  |0.0254|
|  - college_medicine                   |      1|none  |     0|acc   |↑  |0.7572|±  |0.0327|
|  - global_facts                       |      1|none  |     0|acc   |↑  |0.3700|±  |0.0485|
|  - human_aging                        |      1|none  |     0|acc   |↑  |0.7040|±  |0.0306|
|  - management                         |      1|none  |     0|acc   |↑  |0.8932|±  |0.0306|
|  - marketing                          |      1|none  |     0|acc   |↑  |0.9316|±  |0.0165|
|  - medical_genetics                   |      1|none  |     0|acc   |↑  |0.8400|±  |0.0368|
|  - miscellaneous                      |      1|none  |     0|acc   |↑  |0.8442|±  |0.0130|
|  - nutrition                          |      1|none  |     0|acc   |↑  |0.7843|±  |0.0236|
|  - professional_accounting            |      1|none  |     0|acc   |↑  |0.5709|±  |0.0295|
|  - professional_medicine              |      1|none  |     0|acc   |↑  |0.7941|±  |0.0246|
|  - virology                           |      1|none  |     0|acc   |↑  |0.5422|±  |0.0388|
| - social sciences                     |      2|none  |      |acc   |↑  |0.8222|±  |0.0068|
|  - econometrics                       |      1|none  |     0|acc   |↑  |0.6491|±  |0.0449|
|  - high_school_geography              |      1|none  |     0|acc   |↑  |0.8384|±  |0.0262|
|  - high_school_government_and_politics|      1|none  |     0|acc   |↑  |0.9223|±  |0.0193|
|  - high_school_macroeconomics         |      1|none  |     0|acc   |↑  |0.7795|±  |0.0210|
|  - high_school_microeconomics         |      1|none  |     0|acc   |↑  |0.9034|±  |0.0192|
|  - high_school_psychology             |      1|none  |     0|acc   |↑  |0.9009|±  |0.0128|
|  - human_sexuality                    |      1|none  |     0|acc   |↑  |0.8168|±  |0.0339|
|  - professional_psychology            |      1|none  |     0|acc   |↑  |0.7631|±  |0.0172|
|  - public_relations                   |      1|none  |     0|acc   |↑  |0.7182|±  |0.0431|
|  - security_studies                   |      1|none  |     0|acc   |↑  |0.7878|±  |0.0262|
|  - sociology                          |      1|none  |     0|acc   |↑  |0.8557|±  |0.0248|
|  - us_foreign_policy                  |      1|none  |     0|acc   |↑  |0.8400|±  |0.0368|
| - stem                                |      2|none  |      |acc   |↑  |0.7168|±  |0.0078|
|  - abstract_algebra                   |      1|none  |     0|acc   |↑  |0.6100|±  |0.0490|
|  - anatomy                            |      1|none  |     0|acc   |↑  |0.7185|±  |0.0389|
|  - astronomy                          |      1|none  |     0|acc   |↑  |0.8684|±  |0.0275|
|  - college_biology                    |      1|none  |     0|acc   |↑  |0.8403|±  |0.0306|
|  - college_chemistry                  |      1|none  |     0|acc   |↑  |0.5300|±  |0.0502|
|  - college_computer_science           |      1|none  |     0|acc   |↑  |0.6100|±  |0.0490|
|  - college_mathematics                |      1|none  |     0|acc   |↑  |0.5600|±  |0.0499|
|  - college_physics                    |      1|none  |     0|acc   |↑  |0.5784|±  |0.0491|
|  - computer_security                  |      1|none  |     0|acc   |↑  |0.8200|±  |0.0386|
|  - conceptual_physics                 |      1|none  |     0|acc   |↑  |0.8085|±  |0.0257|
|  - electrical_engineering             |      1|none  |     0|acc   |↑  |0.7586|±  |0.0357|
|  - elementary_mathematics             |      1|none  |     0|acc   |↑  |0.7116|±  |0.0233|
|  - high_school_biology                |      1|none  |     0|acc   |↑  |0.8903|±  |0.0178|
|  - high_school_chemistry              |      1|none  |     0|acc   |↑  |0.7044|±  |0.0321|
|  - high_school_computer_science       |      1|none  |     0|acc   |↑  |0.8600|±  |0.0349|
|  - high_school_mathematics            |      1|none  |     0|acc   |↑  |0.5037|±  |0.0305|
|  - high_school_physics                |      1|none  |     0|acc   |↑  |0.6689|±  |0.0384|
|  - high_school_statistics             |      1|none  |     0|acc   |↑  |0.7269|±  |0.0304|
|  - machine_learning                   |      1|none  |     0|acc   |↑  |0.6250|±  |0.0460|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.7279|±  |0.0036|
| - humanities     |      2|none  |      |acc   |↑  |0.6413|±  |0.0068|
| - other          |      2|none  |      |acc   |↑  |0.7621|±  |0.0073|
| - social sciences|      2|none  |      |acc   |↑  |0.8222|±  |0.0068|
| - stem           |      2|none  |      |acc   |↑  |0.7168|±  |0.0078|

"""
