import argparse
import time
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--max-generated-tokens", type=int, default=250)
parser.add_argument("--num-samples", type=int, default=1000)
parser.add_argument("--max-num-seqs", type=int, default=256)
parser.add_argument("--kv-cache-dtype", type=str, default="auto")
parser.add_argument("--gpu-memory-utilization", type=float, default=.9)

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
NUM_TURNS_PROMPT = 3

if __name__ == "__main__":
    args = parser.parse_args()
    MODEL_ID = args.model
    MAX_GENERATED_TOKENS = args.max_generated_tokens
    MAX_NUM_SEQS = args.max_num_seqs
    NUM_SAMPLES = args.num_samples
    KV_CACHE_DTYPE = args.kv_cache_dtype
    GPU_MEMORY_UTILIZATION = args.gpu_memory_utilization

    # Pre-process your dataset.
    # Its a good idea to use the chat template.
    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"][:NUM_TURNS_PROMPT], tokenize=False, add_generation_prompt=True
        )}

    dataset = load_dataset(DATASET_ID, split="train_sft")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    ds = dataset.shuffle().select(range(NUM_SAMPLES))
    ds = ds.map(preprocess)

    # BE CAREFUL WITH THE TOKENIZER
    #   apply_chat_template already adds the bos_token
    #   so we set add_special_token to false
    examples = [
        tokenizer(example["text"], add_special_tokens=False).input_ids 
        for example in ds
    ]

    # Initialize vLLM
    model = LLM(
        MODEL_ID,
        max_num_seqs=MAX_NUM_SEQS,
        kv_cache_dtype=KV_CACHE_DTYPE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )

    # Generate.
    start = time.perf_counter()
    generations = model.generate(
        prompt_token_ids=examples,
        use_tqdm=True,
        sampling_params=SamplingParams(
            max_tokens=MAX_GENERATED_TOKENS),
    )
    end = time.perf_counter()

    total_generations = len(generations)
    total_prompt_tokens = 0
    total_generation_tokens = 0
    total_time = end - start

    for generation in generations:
        total_prompt_tokens += len(generation.prompt_token_ids)
        total_generation_tokens += len(generation.outputs[0].token_ids)

    print("* ==========================================================")
    print(f"* Total Time: \t\t\t{total_time: 0.2f}")
    print(f"* Total Generations: \t\t{total_generations}")
    print("\n")
    print(f"* Generations / Sec: \t\t{total_generations / total_time :0.2f}")
    print(f"* Generation Tok / Sec: \t{total_generation_tokens / total_time :0.2f}")
    print(f"* Prompt Tok / Sec: \t\t{total_prompt_tokens / total_time :0.2f}")
    print("\n")
    print(f"* Avg Generation Tokens: \t{total_generation_tokens / total_generations :0.2f}")
    print(f"* Avg Prompt Tokens: \t\t{total_prompt_tokens / total_generations :0.2f}")
    print("* ==========================================================")
