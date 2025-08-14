import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.loggers import EvaluationTracker
from lm_eval.utils import make_table
from vllm.forward_context import get_forward_context
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


# -----------------------------------------
# --- Step 0. define constants and args ---
# -----------------------------------------
# NOTE !!! starts here !!!

# MODEL_ID = "/net/storage149/autofs/css22/nmg/models/cos/f05940d/lake-models/models/base_training/shared/granite-4.0-small-base-prerelease-greylock-128k/r250709a"
# MODEL_ID = "ibm-granite/granite-4.0-tiny-preview"
# MODEL_ID = "/net/storage149/autofs/css22/cliu22/ssm_state_compression/gr4small_fp8_skipRouter_dequant"
MODEL_ID = "/net/storage149/autofs/css22/cliu22/ssm_state_compression/gr4small_fp8_skipRouter_lin"

tasks_to_evaluate = ["gsm8k"]  # , "gsm8k", "truthfulqa", "lambada_openai", "hellaswag"]  # Or your desired tasks
num_fewshot = 5  # Number of few-shot examples, None -> task default, specify only when needed.

model_type = "hf"  #"vllm"  # 
limit = None  # 30  #  
trust_remote_code = True
cache_req = True
batch_size = "auto"  # 1  # 
model_dtype = "bfloat16"  # auto?
# output_samples_to_json = True

# "hf" -> args will be sent to HFLM.__init__(), attach to self._model
# "vllm" -> VLLM.init() which calls vllm's LLM class's init and attaches to self.model, better use enforce_eager=True 
match model_type:
    case "hf":
        model_args = f"pretrained={MODEL_ID},dtype={model_dtype},parallelize=True"  # ,tp_plan=auto
    case "vllm":
        model_args = f"pretrained={MODEL_ID},dtype={model_dtype},tensor_parallel_size=1,gpu_memory_utilization=0.95,enforce_eager=True"  # ,quantization=fp8
    case _:
        raise RuntimeError("only hf and vllm are allowed for now.")

# --- Step 1. Load/create HF model or use lm-eval's get_model() directly.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto").cuda()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# lm_obj = lm_eval.api.registry.get_model(model_type).create_from_arg_string(
#     # signature is (args, additional_args)
#     model_args,
#     {
#         "batch_size": batch_size,  # "auto",
#         # "max_batch_size": max_batch_size,
#         "trust_remote_code": trust_remote_code,
#         "device": "cuda",  # "auto", #
#     },
# )
lm_obj = HFLM(
    model,
    tokenizer=tokenizer,
    batch_size=batch_size,
    trust_remote_code=trust_remote_code,
    device="cuda",
    # parallelize=True,
)

if model_type == "hf":
    model_shortcut = lm_obj.model
else:
    model_shortcut = lm_obj.model.llm_engine.model_executor.driver_worker.worker.model_runner.model

print(model_shortcut)


# --- Step 3.2: full eval, invoke lm-eval
eval_kwargs = {
    "model": lm_obj,
    "model_args": model_args,
    "tasks": tasks_to_evaluate,
    "num_fewshot": num_fewshot,
    # Add other parameters as needed, e.g., batch_size, device
    "limit": limit,
    "cache_requests": cache_req,
    "device": "cuda",
}
    
results = lm_eval.simple_evaluate(**eval_kwargs,)




# --- step 4: print out results (borrowed from lm-eval cli)
print(
    f"{model_type} ({model_args}), limit: {limit}, num_fewshot: {"task default" if num_fewshot is None else num_fewshot}, "
    # f"batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
)
print(make_table(results))
if "groups" in results:
    print(make_table(results, "groups"))



