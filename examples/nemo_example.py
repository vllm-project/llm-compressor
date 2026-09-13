# requires: einops, fla-core, tiktoken
from compressed_tensors.distributed import init_dist
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import REAPPruningModifier
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.utils import load_context

# Small representative model with same MXFP4 quantization
MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"

# Load model with the modified quantization config
init_dist()
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto_offload",
        max_memory={},
        offload_folder="/data/kylesayrs/hub/offload_folder",
    )
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

recipe = [
    REAPPruningModifier(sparsity=0.25),
    GPTQModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=[
            r"re:.*conv1d.*",
            r"backbone\.embeddings",
            r"re:.*_latent_proj.*",
            r"re:.*mixer.gate\..*",
            r"re:mtp.layers.*",
            "backbone.norm_f",
            "lm_head",
        ],
    )
]

oneshot(
    model=model,
    tokenizer=processor.tokenizer,
    dataset="perfectblend",
    splits="train[:512]",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=1024,
    trust_remote_code_model=True,
    batch_size=16,
    pipeline="sequential"
)

SAVE_DIR = "/data/kylesayrs/hub/" + MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-REAP25"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
