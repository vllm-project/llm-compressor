from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# Already compressed
MODEL_ID = "RedHatAI/GLM-5.3-MXFP4"
SAVE_DIR = (
    "/data-tier-1/machine/Roderick-Wu/"
    + MODEL_ID.rstrip("/").split("/")[-1]
    + "-NVFP4-Layerwise"
)
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

IGNORE = [
    "re:.*mlp.gate$",
    "re:.*lm_head",
    "re:.*embed_tokens$",
    "re:.*eh_proj$",
    "re:.*self_attn.indexer.weights_proj$",
]

config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
config.quantization_config["ignore"] = [
    *config.quantization_config.get("ignore", []),
    *IGNORE,
]

with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        device_map="auto_offload",
        offload_folder="./offload_folder",
        dtype="auto",
        trust_remote_code=True,
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=IGNORE,
)

oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset="perfectblend",
    splits=f"train[:{NUM_CALIBRATION_SAMPLES}]",
    batch_size=4,
    recipe=recipe,
    trust_remote_code_model=True,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    shuffle_calibration_samples=False,
    pipeline="sequential",
    sequential_targets=["GlmMoeDsaAttention", "GlmMoeDsaMoE"],
    layerwise_decompression=True,
    layerwise_compression=True,
)

model.generation_config.top_p = None
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
