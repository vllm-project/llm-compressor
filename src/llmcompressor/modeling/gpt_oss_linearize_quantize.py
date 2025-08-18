import gc
import torch
from torch import nn
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.utils.dev import skip_weights_initialize
from llmcompressor.modifiers.quantization import QuantizationModifier

def convert_model_for_quantization_gptoss(model):
    to_delete = []

    for name, module in model.named_modules():
        if not (hasattr(module, "experts") and hasattr(module, "router")):
            continue
        experts = module.experts
        if not (hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")):
            continue

        gup = experts.gate_up_proj        # [E, H, 2I]
        dwn = experts.down_proj           # [E, I, H]
        assert gup.dim() == 3 and dwn.dim() == 3
        E, gH, g2i = gup.shape
        Ed, dI, dH = dwn.shape
        assert E == Ed and gH == dH
        assert g2i % 2 == 0
        intermediate = g2i // 2
        hidden = gH

        parent, child_name = _get_parent_and_child(model, name)
        top_k = int(max(1, min(_get_top_k(model.config) or 1, E)))
        seq = SequentialGPTOSSMoE(
            hidden_size=hidden,             
            intermediate_size=intermediate,
            top_k=top_k,
            original_moe=module,
        )
        parent._modules[child_name] = seq
        to_delete.append(module)
        print(f"[GPT-OSS] Patched {name} -> SequentialGPTOSSMoE (E={E}, inter={intermediate}, hidden={hidden})", flush=True)

    for m in to_delete:
        del m
    if to_delete:
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _get_parent_and_child(model, dotted_name: str):
    parts = dotted_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _get_hidden_size(config):
    return getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)


def _get_top_k(config):
    # GPT-OSS MoE: experts per token
    return getattr(config, "num_experts_per_tok", None) or getattr(config, "num_experts_per_token", 1)


class GPTOSSMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.alpha = 1.702
        self.limit = 7.0
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x):
        gate = self.gate_proj(x)
        up   = self.up_proj(x)
        gate = gate.clamp(max=self.limit)
        up   = up.clamp(min=-self.limit, max=self.limit)
        glu  = gate * torch.sigmoid(gate * self.alpha)
        act  = (up + 1) * glu
        return self.down_proj(act)


class SequentialGPTOSSMoE(nn.Module):
    """
    Replaces GPT-OSS fused-expert MoE with per-expert GPTOSSMLP modules.
    Copies weights from fused tensors and reuses the original router and optional shared_expert.
    """
    def __init__(self, hidden_size, intermediate_size, top_k, original_moe):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate = intermediate_size
        self.top_k = top_k
        self.router = original_moe.router
        self.shared_expert = getattr(original_moe, "shared_expert", None)

        # Number of experts
        E = original_moe.experts.gate_up_proj.shape[0]
        self.num_experts = E

        # Build per-expert MLPs
        self.experts = nn.ModuleList()
        with skip_weights_initialize():
            for _ in range(E):
                self.experts.append(GPTOSSMLP(hidden_size, intermediate_size))

        gup   = original_moe.experts.gate_up_proj        # [E, H, 2I]
        gup_b = original_moe.experts.gate_up_proj_bias   # [E, 2I]
        dwn   = original_moe.experts.down_proj           # [E, I, H]
        dwn_b = original_moe.experts.down_proj_bias      # [E, H]

        for i in range(E):
            gup_i = gup[i]                 # [H, 2I]
            gate_w = gup_i[:, ::2]         # [H, I]
            up_w   = gup_i[:, 1::2]        # [H, I]
            down_w = dwn[i]                # [I, H]

            mlp = self.experts[i]
            mlp.gate_proj.weight.data.copy_(gate_w.T)    # [I, H]
            mlp.up_proj.weight.data.copy_(up_w.T)        # [I, H]
            mlp.down_proj.weight.data.copy_(down_w.T)    # [H, I]

            gate_b = gup_b[i]                              # [2I]
            mlp.gate_proj.bias.data.copy_(gate_b[::2])     # [I]
            mlp.up_proj.bias.data.copy_(gate_b[1::2])      # [I]
            mlp.down_proj.bias.data.copy_(dwn_b[i])        # [H]



    def forward(self, hidden_states):
        B, T, H = hidden_states.shape
        x = hidden_states.reshape(-1, H)

        # Use the original router (it returns scores and indices already softmaxed over top-k)
        router_scores, router_indices = self.router(x)   # scores: [tokens, E], indices: [tokens, k]

        out = self.shared_expert(x) if self.shared_expert is not None else torch.zeros_like(x)

        # Accumulate expert outputs for chosen experts only
        for j in range(self.top_k):
            idx = router_indices[:, j]
            w   = router_scores[torch.arange(idx.size(0), device=idx.device), idx].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (idx == e)
                if not torch.any(mask):
                    continue
                out[mask] += self.experts[e](x[mask]) * w[mask]

        out = out.view(B, T, H)
        router_scores = router_scores.view(B * T, -1) 
        return out, router_scores


model_id = "/mnt/nvme4/openai/gpt-oss-120b-BF16"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

convert_model_for_quantization_gptoss(model)

# -----------------------------
# Calibration data & preprocessing
# -----------------------------
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048


# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }

ds = ds.map(preprocess)

# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

ds = ds.map(tokenize, remove_columns=ds.column_names)

# -----------------------------
# Quantization recipe
# -----------------------------
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=[
        "re:.*lm_head",
        're:.*self_attn',
        're:.*attn',
        're:.*attention.*',
        're:.*router',
    ],
)

SAVE_DIR = f"/proving-grounds/machine/shubhra/gpt_oss_120b/{os.path.basename(model_id)}-{recipe.scheme}_ns{NUM_CALIBRATION_SAMPLES}"

# -----------------------------
# Oneshot quantization
# -----------------------------
oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    output_dir=SAVE_DIR,
)

# Save compressed
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
