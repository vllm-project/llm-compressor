from transformers import Llama4ForConditionalGeneration, Llama4Processor
from transformers.quantizers.quantizers_utils import get_module_from_name
import torch
from datasets import load_dataset

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils.dev import skip_weights_initialize
from transformers.models.llama4.modeling_llama4 import Llama4TextMLP
import gc


def convert_model_for_quantization(model):
    to_delete = []
    for name, module in model.named_modules():
        module_class_name = module.__class__.__name__
        if module_class_name == "Llama4TextMoe":
            parent_module, module_name = get_module_from_name(model, name)
            parent_module._modules[module_name] = SequentialLlama4TextMoe(
                model.config.get_text_config(),
                module,
            )
            to_delete.append(module)
            print(f"Patched {name} with SequentialLlama4TextMoe", flush=True)

    for module in to_delete:
        del module
        gc.collect()
        torch.cuda.empty_cache()


class SequentialLlama4TextMoe(torch.nn.Module):
    def __init__(self, config, original_moe):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.experts = SequentialLlama4TextExperts(config, original_moe.experts)
        self.router = original_moe.router
        self.shared_expert = original_moe.shared_expert

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self.router(hidden_states)

        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)

        router_scores = (
            torch.full_like(router_logits, float("-inf")).scatter_(1, router_indices, router_top_value).transpose(0, 1)
        )
        router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

        out = self.shared_expert(hidden_states)
        for i in range(self.num_experts):
            out += self.experts[i](hidden_states) * router_scores[i].reshape(-1, 1)

        return out, router_scores


class SequentialLlama4TextExperts(torch.nn.ModuleList):
    def __init__(self, config, original_experts):
        self.num_experts = original_experts.gate_up_proj.shape[0]
        with skip_weights_initialize():
            super().__init__([Llama4TextMLP(config) for _ in range(self.num_experts)])

        intermediate_size = original_experts.down_proj.shape[1]

        for i in range(self.num_experts):
            gate_up = original_experts.gate_up_proj[i]
            down = original_experts.down_proj[i]

            gate_proj = gate_up[:, :intermediate_size]
            up_proj = gate_up[:, intermediate_size:]

            self[i].gate_proj.weight.data = gate_proj.t().clone().contiguous()
            self[i].up_proj.weight.data = up_proj.t().clone().contiguous()
            self[i].down_proj.weight.data = down.t().clone().contiguous()

        original_experts.gate_up_proj = None
        original_experts.down_proj = None
        gc.collect()
        torch.cuda.empty_cache()


model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"

model = Llama4ForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16  # load on cpu
)
processor = Llama4Processor.from_pretrained(model_id)

convert_model_for_quantization(model)

# Oneshot arguments
DATASET_ID = "neuralmagic/calibration"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 8192

ds = load_dataset(DATASET_ID, name="LLM", split=f"train[:{NUM_CALIBRATION_SAMPLES}]")

def preprocess_function(example):
    messgages = []
    for message in example["messages"]:
        messgages.append(
            {
                "role": message["role"], 
                "content": [{"type": "text", "text": message["content"]}]
            }
        )
    
    return processor.apply_chat_template(
        messgages, 
        return_tensors="pt", 
        padding=False, 
        truncation=True, 
        max_length=MAX_SEQUENCE_LENGTH,
        tokenize=True,
        add_special_tokens=False,
        return_dict=True,
        add_generation_prompt=False,
    ).to("cuda:0")

ds = ds.map(
    preprocess_function,
    batched=False,
    remove_columns=ds.column_names
)


# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    return {
        key: torch.tensor(value) if key != "pixel_values" else torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
        for key, value in batch[0].items()
    }


# Recipe
recipe = [
    GPTQModifier(
        targets="Linear",
        dampening_frac=0.02,
        ignore=[
            're:.*lm_head',
            're:.*self_attn',
            're:.*router',
            're:.*vision_model',
            're:.*multi_modal_projector',
            're:.*multi_modal_projector',
            "Llama4TextAttention",
        ],
        sequential_targets=["Llama4TextMLP"],
        config_groups={
            "group0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "strategy": "group",
                    "group_size": 128,
                    "symmetric": True,
                    "actorder": "weight",
                    "observer": "minmax",
                }
            }
        }
    ),
]

# Perform oneshot
oneshot(
    model=model,
    tokenizer=model_id,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    data_collator=data_collator,
    oneshot_device="cuda:0",
)

# Save to disk compressed.
SAVE_DIR = model_id.split("/")[1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
