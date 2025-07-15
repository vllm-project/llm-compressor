import torch
from compressed_tensors.utils import update_parameter_data
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.modifiers.transform import SpinQuantModifier
from llmcompressor.utils import dispatch_for_generation

hidden_dim = intermediate_dim = 64
up_dim = 128
num_embeddings = 12


class DummySelfAttn(torch.nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=None)
        self.k_proj = torch.nn.Linear(hidden_dim, intermediate_dim, bias=None)
        self.v_proj = torch.nn.Linear(hidden_dim, intermediate_dim, bias=None)
        self.o_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=None)
        self.num_heads = 1
        self.num_key_value_groups = 1

    def forward(self, hidden_states):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        ### EAGER ATTENTION
        attn_weights = torch.matmul(q.T, k)

        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v.T)
        attn_output = attn_output.T.contiguous()

        return self.o_proj(attn_output)


class DummyMLP(torch.nn.Module):
    def __init__(self, hidden_dim, up_dim):
        super().__init__()
        self.up_proj = torch.nn.Linear(hidden_dim, up_dim, bias=None)
        self.gate_proj = torch.nn.Linear(hidden_dim, up_dim, bias=None)
        self.down_proj = torch.nn.Linear(up_dim, hidden_dim, bias=None)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DummyModel(torch.nn.Module):
    def __init__(self, num_embeddings, hidden_dim, intermediate_dim, up_dim):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(num_embeddings, hidden_dim)
        self.input_layernorm = LlamaRMSNorm(hidden_dim)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_dim)
        self.self_attn = DummySelfAttn(hidden_dim, intermediate_dim)
        self.mlp = DummyMLP(hidden_dim, up_dim)
        self.lm_head = torch.nn.Linear(hidden_dim, num_embeddings, bias=None)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return self.lm_head(x)


def test_dummy_model():
    model = DummyModel(num_embeddings, hidden_dim, intermediate_dim, up_dim)

    # TODO Uncomment this to see norm diff > 1e-6
    # This is due to issue Kyle spotted in https://arxiv.org/pdf/2405.16406 Page 5 Footnote 2
    # Will have to fuse layernorms with subsequent layers so that input_layernorm.weight is equal to torch.ones() (this apparently makes it rotation invariant)
    # https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/utils/fuse_norm_utils.py#L39
    # update_parameter_data(
    #     model.input_layernorm,
    #     torch.rand(model.input_layernorm.weight.shape),
    #     "weight",
    # )

    input_ids = torch.IntTensor([1, 2, 3, 4, 5])
    orig_output = model(input_ids)

    recipe = [
        SpinQuantModifier(rotations=["R1", "R2"]),
    ]

    # TODO: work around preprocessing?
    oneshot(
        model=model,
        recipe=recipe,
        pipeline="datafree",
        log_dir=None,
    )

    # # Confirm generations of the quantized model look the same
    transformed_output = model(input_ids)

    print(f"Norm Diff {(orig_output-transformed_output).norm()}")
    print(f"Norm {orig_output.norm()}, {transformed_output.norm()}")
