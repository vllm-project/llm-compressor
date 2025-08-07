# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from compressed_tensors.transform import TransformArgs, TransformFactory
from transformers import PretrainedConfig, PreTrainedModel


class TransformableModel(PreTrainedModel):
    def __init__(self, *sizes):
        super().__init__(config=PretrainedConfig())
        self.fcs = torch.nn.ModuleList(
            [
                torch.nn.Linear(sizes[index], sizes[index + 1], bias=False)
                for index in range(0, len(sizes) - 1)
            ]
        )

    def forward(self, x):
        for layer in self.fcs:
            x = layer(x)
        return x


class MockAttention(torch.nn.Module):
    def __init__(
        self, hidden_size: int, num_attention_heads: int, num_key_value_heads: int
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim**-0.5
        assert hidden_size >= num_attention_heads * self.head_dim

        self.q_proj = torch.nn.Linear(
            hidden_size, num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = torch.nn.Linear(
            hidden_size, num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = torch.nn.Linear(
            hidden_size, num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = torch.nn.Linear(
            num_attention_heads * self.head_dim, hidden_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        )

        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape((batch_size, seq_len, -1)).contiguous()

        return self.o_proj(attn_output)

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@pytest.fixture(scope="function")
def model_apply():
    model = TransformableModel(2, 4, 8, 16, 32, 64)
    apply = [
        # weight output -> input
        TransformArgs(targets="fcs.0", location="weight_output"),
        TransformArgs(targets="fcs.1", location="input", inverse=True),
        # output -> weight input
        TransformArgs(targets="fcs.1", location="output"),
        TransformArgs(targets="fcs.2", location="weight_input", inverse=True),
        # output -> input
        TransformArgs(targets="fcs.2", location="output"),
        TransformArgs(targets="fcs.3", location="input", inverse=True),
        # weight output -> weight input
        TransformArgs(targets="fcs.3", location="weight_output"),
        TransformArgs(targets="fcs.4", location="weight_input", inverse=True),
    ]

    return model, apply
