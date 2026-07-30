import types

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.core.session_functions import active_session
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.pipelines import CalibrationPipeline

VOCAB, HIDDEN, SEQ_LEN, NUM_SAMPLES = 64, 128, 16, 8


class _MainBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.proj(x)


class _MainTransformer(nn.Module):
    """Mirrors a vendored model whose top level forward is decorated for
    inference only use (e.g. deepseekv32.model.Transformer.forward prior to
    the fix for #2745)."""

    def __init__(self, embed, hidden):
        super().__init__()
        self.embed = embed
        self.main_block = _MainBlock(hidden)

    @torch.no_grad()
    def forward(self, input_ids):
        x = self.embed(input_ids)
        return self.main_block(x)


class _AuxBlock(nn.Module):
    """Auxiliary head that reuses the same embedding module as the main path
    and consumes the main path's output, mirroring a speculative decode head
    that aliases the base model's embedding."""

    def __init__(self, embed, hidden):
        super().__init__()
        self.embed = embed
        self.e_proj = nn.Linear(hidden, hidden, bias=False)
        self.h_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden_states, input_ids):
        embed_out = self.embed(input_ids)
        return self.e_proj(embed_out) + self.h_proj(hidden_states)


class _ToyModel(nn.Module):
    def __init__(self, vocab=VOCAB, hidden=HIDDEN):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.transformer = _MainTransformer(self.embed, hidden)
        self.aux_block = _AuxBlock(self.embed, hidden)
        self.config = types.SimpleNamespace(_attn_implementation="eager")

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids):
        x = self.transformer(input_ids)
        aux_out = self.aux_block(x, input_ids)
        return x, aux_out


class _ToyDataset(Dataset):
    def __len__(self):
        return NUM_SAMPLES

    def __getitem__(self, idx):
        return {"input_ids": torch.randint(0, VOCAB, (SEQ_LEN,))}


def test_gptq_writeback_survives_forward_that_escapes_no_grad():
    """
    Regression test for https://github.com/vllm-project/llm-compressor/issues/2745

    A submodule whose forward pass is decorated with @torch.inference_mode()
    (as some vendored model definitions were, see deepseekv32/model.py)
    produces output tensors that stay tagged as inference tensors even after
    the call returns. If a downstream module (e.g. a speculative decode head
    reusing the base model's embedding) consumes that output, GPTQ's weight
    writeback later crashes trying to mutate a tensor derived from it in
    place. Using @torch.no_grad() instead never produces inference tensors,
    so the same calibration flow should complete without error.
    """
    model = _ToyModel()
    dataloader = DataLoader(_ToyDataset(), batch_size=1)

    dataset_args = DatasetArguments(
        pipeline="sequential",
        sequential_targets=["_MainTransformer", "_AuxBlock"],
    )
    modifier = GPTQModifier(targets=[r"re:.*proj$"], scheme="W4A16")

    session = active_session()
    session.reset()
    session.initialize(
        model=model,
        recipe=[modifier],
        calib_data=dataloader,
        sequential_targets=dataset_args.sequential_targets,
    )

    pipeline = CalibrationPipeline.from_modifiers(
        session.lifecycle.recipe.modifiers, user="sequential"
    )
    pipeline(model, dataloader, dataset_args)
    session.finalize()
