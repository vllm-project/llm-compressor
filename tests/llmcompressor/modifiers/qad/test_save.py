import json

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)
from transformers.utils.quantization_config import CompressedTensorsConfig

from llmcompressor import oneshot
from llmcompressor.modifiers.qad import QADModifier

from .test_base import _quantizer


@pytest.mark.parametrize("kind", ["rtn", "gptq"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("qad_data", ["shared", "dataset", "dataloader"])
@pytest.mark.parametrize("teacher_mode", ["local", "full"])
def test_fp16_nvfp4_save_reload(kind, device, qad_data, teacher_mode, tmp_path):
    has_cuda = (
        torch.accelerator.is_available()
        and torch.accelerator.current_accelerator().type == "cuda"
    )
    if device == "cuda" and not has_cuda:
        pytest.skip("CUDA device required")
    if device == "cpu" and has_cuda:
        pytest.skip("Run CPU tests with CUDA_VISIBLE_DEVICES empty")
    torch.manual_seed(31)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    ).to(device=device, dtype=torch.float16)
    data = [
        {
            "input_ids": torch.randint(0, 64, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }
        for _ in range(4)
    ]
    qad = QADModifier(
        num_epochs=2,
        learning_rate=2e-6,
        gradient_accumulation_steps=2,
        teacher_mode=teacher_mode,
    )
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    vocab.update({f"token{i}": i for i in range(4, 64)})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    dataset = Dataset.from_list(
        [{name: value[0].tolist() for name, value in batch.items()} for batch in data]
    )
    extra = {}
    if qad_data != "shared":
        rows = [{"input_ids": [i + 4] * 8, "attention_mask": [1] * 8} for i in range(6)]
        extra = {
            "qad_dataset": Dataset.from_list(rows),
            "qad_dataset_args": {
                "num_calibration_samples": 6,
                "max_seq_length": 8,
                "shuffle_calibration_samples": False,
            },
        }
        if qad_data == "dataloader":
            extra["qad_dataset"] = DataLoader(
                [
                    {key: torch.tensor(value) for key, value in row.items()}
                    for row in rows
                ],
                batch_size=1,
            )
    oneshot(
        model=model,
        processor=tokenizer,
        dataset=dataset,
        recipe=[_quantizer(kind), qad],
        pipeline="sequential",
        sequential_targets=["LlamaDecoderLayer"],
        sequential_targets_per_subgraph=1 if teacher_mode == "full" else 2,
        num_calibration_samples=4,
        max_seq_length=16,
        batch_size=1,
        shuffle_calibration_samples=False,
        **extra,
    )
    steps = 4 if qad_data == "shared" else 6
    assert qad.optimizer_steps == (
        {"subgraph_1": steps, "subgraph_2": steps}
        if teacher_mode == "full"
        else {"subgraph_1": steps}
    )
    model.save_pretrained(
        tmp_path,
        save_compressed=True,
        max_shard_size="20KB",
        save_original_format=False,
    )
    config = json.loads((tmp_path / "config.json").read_text())
    for group in config["quantization_config"]["config_groups"].values():
        assert group["weights"]["num_bits"] == 4
        assert group["weights"]["type"] == "float"
        assert group["input_activations"] is None
        assert group["output_activations"] is None
    assert len(list(tmp_path.glob("*.safetensors"))) > 1
    loaded = AutoModelForCausalLM.from_pretrained(
        tmp_path,
        dtype=torch.float16,
        device_map=device,
        quantization_config=CompressedTensorsConfig(run_compressed=False),
    ).to(dtype=torch.float16)
    with torch.no_grad():
        ids = data[0]["input_ids"].to(device)
        assert torch.isfinite(loaded(input_ids=ids).logits).all()
        generated = loaded.generate(input_ids=ids, max_new_tokens=2, do_sample=False)
        assert generated.shape[1] > ids.shape[1]
