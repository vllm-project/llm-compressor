"""Example loader for Model Optimizer's seven-source QAT/QAD data blend."""

import json
from dataclasses import asdict, dataclass

import fsspec
import pyarrow.parquet as pq
from datasets import Dataset, IterableDataset, load_dataset, load_dataset_builder
from loguru import logger

MODELOPT_QAD_MIX = "modelopt_qad_mix"
MODELOPT_BLEND_REVISION = "6e4789fa43726f800b6d6f63d6611b6472b00ba0"
MODELOPT_BLEND_URL = (
    "https://github.com/NVIDIA/Model-Optimizer/blob/"
    f"{MODELOPT_BLEND_REVISION}/examples/llm_qat/configs/dataset/blend.yaml"
)


@dataclass(frozen=True)
class QADBlendSource:
    dataset: str
    split: str
    weight: int
    revision: str
    parquet: bool = False


# Relative example weights, not sample counts. Dataset revisions pin the data
# inspected on 2026-09-07 independently of the Model Optimizer config revision.
MODELOPT_QAD_SOURCES = (
    QADBlendSource(
        "nvidia/Nemotron-SWE-v1",
        "r2e_gym",
        6000,
        "0fe17a965b297a9c943a59050a14c42d5f0083ce",
    ),
    QADBlendSource(
        "nvidia/Nemotron-Math-v2",
        "medium",
        2500,
        "8e793210e175b6406c752a870f585f62de98c0d3",
        parquet=True,
    ),
    QADBlendSource(
        "nvidia/Nemotron-Science-v1",
        "MCQ",
        1500,
        "82e1af468197076b4f0f392c239274eac032adc7",
    ),
    QADBlendSource(
        "nvidia/Nemotron-Science-v1",
        "RQA",
        1500,
        "82e1af468197076b4f0f392c239274eac032adc7",
    ),
    QADBlendSource(
        "nvidia/Nemotron-Instruction-Following-Chat-v1",
        "chat_if",
        5000,
        "83dcd3aded0d289b0bbc018d3f9af4c5dd4005df",
    ),
    QADBlendSource(
        "nvidia/Nemotron-Post-Training-Dataset-v2",
        "chat",
        1500,
        "5c89e01dd720ae0f4058445ed49c5fb68a03c76e",
        parquet=True,
    ),
    QADBlendSource(
        "nvidia/Nemotron-Competitive-Programming-v1",
        "competitive_coding_python_part00",
        1000,
        "d6e7c6b404ed5db6e1104b41d0f80a0c7dad7bf8",
    ),
)


def modelopt_qad_sample_counts(num_samples: int) -> list[int]:
    """Normalize weights and use largest remainders to allocate exactly N rows."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    total = sum(source.weight for source in MODELOPT_QAD_SOURCES)
    counts = [num_samples * source.weight // total for source in MODELOPT_QAD_SOURCES]
    order = sorted(
        range(len(counts)),
        key=lambda i: -(num_samples * MODELOPT_QAD_SOURCES[i].weight % total),
    )
    for i in order[: num_samples - sum(counts)]:
        counts[i] += 1
    return counts


def _normalize_messages(messages):
    # Parquet's fixed struct schema materializes absent optional fields. Llama's
    # template tests key presence and treats even tool_calls=[] as a tool call.
    # Copy each message so normalization cannot mutate the source example.
    return [
        {
            key: value
            for key, value in message.items()
            if value is not None and not (key == "tool_calls" and not value)
        }
        for message in messages
    ]


def _iter_parquet_rows(files):
    # Read bounded batches synchronously. Arrow's dataset scanner can retain
    # Python-backed HF file reads at interpreter shutdown after a partial scan.
    for path in files:
        with fsspec.open(path, "rb") as handle:
            with pq.ParquetFile(handle, pre_buffer=False) as parquet:
                for batch in parquet.iter_batches(batch_size=32, use_threads=False):
                    yield from batch.to_pylist()


def _stream_source(source):
    if source.parquet:
        # Resolve the pinned split's files using HF metadata, without downloading
        # the dataset. fsspec's hf:// handler uses normal HF authentication.
        builder = load_dataset_builder(source.dataset, revision=source.revision)
        files = list(builder.config.data_files[source.split])
        return IterableDataset.from_generator(
            _iter_parquet_rows, gen_kwargs={"files": files}
        )
    return load_dataset(
        source.dataset, split=source.split, revision=source.revision, streaming=True
    )


def load_modelopt_qad_mix(
    tokenizer,
    num_samples: int = 512,
    max_seq_length: int = 2048,
    seed: int = 42,
    shuffle_buffer: int = 128,
) -> Dataset:
    """Load the pinned seven-source mix using the model's chat template.

    Only the requested tokenized rows are retained. Streaming may read ahead;
    ``shuffle_buffer`` bounds the number of raw examples shuffled per source.
    Source failures/shortfalls raise instead of silently changing the mixture.
    The caller controls the QAD validation split and loss masking; this helper
    adopts the upstream source ratios, not its training/split/masking recipe.
    Provenance (including exact counts) is JSON in ``dataset.info.description``.
    """
    counts = modelopt_qad_sample_counts(num_samples)
    if max_seq_length < 1 or shuffle_buffer < 1:
        raise ValueError("max_seq_length and shuffle_buffer must be positive")
    rows = []
    sources = []
    for source, count in zip(MODELOPT_QAD_SOURCES, counts):
        sources.append({**asdict(source), "samples": count})
        if not count:
            continue
        logger.info("QAD mix: {} [{}], {} samples", source.dataset, source.split, count)
        try:
            stream = _stream_source(source).shuffle(
                seed=seed, buffer_size=shuffle_buffer
            )
            loaded = 0
            for example in stream.take(count):
                messages = example.get("messages")
                if not messages:
                    raise ValueError("Expected a nonempty messages conversation")
                text = tokenizer.apply_chat_template(
                    _normalize_messages(messages), tokenize=False
                )
                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_seq_length,
                    padding=False,
                    add_special_tokens=False,
                )
                if not encoded["input_ids"]:
                    raise ValueError("Chat template produced an empty token sequence")
                rows.append(dict(encoded))
                loaded += 1
            if loaded != count:
                raise ValueError(f"Requested {count} samples but found {loaded}")
        except Exception as error:
            raise RuntimeError(
                f"Failed to load QAD mix source {source.dataset} [{source.split}]. "
                "No sources are skipped. Check the source data and HF access; "
                "Nemotron-Post-Training-Dataset-v2 requires an authorized HF token."
            ) from error
    dataset = Dataset.from_list(rows).shuffle(seed=seed)
    dataset.info.description = json.dumps(
        {
            "preset": MODELOPT_QAD_MIX,
            "config_url": MODELOPT_BLEND_URL,
            "sources": sources,
            "num_samples": num_samples,
            "seed": seed,
            "shuffle_buffer": shuffle_buffer,
            "parquet_reader": "synchronous iter_batches, batch_size=32",
            "max_seq_length": max_seq_length,
            "allocation": "normalized weights, largest remainder (source-order ties)",
            "tokenization": "model chat template, truncation, all-token loss",
        }
    )
    return dataset
