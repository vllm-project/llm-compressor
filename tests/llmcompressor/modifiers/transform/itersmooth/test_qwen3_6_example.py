"""Check example input handling and pipeline dispatch without model downloads.

These tests use real Datasets/Arrow and lightweight model/runtime stand-ins.
They do not validate PyTorch execution, processor numerics, or quantization accuracy.
"""

import ast
import json
import unittest
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset

ROOT = Path(__file__).resolve().parents[5]
EXAMPLE = ROOT / "examples/quantization_w8a8_int8/qwen3_6_itersmooth_example.py"


class Tensor:
    def __init__(self, value, dtype=None):
        self.value = value
        self.dtype = dtype
        self.device = None

    def tolist(self):
        return self.value

    def to(self, device):
        self.device = device
        return self


# Exercise the actual basic-pipeline transfer helper without importing PyTorch.
helper_path = ROOT / "src/llmcompressor/pytorch/utils/helpers.py"
helper_tree = ast.parse(helper_path.read_text(encoding="utf-8"))
helper_tree.body = [
    node
    for node in helper_tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "tensors_to_device"
]
helper_namespace = dict(
    Tensor=Tensor, OrderedDict=OrderedDict, Iterable=Iterable, Mapping=Mapping, Any=Any
)
exec(compile(helper_tree, str(helper_path), "exec"), helper_namespace)
tensors_to_device = helper_namespace["tensors_to_device"]


class Processor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        values = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
        if messages[0]["content"][0]["type"] == "image":
            values.update(
                pixel_values=[[1.0, 2.0], [3.0, 4.0]], image_grid_thw=[[1, 2, 2]]
            )
        return {key: Tensor(value) for key, value in values.items()}

    def save_pretrained(self, directory):
        pass


class Modifier:
    def __init__(self, **kwargs):
        self.options = kwargs


class IterSmooth(Modifier):
    pass


class Quantization(Modifier):
    pass


class TestQwen36CalibrationExample(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(dir=ROOT)
        self.directory = Path(temporary.name).resolve()
        assert self.directory.parent == ROOT
        self.addCleanup(temporary.cleanup)
        self.image = self.directory / "image.png"
        self.image.touch()  # Processor stand-in does not decode image bytes.
        self.image_data = self.write_jsonl(
            "images.jsonl", [{"image": "image.png", "text": "Read the chart."}]
        )
        self.text_data = self.write_jsonl(
            "qwen36.jsonl", [{"text": "Explain the result."}, {"text": "Solve this."}]
        )

    def write_jsonl(self, filename, rows):
        path = self.directory / filename
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        return str(path)

    def run_example(self, **settings):
        processor = Processor()
        model = SimpleNamespace(eval=lambda: None, save_pretrained=lambda *a, **k: None)
        calls, offloads = [], []

        def oneshot(**kwargs):
            batches = (
                [kwargs["data_collator"]([row]) for row in kwargs["dataset"]]
                if "dataset" in kwargs
                else []
            )
            if kwargs["pipeline"] == "basic":
                batches = [tensors_to_device(batch, "cpu") for batch in batches]
            calls.append((kwargs, batches))

        namespace = dict(
            Path=Path,
            torch=SimpleNamespace(as_tensor=Tensor, long="int64"),
            find_spec=lambda name: None,
            Dataset=Dataset,
            load_dataset=load_dataset,
            concatenate_datasets=concatenate_datasets,
            AutoProcessor=SimpleNamespace(from_pretrained=lambda name: processor),
            Qwen3_5ForConditionalGeneration=SimpleNamespace(
                from_pretrained=lambda *a, **k: model
            ),
            IterSmoothModifier=IterSmooth,
            QuantizationModifier=Quantization,
            get_main_device=lambda: SimpleNamespace(type="cuda"),
            load_context=lambda cls: nullcontext(),
            set_onload_device=lambda *args: offloads.append(args),
            oneshot=oneshot,
            print=lambda *args: None,
        )
        tree = ast.parse(EXAMPLE.read_text(encoding="utf-8"))
        tree.body = [
            node
            for node in tree.body
            if not isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in settings:
                    node.value = ast.Constant(settings[target.id])
        exec(compile(ast.fix_missing_locations(tree), str(EXAMPLE), "exec"), namespace)
        return processor.calls, calls, offloads

    def test_mixed_inputs_use_one_smoothing_pass_before_quantization(self):
        prompts, calls, offloads = self.run_example(
            CALIBRATION_DATA=self.image_data,
            TEXT_CALIBRATION_DATA=self.text_data,
            TEXT_MAX_SEQUENCE_LENGTH=2048,
        )
        self.assertEqual(len(offloads), 1)
        self.assertEqual([call[0]["pipeline"] for call in calls], ["basic", "datafree"])
        self.assertIsInstance(calls[0][0]["recipe"], IterSmooth)
        self.assertIsInstance(calls[1][0]["recipe"], Quantization)
        self.assertNotIn("dataset", calls[1][0])
        self.assertEqual(calls[0][0]["num_calibration_samples"], 3)
        batches = calls[0][1]
        self.assertEqual(batches[0]["pixel_values"].value, [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(batches[0]["image_grid_thw"].value, [[1, 2, 2]])
        self.assertEqual(batches[1]["input_ids"].value, [[1, 2, 3]])
        self.assertNotIn("pixel_values", batches[1])
        self.assertNotIn("image_grid_thw", batches[1])
        for batch in batches:
            self.assertEqual(batch["logits_to_keep"].value, [-1])
            self.assertEqual(batch["logits_to_keep"].dtype, "int64")
            self.assertEqual(batch["logits_to_keep"].device, "cpu")
        self.assertEqual(prompts[0][0][0]["content"][0]["image"], str(self.image))
        self.assertEqual(prompts[0][1]["processor_kwargs"], {})
        self.assertEqual(
            prompts[1][1]["processor_kwargs"], {"truncation": True, "max_length": 2048}
        )
        self.assertNotIn("max_length", prompts[1][1])
        self.assertNotIn("truncation", prompts[1][1])
        self.assertEqual(
            prompts[1][0][0]["content"],
            [{"type": "text", "text": "Explain the result."}],
        )

    def test_text_only_keeps_sequential_decoder_boundaries(self):
        _, calls, offloads = self.run_example(
            CALIBRATION_DATA=None, TEXT_CALIBRATION_DATA=self.text_data
        )
        self.assertEqual(offloads, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0]["pipeline"], "independent")
        self.assertEqual(calls[0][0]["sequential_targets"], ["Qwen3_5DecoderLayer"])
        self.assertNotIn("pixel_values", calls[0][1][0])

    def test_image_only_retains_untruncated_inputs_and_default_prompt(self):
        image_data = self.write_jsonl("default.jsonl", [{"image": "image.png"}])
        prompts, calls, _ = self.run_example(CALIBRATION_DATA=image_data)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0]["pipeline"], "independent")
        self.assertEqual(
            prompts[0][0][0]["content"][1]["text"], "Describe this image in detail."
        )
        self.assertEqual(prompts[0][1]["processor_kwargs"], {})

    def test_rejects_invalid_text_instead_of_substituting_image_prompt(self):
        for value in (None, "", "   ", 17):
            with self.subTest(value=value):
                text_data = self.write_jsonl("bad.jsonl", [{"text": value}])
                with self.assertRaisesRegex(ValueError, "nonempty user prompt"):
                    self.run_example(
                        CALIBRATION_DATA=None, TEXT_CALIBRATION_DATA=text_data
                    )

    def test_rejects_images_in_text_source(self):
        with self.assertRaisesRegex(ValueError, "Use CALIBRATION_DATA"):
            self.run_example(
                CALIBRATION_DATA=None, TEXT_CALIBRATION_DATA=self.image_data
            )

    def test_rejects_missing_sources(self):
        with self.assertRaisesRegex(ValueError, "and/or"):
            self.run_example(CALIBRATION_DATA=None, TEXT_CALIBRATION_DATA=None)

    def test_rejects_missing_image_instead_of_converting_to_text(self):
        image_data = self.write_jsonl(
            "missing.jsonl", [{"image": None, "text": "Question"}]
        )
        with self.assertRaisesRegex(ValueError, "image path"):
            self.run_example(CALIBRATION_DATA=image_data)

    def test_text_inputs_are_untruncated_by_default(self):
        prompts, _, _ = self.run_example(
            CALIBRATION_DATA=None,
            TEXT_CALIBRATION_DATA=self.text_data,
        )
        self.assertEqual(prompts[0][1]["processor_kwargs"], {})

    def test_reads_all_text_records_without_a_shared_sample_limit(self):
        text_data = self.write_jsonl(
            "many.jsonl", [{"text": f"Question {index}"} for index in range(150)]
        )
        prompts, calls, _ = self.run_example(
            CALIBRATION_DATA=self.image_data, TEXT_CALIBRATION_DATA=text_data
        )
        self.assertEqual(len(prompts), 151)
        self.assertEqual(calls[0][0]["num_calibration_samples"], 151)

    def test_image_directory_order_and_prompt_file_guard(self):
        directory = self.directory / "images"
        directory.mkdir()
        for name in ("b.png", "a.jpg"):
            (directory / name).touch()
        prompts, _, _ = self.run_example(CALIBRATION_DATA=str(directory))
        image_paths = [prompt[0][0]["content"][0]["image"] for prompt in prompts]
        self.assertEqual([Path(path).name for path in image_paths], ["a.jpg", "b.png"])
        (directory / "index.jsonl").touch()
        with self.assertRaisesRegex(ValueError, "prompt file"):
            self.run_example(CALIBRATION_DATA=str(directory))


if __name__ == "__main__":
    unittest.main()
