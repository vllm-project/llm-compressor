from typing import Any, Dict, List, Optional

import kfp
from kfp import dsl, kubernetes


@kfp.dsl.component(
    base_image="quay.io/opendatahub/llmcompressor-pipeline-runtime:main",
    packages_to_install=["llmcompressor~=0.5.0"],
)
def run_oneshot_datafree(
    model_id: str, recipe: str, output_model: dsl.Output[dsl.Artifact]
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmcompressor import oneshot

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = oneshot(model=model, recipe=recipe, tokenizer=tokenizer)
    model.save_pretrained(output_model.path)
    tokenizer.save_pretrained(output_model.path)

    return


@kfp.dsl.component(
    base_image="quay.io/opendatahub/llmcompressor-pipeline-runtime:main",
    packages_to_install=["llmcompressor~=0.5.0"],
)
def run_oneshot_calibrated(
    model_id: str,
    recipe: str,
    dataset_id: str,
    dataset_split: str,
    output_model: dsl.Output[dsl.Artifact],
    num_calibration_samples: int = 512,
    max_sequence_length: int = 2048,
    seed: int = 42,
):
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from llmcompressor import oneshot

    # Load dataset.
    ds = load_dataset(dataset_id, split=dataset_split)
    ds = ds.shuffle(seed=seed).select(range(num_calibration_samples))

    # Preprocess the data into the format the model is trained with.
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }

    ds = ds.map(preprocess)

    # Tokenize the data (be careful with bos tokens - we need add_special_tokens=False since the chat_template already added it).
    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        tokenizer=tokenizer,
        max_seq_length=max_sequence_length,
        num_calibration_samples=num_calibration_samples,
    )
    model.save_pretrained(output_model.path)
    tokenizer.save_pretrained(output_model.path)

    return


@kfp.dsl.component(
    base_image="quay.io/opendatahub/llmcompressor-pipeline-runtime:main",
    packages_to_install=["lm_eval~=0.4.8", "vllm~=0.8.4"],
)
def eval_model(
    input_model: dsl.Input[dsl.Artifact],
    tasks: List[str],
    # TODO can model be of type `Literal["hf", "vllm"]`?
    model: str = "vllm",
    model_args: dict = {
        "add_bos_token": True,
        "dtype": "bfloat16",
        "device": "auto",
    },
    limit: Optional[int] = None,
    num_fewshot: int = 5,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    import lm_eval
    from lm_eval.utils import make_table

    model_args["pretrained"] = input_model.path

    results = lm_eval.simple_evaluate(
        model=model,
        model_args=model_args,
        limit=limit,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size or "auto",
    )
    print("lm_eval finished:\n", make_table(results))
    return make_table(results)


@kfp.dsl.pipeline(
    name="llmcompressor-oneshot",
    description="A demo pipeline to showcase how multiple recipes can be applied"
    " to a given model, followed by an eval step",
)
def pipeline(
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # "meta-llama/Llama-3.1-8B-Instruct",
    dataset_id: str = "HuggingFaceH4/ultrachat_200k",
):
    datafree_recipes: List[str] = [
        # TODO cannot pass in as type list annotation,
        #  do we need a more concrete base type for this to work?
        # QuantizationModifier(
        #     targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
        # ),
        """
        quant_stage:
            quant_modifiers:
                QuantizationModifier:
                    ignore: ["lm_head"]
                    targets: ["Linear"]
                    scheme: "W4A16"
        """,
    ]
    for recipe in datafree_recipes:
        oneshot_task = (
            run_oneshot_datafree(model_id=model_id, recipe=recipe)
            .set_cpu_request("3000m")
            .set_memory_request("4G")
            .set_cpu_limit("4000m")
            .set_memory_limit("5G")
        )
        kubernetes.use_secret_as_env(
            oneshot_task,
            secret_name="hf-hub-secret",
            secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
        )
        # Run on GPU, need to add toleration depending on RHOAI Cluster setup
        eval_task = (
            eval_model(
                input_model=oneshot_task.outputs["output_model"],  # noqa: F841
                tasks=["wikitext", "gsm8k"],
            )
            .set_accelerator_type("nvidia.com/gpu")
            .set_accelerator_limit("1")
            .set_cpu_request("3000m")
            .set_memory_request("4G")
            .set_cpu_limit("4000m")
            .set_memory_limit("5G")
        )
        kubernetes.add_toleration(
            eval_task,
            key="nvidia.com/gpu",
            operator="Equal",
            value="Tesla-T4-SHARED",
            effect="NoSchedule",
        )

    calibrated_recipes: List[str] = [
        """
        quant_stage:
            quant_modifiers:
                GPTQModifier:
                    ignore: ["lm_head"]
                    targets: ["Linear"]
                    scheme: "W4A16"
        """,
        # """
        # quant_stage:
        #     quant_modifiers:
        #         SmoothQuantModifier:
        #             smoothing_strength: 0.8
        #         GPTQModifier:
        #             ignore: ["lm_head"]
        #             targets: ["Linear"]
        #             scheme: "W4A16"
        # """,
    ]
    for recipe in calibrated_recipes:
        calibrated_task = (
            run_oneshot_calibrated(
                model_id=model_id,
                recipe=recipe,
                dataset_id=dataset_id,
                dataset_split="train_sft",
            )
            .set_accelerator_type("nvidia.com/gpu")
            .set_accelerator_limit("1")
            .set_cpu_request("3000m")
            .set_memory_request("4G")
            .set_cpu_limit("4000m")
            .set_memory_limit("5G")
        )
        kubernetes.use_secret_as_env(
            oneshot_task,
            secret_name="hf-hub-secret",
            secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
        )
        kubernetes.add_toleration(
            calibrated_task,
            key="nvidia.com/gpu",
            operator="Equal",
            value="Tesla-T4-SHARED",
            effect="NoSchedule",
        )
        eval_task = (
            eval_model(
                input_model=calibrated_task.outputs["output_model"],  # noqa: F841
                tasks=["wikitext", "gsm8k"],
            )
            .set_accelerator_type("nvidia.com/gpu")
            .set_accelerator_limit("1")
            .set_cpu_request("3000m")
            .set_memory_request("4G")
            .set_cpu_limit("4000m")
            .set_memory_limit("5G")
        )
        kubernetes.add_toleration(
            eval_task,
            key="nvidia.com/gpu",
            operator="Equal",
            value="Tesla-T4-SHARED",
            effect="NoSchedule",
        )


if __name__ == "__main__":
    # 1) compile to yaml, to upload to RHOAI
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline, package_path=__file__.replace(".py", ".yaml")
    )

    # 2) or run locally
    #  - in Docker (requires `pip install docker` & Docker or Podman installed)
    #  - or subprocess, using venv
    # kfp.local.init(
    #     runner=kfp.local.DockerRunner()
    #     # runner=kfp.local.SubprocessRunner(use_venv=True)
    # )
    # pipeline()
