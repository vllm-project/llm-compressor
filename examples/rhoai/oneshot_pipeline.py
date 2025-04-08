from typing import List, Dict, Any, Optional

import kfp
from kfp import dsl


@kfp.dsl.component(
    base_image="quay.io/opendatahub/llmcompressor-pipeline-runtime:main",
    packages_to_install=["llmcompressor~=0.5.0"],
)
def run_oneshot_datafree(
    model_id: str, recipe: str, output_model: dsl.Output[dsl.Artifact]
):
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = oneshot(model=model, recipe=recipe, tokenizer=tokenizer)
    model.save_pretrained(output_model.path)
    tokenizer.save_pretrained(output_model.path)

    return


# TODO
# def run_oneshot_calibrated(model_id: str, dataset_id: str, recipe: str, output_path: OutputPath):


@kfp.dsl.component(
    base_image="quay.io/opendatahub/llmcompressor-pipeline-runtime:main",
    packages_to_install=["lm_eval~=0.4.8"],
)
def eval_model(
    input_model: dsl.Input[dsl.Artifact],
    tasks: List[str],
    model: str = "hf",
    # model: Union[Literal["hf"], Literal["vllm"]] = "hf",
    model_args: dict = {
        "add_bos_token": True,
        "dtype": "bfloat16",
        "device": "cpu",
    },
    limit: Optional[int] = None,
    num_fewshot: int = 5,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    import lm_eval

    model_args["pretrained"] = input_model.path

    results = lm_eval.simple_evaluate(
        model=model,
        model_args=model_args,
        limit=limit,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size or "auto",
    )
    print("lm_eval finished: ", results["results"])
    return results["results"]


@kfp.dsl.pipeline(
    name="llmcompressor-oneshot",
    description="A demo pipeline to showcase how multiple recipes can be applied to a given model, followed by an eval step",
)
def pipeline(model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    recipe_map: Dict[str, str] = {
        # TODO cannot pass in as type list annotation, do we need a more concrete base type for this to work?
        # "FP8_DYNAMIC": [
        #     QuantizationModifier(
        #         targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
        #     )
        # ],
        "FP8_DYNAMIC": """
        quant_stage:
            quant_modifiers:
                QuantizationModifier:
                    ignore: ["lm_head"]
                    targets: ["Linear"]
                    scheme: "FP8_DYNAMIC"
        """,
        "W4A16": """
        quant_stage:
            quant_modifiers:
                QuantizationModifier:
                    ignore: ["lm_head"]
                    targets: ["Linear"]
                    scheme: "W4A16"
        """,
    }
    for _recipe_id, recipe in recipe_map.items():
        oneshot_task = run_oneshot_datafree(model_id=model_id, recipe=recipe)
        eval_model(
            input_model=oneshot_task.outputs["output_model"],  # noqa: F841
            tasks=["wikitext", "gsm8k"],
            # TODO don't run just 4 samples
            limit=4,
            batch_size=4,
        )


if __name__ == "__main__":
    # # 1) compile to yaml, to upload to RHOAI
    # kfp.compiler.Compiler().compile(
    #     pipeline_func=pipeline, package_path=__file__.replace(".py", ".yaml")
    # )

    # 2) or run locally
    #  - in Docker (requires `pip install docker` with Docker or Podman Desktop installed)
    #  - or subprocess, using venv
    kfp.local.init(
        runner=kfp.local.DockerRunner()
        # runner=kfp.local.SubprocessRunner(use_venv=True)
    )
    pipeline()
