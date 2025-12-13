from tempfile import TemporaryDirectory

from lm_eval.evaluator import simple_evaluate

from llmcompressor.modifiers.awq.convert_autoawq import convert_and_save
from tests.testing_utils import requires_gpu


def run_lm_eval(model_name_or_path: str):
    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_name_or_path},dtype=float16",
        tasks=["arc_challenge", "arc_easy"],
        num_fewshot=5,
        batch_size=16,
    )

    return results


def compare_models(model_name_or_path: str):
    autoawq_result = run_lm_eval(model_name_or_path)
    with TemporaryDirectory() as converted_model_dir:
        convert_and_save(model_name_or_path, converted_model_dir, "naive-quantized")
        converted_result = run_lm_eval(converted_model_dir)

    arc_c_autoawq = autoawq_result["results"]["arc_challenge"]["acc_norm,none"]
    arc_c_converted = converted_result["results"]["arc_challenge"]["acc_norm,none"]
    arc_e_autoawq = autoawq_result["results"]["arc_easy"]["acc_norm,none"]
    arc_e_converted = converted_result["results"]["arc_easy"]["acc_norm,none"]

    assert abs(arc_e_autoawq - arc_e_converted) < 1e-2, (
        f"Arc Easy: autoawq={arc_e_autoawq} != converted={arc_e_converted}."
    )
    assert abs(arc_c_autoawq - arc_c_converted) < 1e-2, (
        f"Arc Challenge: autoawq={arc_c_autoawq} != converted={arc_c_converted}."
    )


@requires_gpu
def test_mistral():
    compare_models(
        "fbaldassarri/mistralai_Mistral-7B-Instruct-v0.3-autoawq-int4-gs128-asym"
    )


@requires_gpu
def test_qwen():
    compare_models(
        "ruikangliu/DeepSeek-R1-Distill-Qwen-1.5B-quantized.awq-autoawq-w4g128"
    )


@requires_gpu
def test_llama():
    compare_models("AMead10/Llama-3.2-3B-Instruct-AWQ")
