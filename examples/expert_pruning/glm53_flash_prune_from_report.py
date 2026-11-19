from compressed_tensors.entrypoints.convert import convert_checkpoint

from llmcompressor.entrypoints.converter import ExpertPruner

MODEL_ID = "RedHatAI/GLM-5.3-Flash-NVFP4"
REPORT_PATH = "glm53_flash_razorcal_report.json"
SPARSITY = 0.20

pruner = ExpertPruner.from_pretrained(
    MODEL_ID,
    sparsity=SPARSITY,
    metric="saliency",
    uniform=True,
    saliency_report_path=REPORT_PATH,
    router_pattern=(
        r"model\.language_model\.layers\.(?:[3-9]|[1-3][0-9]|4[0-4])\.mlp\.gate\.weight"
    ),
    expert_pattern=(
        r"model\.language_model\.layers\.(?:[3-9]|[1-3][0-9]|4[0-4])"
        r"\.mlp\.experts\.\d+\.(?:gate|up|down)_proj"
    ),
    # per-expert routing bias, sliced along dim 0 like the router weight rows
    aux_pattern=(
        r"model\.language_model\.layers\.(?:[3-9]|[1-3][0-9]|4[0-4])"
        r"\.mlp\.gate\.e_score_correction_bias"
    ),
)

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + f"-RAZOR{int(SPARSITY * 100)}"
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converter=pruner,
)
