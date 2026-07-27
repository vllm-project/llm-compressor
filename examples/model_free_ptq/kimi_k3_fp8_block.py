from compressed_tensors.entrypoints.convert import CompressedTensorsDequantizer

from llmcompressor import model_free_ptq

MODEL_ID = "moonshotai/Kimi-K3"
SAVE_DIR = "/mnt/nvme-data/engine/kylesayrs/" + MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"

ignore = [
    "re:.*lm_head.*",
    "re:.*vision_tower.*",
    "re:.*mm_projector.*",
    "re:.*mm_projector.*",
    "re:.*block_sparse_moe\.gate.*",
    "re:.*self_attention_res_proj.*",
    "re:.*self_attn.*",
    #"re:.*self_attn.b_proj.*",
    #"re:.*self_attn.b_proj.*",
    #"re:.*self_attn.f_a_proj.*",
]

# language_model.model.layers.74.self_attention_res_proj.weight                          [1, 7168]          F8_E4M3 
# language_model.model.layers.74.self_attn.b_proj.weight                                 [96, 7168]         F8_E4M3 YES
# language_model.model.layers.74.self_attn.f_a_proj.weight                               [128, 7168]        F8_E4M3 YES
# language_model.model.layers.74.self_attn.f_b_proj.weight                               [12288, 128]       F8_E4M3 YES
# language_model.model.layers.74.self_attn.g_proj.weight                                 [12288, 7168]      F8_E4M3 YES
# language_model.model.layers.74.self_attn.k_proj.weight                                 [12288, 7168]      F8_E4M3 YES
# language_model.model.layers.74.self_attn.o_proj.weight                                 [7168, 12288]      F8_E4M3 YES
# language_model.model.layers.74.self_attn.q_proj.weight                                 [12288, 7168]      F8_E4M3 YES
# language_model.model.layers.74.self_attn.v_proj.weight                                 [12288, 7168]      F8_E4M3 YES

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="FP8_BLOCK",
    ignore=ignore,
    converter=CompressedTensorsDequantizer(
        MODEL_ID,
        ignore=ignore,
    ),
    max_workers=5,
    device=[
        f"cuda:{i}"
        for i in range(6)
    ],
)
