# MXFP4 ablation reproduction index

Use the [`mxfp4-ablation-reproduction` skill](../../.agents/skills/mxfp4-ablation-reproduction/SKILL.md)
for the complete environment and workflow. Method-specific settings and
commands are documented in these guides:

| Method | Scheme | Runner | Detailed guide |
|---|---|---|---|
| Imatrix | MXFP4A16 | [`run_mxfp4a16_imatrix_ablation.py`](run_mxfp4a16_imatrix_ablation.py) | [`MXFP4A16_IMATRIX_REPRODUCTION.md`](MXFP4A16_IMATRIX_REPRODUCTION.md) |
| Imatrix | MXFP4xFP8 | [`run_mxfp4xfp8_imatrix_ablation.py`](run_mxfp4xfp8_imatrix_ablation.py) | [`MXFP4XFP8_RESEARCH_REPRODUCTION.md`](MXFP4XFP8_RESEARCH_REPRODUCTION.md) |
| Scope-targeted imatrix | MXFP4xFP8 | [`run_mxfp4xfp8_imatrix_ablation.py`](run_mxfp4xfp8_imatrix_ablation.py) with `MXFP4_IMATRIX_SCOPE_ABLATION=1` | [`MXFP4XFP8_RESEARCH_REPRODUCTION.md`](MXFP4XFP8_RESEARCH_REPRODUCTION.md) |
| GPTQ + imatrix | MXFP4xFP8 | [`run_mxfp4xfp8_gptq_imatrix_ablation.py`](run_mxfp4xfp8_gptq_imatrix_ablation.py) | [`MXFP4XFP8_RESEARCH_REPRODUCTION.md`](MXFP4XFP8_RESEARCH_REPRODUCTION.md) |
| Model-free MSE | MXFP4A16 or MXFP4xFP8 | [`run_mxfp4_mse_model_free_ablation.py`](run_mxfp4_mse_model_free_ablation.py) | [`MXFP4XFP8_RESEARCH_REPRODUCTION.md`](MXFP4XFP8_RESEARCH_REPRODUCTION.md) |

The A16 imatrix runner and guide were merged in llm-compressor PR #3477. This
PR adds the xFP8 imatrix, GPTQ+imatrix, and MSE reproduction paths, including
MSE support for both activation schemes. Generated results are deliberately
kept out of the source tree; each runner writes summaries and full run records
to its selected run directory.
