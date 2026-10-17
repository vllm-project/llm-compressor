# 服务器 L4 第二阶段：完整单层 QuaRot → FlexSmooth

本轮验证完整 decoder layer 的浮点等价性、与 ModelSlim 的差分，以及两个真实
Modifier 的串联生命周期。先前的 attention 子图 L4 结果继续保留。
这次不修改 MXFP 规则，也不执行整模型量化或精度评测。

本机无真实权重：完整 L0–L3/交接回归 181 passed、0 skipped（42.77 秒），其中新增
13 项验证这套整层 capture/replay，包括合成 dense/MoE、BF16 cache 和失败清理。
这些是脚本就绪的证据，服务器整层数值结果仍待返回。

## 假设与范围

1. QuaRot 对真实 dense MLP、全部 routed/shared experts、router 和 attention 的
   变换，与原始 ModelSlim 函数一致，完整残差输出保持等价。
2. 完整 attention 的 RoPE、softmax、indexer top-k 与连续 router logits 在 FP32
   变换前后保持等价；共享 indexer 使用同一份上游索引。稀疏 MoE 的中间态若因
   top-k cutoff 附近的 near-tie 发生 expert 离散翻转，在 router logits 仍通过原
   浮点阈值、非 routing-dependent trace 全部通过、ModelSlim 差分通过且最终组合
   expert selection 恢复严格一致时，记录为 routing sensitivity 诊断而非实现失败。
3. FlexSmooth 在 QuaRot 后的真实 forward 上采集激活，完成搜索、缩放、事件收尾，
   结果与 ModelSlim 在独立旋转副本上采集的激活/变换一致。

执行 HF `GlmMoeDsaDecoderLayer` 的原始 forward；每次只读取所选层的完整参数，
包括 indexer LayerNorm 和 router correction buffer。所有 expert 权重均参与比较；
forward 仅覆盖当前输入实际路由到的 expert，报告列出这些编号。
缺失任一 expert、出现未知层内参数或 packed expert 布局时直接拒绝，不降级抽样。

为使用公开 Modifier API，在所选层前后添加合成 identity 输入/输出适配器，旋转后
恢复相同残差坐标。因此这不验证 checkpoint 的 embedding、最终 norm、lm_head 或 MTP。
共享层的 indexer producer 不在该次回放中；通过缓存传入它的原始 top-k。

## 1. 在现有 baseline 推理脚本中重新采集

旧 `glm52-layer*-baseline-real.safetensors` **不能直接复用**：它没有完整 mask、
RoPE、共享 indexer 索引和整层输出，也在 layer 3 的 MLP 之前停止。

继续使用服务器已有的模型加载、tokenizer 和 offload 代码。用当前分支完整仓库，
包含 `tools/` 和 `tests/`；不要只复制单个脚本。模型保持未量化、未变换、`eval()`，
加载时设置 `attn_implementation="eager"`，expert 保持 LLMC 的显式 Linear 布局。
删除上次安装在 `o_proj` 上的 early-stop hook。在原来执行 forward 的地方调用：

```python
from tools.glm52_layer_l4_capture import capture_prefill

# model、inputs、MODEL_DIR 沿用你现有脚本。
# inputs 是一个真实 prompt 的 tokenized tensor dict，batch=1，无 padding。
# 维持现有 offload execution context，不能在这里退出原上下文。
paths = capture_prefill(
    model,
    inputs,
    model_dir=MODEL_DIR,
    output_dir="/tmp/glm52-complete-l4-cache",  # 新目录，不能覆盖旧文件
    layers=(0, 3),
    input_kind="real",
    input_description="填写真实 prompt/数据来源、tokenizer 与加载脚本版本",
    max_tokens=128,
    max_weight_gib=40,
)
print(paths)
```

helper 不加载模型或 tokenizer。它会运行这一次已有输入的 baseline prefill，
在 layer 3 **整个 MLP 和残差输出完成后**停止，不继续后续层或 LM head。
已有加载流程可能仍构造完整模型；这是沿用服务器流程，不是本地开发要求。
如只先验证 dense 层，设置 `layers=(0,)`、`max_weight_gib=2`。

每个 cache 保存完整 `[1, sequence, hidden]` 输入/输出、additive causal mask、
position_ids、RoPE cos/sin、top-k，以及共享层的 prev_topk_indices；保留采集 dtype。
不会扁平化截断 attention 上下文。要求 `use_cache=False`、一次连续 prefill；未知额外
forward kwargs、padding、自定义 mask、decode、第二次 forward 会明确拒绝。
权重在 capture 前后计算指纹，replay 再核对 checkpoint 的同名权重。

18-token prompt 足以先做串联回归，但小于真实 `index_topk=2048`，报告会标记
`indexer_selective=false`。它不能证明稀疏筛选充分覆盖。后续若需要单独验证该分支，
使用 2049..4096 token 的连续真实 prompt，并显式提高 `max_tokens` 和 replay 内存预算；
Flex 校准仍最多使用前 128 token，整层 forward 保留全部上下文。先跑短输入。
cache 读取另有 512 MiB 的保守分配上限，部分配置的 4096-token cache 会超限；
4096 是序列上限，不保证任何形状都可接收。超限时返回错误，先用更短的连续输入。

## 2. 独立回放：先 layer 0，再 layer 3

在仓库根目录执行，替换路径。没有任何下载。ModelSlim 源码基线为
`beb917d011bc1d11a8bf5e46f62928f6d84b06a7`。需要与第一阶段相同的可运行环境。
若换了仓库目录，先在该目录执行 `python -m pip install --no-deps --no-build-isolation -e .`，
使当前环境导入这份 checkout；脚本会检查路径，避免实际运行到旧安装。

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MODEL_DIR=/path/to/existing/GLM-5.2-BF16
MS_SOURCE=/path/to/existing/msmodelslim
CACHE=/tmp/glm52-complete-l4-cache
OUT=/tmp/glm52-complete-l4-results
mkdir -p "$OUT"
set -o pipefail

python tools/glm52_layer_l4.py \
  --model-dir "$MODEL_DIR" --modelslim-source "$MS_SOURCE" --layer 0 \
  --activation-cache "$CACHE/glm52-layer0-complete.safetensors" \
  --max-weight-gib 2 --max-estimated-memory-gib 16 \
  --dry-run --output "$OUT/layer0-plan.json"

python tools/glm52_layer_l4.py \
  --model-dir "$MODEL_DIR" --modelslim-source "$MS_SOURCE" --layer 0 \
  --activation-cache "$CACHE/glm52-layer0-complete.safetensors" \
  --max-weight-gib 2 --max-estimated-memory-gib 16 \
  --output "$OUT/layer0.json" 2>&1 | tee "$OUT/layer0.log"
```

dry-run 仅读权重 header 和小 cache，不读权重 payload。确认 `layer0.json` 的
`full_layer_l4_passed=true` 后，再把上述两条命令分别改为 layer 3、对应 cache/输出名、
`--max-weight-gib 40 --max-estimated-memory-gib 192`。不要覆盖第一次的结果。

实际配置下 layer 0 完整 FP32 权重约 1.5 GiB；layer 3 有 256 个 routed experts，
约 36.8 GiB。后者明显比上次 attention-only 慢；原实现会对全部专家执行旋转。
内存估计包含多个副本、临时计算及 attention 概率，不是 OS 硬限制；192 GiB 是显式
预算而非测得峰值。先看 dry-run 的估计和服务器剩余资源，再运行。
如果完整 MoE 层超预算，返回 plan/error，不能删 expert 或把抽样结果标成完整层通过。

## 3. 返回结果与判定

请返回两个 JSON、两个运行 log、两个新 cache，以及实际 capture driver 源码和
`git rev-parse HEAD` / `git diff`。不传真实层权重或完整 checkpoint。

报告包含源文件哈希、Git 状态、依赖版本、cache 哈希和选层参数清单；逐参数比较
QuaRot 后、串联后权重；比较 attention/MLP/残差输出、softmax 概率、router logits/
路由权重/专家集合和 indexer top-k 集合；记录真实 Modifier hook/cache 清理结果。
还保存 baseline/post-QuaRot 校准输入哈希、scale 统计、alpha/beta 和 42 个候选 loss。

通过标准：

- 逐权重、scale、两边校准输入相对 L2 ≤ 1e-6；所有值有限。
- 浮点 forward 及其坐标关系相对 L2 ≤ 1e-4；最终 composition 的离散
  top-k/专家集合必须完全一致。中间 sparse-MoE 仅在连续 router logits 通过原阈值
  且失败严格局限于 routing-dependent trace 时允许记录 near-tie sensitivity；
  不扩大任何数值阈值。
- alpha/beta 一致；相同输入/权重下候选 loss **完全相同**。两条独立 FP32 旋转链的
  candidate-loss 相对 L2 ≤ 1e-5，另保留最大绝对误差；GEMM 舍入差异可传播到搜索。
- Modifier 捕获值与旋转后 forward 的实际输入完全相同，清理完成，状态 applied。

`baseline_capture_diagnostic` 比较 captured dtype 输出与 CPU FP32 回放，仅作诊断；
BF16→FP32 的计算差异不冒充变换误差。中间 sparse-MoE 的 near-tie 路由变化单独记录
`routing_sensitive`；最终 composition 的离散选择不同、连续 router logits 超阈值或
搜索结果不同仍应判失败，不直接扩大阈值。退出码：0=真实 cache 的本轮通过（或 dry-run 成功），
1=数值失败，2=前置条件/执行错误，3=合成 cache 测试通过。必须同时看 dry_run 字段。

`full_layer_l4_passed` 仅指所选层/输入的本轮范围，不表示所有层、BF16/NPU、decode、
offload 或最终精度已验证。短输入的非选择性 indexer、未命中 experts 和合成边界仍是
显式限制；返回结果审核后再决定下一步，不自动建议 L5。

## MXFP 后续范围

保留 LLMC 当前原生 MXFP4/MXFP8 规则。此前发现的 ModelSlim 跨框架舍入/scale 选择
差异仍记录为“非逐数值一致”，不在本轮追求强制对齐，也不作为 QuaRot/FlexSmooth
实现错误。后续仍须验证配置及目标匹配、导出参数/打包、重载、目标运行时部署和任务
精度；仅生成文件不等于通过。这些不由本次浮点 L4 代替。
