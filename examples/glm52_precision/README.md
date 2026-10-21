# GLM-5.2 precision verification

**2026-09-23 数值对齐结论：mixed MXFP 尚未对齐 ModelSlim。** 独立源码对比发现
scale、中点舍入、小值策略和 BF16 边界差异；104 个 tensor 用例中 52 个存在差异，
4 个随机 linear 用例中 3 个输出不同。详见 [MXFP differential](../../tests/mxfp/README.md)。
下文的组合／导出往返测试通过，并不等同于跨框架量化结果一致。

## 本机已经验证的范围

校准 oneshot 的细分区设置及验证范围见
[sequential 分区说明](oneshot_partitions.md)。保持 input norm 与 attention 同段，
逐专家分区；已验证仅路由 token 校准时未命中专家的权重量化与导出。
DDP Modifier 接入及 CPU/disk 两 rank tiny E2E 已完成；先运行下面的单进程
服务器 smoke/profile，再根据结果推进多 rank，尚未验证八卡完整量化。

## L5 calibrated oneshot/profile

The single-process and distributed smoke driver is
[`oneshot_profile.py`](oneshot_profile.py). Read the
[run instructions, measurements and validation limits](oneshot_profile.md).
The distributed Modifier path now uses shared-backing QuaRot and mapping-level
Flex commits; global token caps remain deferred (`max_tokens=None` for smoke).

`mixed_mxfp.yaml` 使用独立的 QuaRotModifier → FlexSmoothModifier →
QuantizationModifier。官方 Transformers 随机 tiny GLM 的 basic / sequential
两条管线验证目标分配、变换后的权重 scale、动态激活 Q/DQ、浮点等价性和清理。
另有 FP32/BF16 压缩 checkpoint 保存与重载，以及 CPU/disk offload 对照。

- MXFP8：q_a/q_b、kv_a、o_proj、indexer.wq_b、dense MLP、shared experts。
- MXFP4：routed experts 的 gate/up/down。
- kv_b、indexer.wk/weights_proj、router 和 lm_head 不在量化目标中。
- group_size=32，激活动态 scale，权重静态 scale。

这只对应 ModelSlim 示例的 linear_quant 部分。没有实现 FA3/indexer-score 或
KV-cache 量化，也没有验证 NPU kernel、通用生产加载器/部署及真实模型精度。
FlexSmooth 的 INT8 搜索代理和最终 MXFP 格式不同，这是 reference 的算法设计。
该 recipe 的 max_tokens=128 是受控实验上限，并非已验证的生产校准设置。

压缩文件自测检查 MXFP4 packed uint8、MXFP8 float8 权重、E8M0 uint8 scale，
经实际 HF quantizer 解压后逐项比较权重、scale、量化配置和 logits。
重载采用测试内限定到 GLM 的 explicit experts 构造，且对解压权重显式统一 dtype。
CPU/disk 变换后的状态与常驻内存版本逐位一致；磁盘 Q/DQ 诊断使用与 sequential
pipeline 相同的 `disable_offloading()` 生命周期上下文，避免重复取权重时丢失临时 Q/DQ。
该上下文仅包围 tiny 诊断前向，不是让真实大模型全部驻留内存的建议。
具体依赖兼容缺口见 [runtime_limits.md](runtime_limits.md)。

本机完整检查（无需任何真实 checkpoint）：

```bash
python tools/run_quarot_checks.py --include-flex-smooth \
  --modelslim-source /path/to/msmodelslim \
  --report-dir /tmp/glm52-local-checks
```

## 现在请在真实服务器运行：布局预检

假设：现有 BF16 checkpoint 的 MLA 维度和参数命名与当前 adapter 相容。
先核实这个假设，才能选对单层实验、tensor 名称和 activation hook。

只需把 `tools/glm52_precision_probe.py` 复制到服务器，不必先安装本项目。
Python 3.9+ 标准库即可，不依赖 torch、NPU、ModelSlim 或联网。
指定服务器**已有的 BF16 目录**，不下载模型；输出路径放在模型目录外。

```bash
python glm52_precision_probe.py \
  --model-dir /path/to/existing/GLM-5.2-BF16 \
  --output /tmp/glm52-layout.json
```

如果在本仓库中运行，用 `python tools/glm52_precision_probe.py`。
默认选第 0 层及 config 中第一种不同 indexer 的层；缺少 indexer 类型时选 0/1。
可用 `--layers 0 10` 明确指定，最多四层，不默认假定某个层号有 full indexer。

脚本只读 config、safetensors index 和文件头。不会读取 tensor payload、加载模型、
修改 checkpoint 或执行推理。它支持标准 HF 单文件或分片 safetensors 命名；
其他命名会生成待检查错误，而不是搜索其他目录。

**请返回 `/tmp/glm52-layout.json`**，其中包括：

- 所选层、MLA/indexer/expert 配置、安装包版本。
- attention、norm、router、shared expert、routed expert 0 的 shape/dtype；
  packed expert 的整体 shape 只来自文件头。
- shape、block_size 和 baseline dtype 检查错误，以及需要人工核对的布局。

预期：`selected_shape_checks_passed=true`、`errors=[]`。退出码 0 表示所选基础
shape 检查通过，2 表示需要检查；两种情况都请返回 JSON。
`numerical_l4_passed` 始终为 false：这一步不能证明真实层数值正确。
indexer/expert 布局、MTP 排除和共享参数关系仍需结合返回信息确认。

## 后续单层 L4 的最小数据及判定

已提供 [L4 第一阶段数值脚本及运行说明](l4.md)：指定一层 attention 权重，
先执行合成输入预检查，再接入已有 BF16 推理流程采集 baseline cache。
此阶段验证 QuaRot/FlexSmooth 的独立 attention 子图，不等于以下完整单层目标均已完成。

布局确认后，选择 full/shared indexer 各一层（若存在），复用服务器推理流程采集
最多 128 tokens 的真实 activation cache；不进行整模型量化。
需要 input_layernorm、q_a_layernorm、kv_a_layernorm、o_proj 的输入，mask/positions，
对应 q_a/q_b、kv_a/kv_b、o_proj、norm、indexer 权重，以及一个 routed expert、
shared expert 和 router。记录 cache 处于原始 BF16 还是 QuaRot 后状态、dtype 和 seed；
不同变换阶段的 cache 不可混用。只返回选定 tensor 或统计，不传完整 checkpoint。

在相同 FP32 副本和输入上对比外部 ModelSlim 与 LLMC：rotation/fusion 中间权重、
alpha/beta、scale、42 个候选 loss、浮点输出、有限值计数和 router/top-k score margin。
初始判据：目标和 shape 精确对应；QᵀQ 最大误差 ≤1e-5；参数相对 L2 ≤1e-5；
浮点子图输出相对 L2 ≤1e-4；无 NaN/Inf；OV 的 K 行不变。近似并列 top-k 单独分析。
BF16 误差与独立的 FP32→BF16 round-trip baseline 对照，不能掩盖 FP32 失败。
单 expert 测试不等于全 MoE 等价性。满足相关 L4 判据之前不运行 L5。
