# 本地压缩重载验证与 upstream review 记录

检查环境：torch 2.14.0+cpu、Transformers 5.17.0、compressed-tensors
0.19.1a20260919；随机 tiny GLM，无真实 checkpoint。

## 已修复：config 重存丢失变换状态

真实 checkpoint 的原始 config 通常包含 dtype。此前 `resave_config` 成功恢复原始
config 时，只更新 dtype、embedding tie 和 expert count，丢掉新增的
`quarot_config` / `flex_smooth_config`。重载后无法可靠阻止二次变换。
旧 tiny 配置恰好缺少 dtype，触发保留 Transformers config 的 fallback，掩盖了问题。

现在明确保留这两个状态字段。独立回归覆盖 applied/failed、保留原始字段和不改写
源 config；正式 tiny 流程包含原始 dtype 字段，检查保存重载后 diagnostics 完整、
新 Modifier 拒绝重复初始化。该修复不改变量化格式或变换算法。

建议该逻辑以独立修复提交 review；其余测试、文档和现有 Modifier 分开评审。
尚未创建或发布上游 PR。

## 已验证的压缩文件边界

basic/sequential × FP32/BF16：`save_pretrained(save_compressed=True)` 生成实际
safetensors，检查 packed 权重 shape/dtype、每个 E8M0 exponent、保存的 compressed
状态。使用实际 Transformers/CT quantizer 解压，无 missing/unexpected keys。
解压后权重等于保存前 Q/DQ 权重、scale 完全相等、动态 activation 配置保留，
logits 逐位相同。这里比较的是同一量化函数的 IO 往返，不是量化前后精度不下降。

以下依赖行为必须保留在报告中，不能把受控测试等同于通用加载器通过：

1. **MoE 通用加载器**：`load_quantizable_moe` 通过 Transformers 全局
   `apply_patches` 扫描模块；5.17.0 中 `dir(module)` 会触发无关的 aria 图像模块导入，
   在当前 CPU 环境报 `ModuleNotFoundError: torchvision`。异常发生于真正读取权重之前。
   测试 `_LinearizedGlm` 只在构造时替换 GLM expert 类，退出即还原；没有修改依赖源码，
   没有跳过实际 HF/CT 文件加载。通用 context manager 的兼容性仍需单独解决。
2. **FP32 解压 dtype**：CT 的 MX 解压器将部分权重还原为 BF16，即便 HF 请求 FP32。
   直接前向会遇到 float/BFloat16 不匹配。测试显式执行 `restored.to(dtype)`，并记录
   统一 dtype 前的权重类型。没有放宽逐 tensor 或 logits 的精确往返要求。
3. **裸 DiskCache Q/DQ 前向**：CT quantized forward 临时修改第一次读取到的
   weight.data，但 nn.Linear 再次读取 weight 时，DiskCache 会重新加载原始权重。
   因而没有缓存上下文时，输出与常驻模型不同，尽管磁盘权重/scale 完全相同。
   `disable_offloading()` 保持单次前向使用同一对象；这是 LLMC sequential pipeline
   已使用的上下文。测试采用该上下文并检查退出后缓存清空。裸调用仍不是本次支持承诺。

后续 upstream review 应将第 1 项归到 Transformers/LLMC loader 边界，第 2/3 项归到
compressed-tensors 的解压和 offload/QDQ 边界；不要在 FlexSmooth 算法内部加入补丁。
真实服务器的 kernel、显存规划、生产 loader 和 L4/L5 仍需单独验证。

## BF16 与 offload 的判据

BF16 变换会产生存储舍入：tiny FP32 严格代数门禁保持不变，BF16 综合变换回归另设
relative L2 < 0.02、max absolute error < 0.02，不将其称为严格数学等价或真实精度。
压缩往返则保持误差 0。CPU/disk 与常驻版本的同 dtype 权重、量化 scale、搜索结果
和有缓存上下文的量化 logits 均要求误差 0。
