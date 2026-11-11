# ResWAM 实验记录 / Iteration Log

日期：2026-11-11。所有结果在当前 CPU 运行环境中执行。**真实预训练 DINOv3 没有被加载**，不能将下表解读为真实机器人任务成功率。

## Iteration 0 — Static interface smoke

- 目的：验证模型内各张量是否能从 demo prefill → robot observation → Delta → FM 正确传播。
- 配置：`ResWAMConfig.tiny()`，`vision_dim=hidden_dim=32`，`K=4`，`action_horizon=4`，Fast Weights `16×16`。
- 结果：Short `[3,4,32]`、Goal `[3,4,32]`、FM Action `[3,4,2]`，非零反向梯度通过预测器与动作头；No Pending Cache。
- 限制：`load_vision=False`，不测试 DINOv3 权重。

## Iteration 1 — 普通两任务合成视频 ICL

- 设置：演示最后一帧直接显示目标；机器人当前画面两个任务相同；60 steps, batch 12, seed 7。
- Fast Weight：heldout short MSE=0.010132，goal MSE=0.012222。
- No Memory：heldout short MSE=0.021774，goal MSE=0.011023。
- 观察：两者都可读取显式演示目标；该任务**不能证明 TTT 长时记忆收益**。因此没有扩大训练规模，而是设计隐藏历史事件任务。

## Iteration 2 — 隐藏历史事件 (关键实验)

- 设置：演示首/尾帧完全中性；目标 `+1/-1` 只在演示中间帧出现，而显式缓存容量 `demo_context_limit=2` 只保留首/尾两帧。机器人当前画面完全相同。
- 每实验 250 steps, batch 24，三个种子 7、17、27，held-out 每种子 64 个合成实例。
- 所有模型采用相同的 Short/Goal 与 FM 任务监督；仅 `memory_mode=delta/none`。无记忆参数量略小，尚未做到严格等参数配平。

| Seed | Memory | Goal MSE | Goal 方向 | FM 动作方向 |
| --- | --- | ---: | ---: | ---: |
| 7 | delta | 0.002813 | 100% | 100% |
| 17 | delta | 0.001454 | 100% | 100% |
| 27 | delta | 0.002683 | 100% | 100% |
| 7 | none | 0.065073 | 51.56% | 51.56% |
| 17 | none | 0.064242 | 48.44% | 48.44% |
| 27 | none | 0.064864 | 48.44% | 48.44% |

三种子均值：`delta` 目标 MSE=0.002317、方向=100%；`none` 目标 MSE=0.064726、方向≈49.48%。

**解释：** 在这个严格人为构造的低维任务中，Fast Weight 记忆能保存已经从显式缓存被清除的任务事件；普通显式历史在此构造下无法辨识两个任务。该实验不证明 TTT-RM 的互补历史教师更好，也不证明真实视频任务性能。

原始日志：`runs/official/toy_occluded_*.json`。

## Iteration 3 — 从预计算特征输入的训练接口

- `scripts/make_toy_cache.py` 创建 6 条模拟 episode，每条 12 帧机器人观测与动作。
- `scripts/train_encoded.py` 完成 10 steps × 2 grad accumulation，成功输出 checkpoint 与逐步损失 JSON。
- 限制：仍是玩具特征；需要真实 RGB 与 DINO 预处理、仿真闭环评估。

## Test matrix

- `pytest`: **16 tests passing** （当前最终检查；后续以 `pytest -q` 为准）。
- 检查项目：DINO special token 处理（mock）、目标固定池化、因果时序无未来泄漏、跨 Segment 时间 Mask、在线记忆重置与快照、时间戳幂等保护、预测不存历史、FM 梯度路径、History Decoder 原型。
- 环境限制：torch CPU，初始未安装 timm。尝试安装受网络 DNS 限制失败；真实 DINO 权重测试待联网机器。

## 下一轮实验优先级

1. 用真实 timm DINOv3-S 运行 256×256 输入结构测试，记录 Patch Tokens、显存、时间与位置约束。
2. 接通真实配对视频 + 动作数据，复现无记忆/无 Delta 的有效闭环 baseline。
3. 检查真实 DINOv3 特征在视频遮挡/背景变化中的时间一致性。
4. 将离线 History Decoder 预训练与 TTT-RM 残差读出监督纳入 G3 trainer，并与原始 TTT-KVB 公平比较。
5. 比较早期压缩、后置条件压缩、因果全历史与预算缓存，确认真实延迟与成功率。

**强制迭代原则：方案和具体模块会随实验结论实时调整，而不是把当前结构视为正确答案。**

## Iteration 4 — 时间位置与 Segment Embedding 后的三种子复现（当前实现）

新增显式 ordinal sinusoidal temporal encoding 和 segment embedding，保持同段 Patch Temporal Attention 与跨段摘要注意力不变。新改动改变了计算，因此**未直接沿用 Iteration 2 的结果**，而是重新跑三种子六组 250-step 实验。

| Seed | Memory | Goal MSE | Goal 方向准确率 | 固定零初始噪声 FM 动作方向 |
|---|---|---:|---:|---:|
| 7 | delta | 0.001788 | 100% | 100% |
| 17 | delta | 0.002627 | 100% | 100% |
| 27 | delta | 0.001300 | 100% | 100% |
| 7 | none | 0.064888 | 51.56% | 48.44% |
| 17 | none | 0.066261 | 48.44% | 48.44% |
| 27 | none | 0.064565 | 48.44% | 51.56% |

三种子平均：`delta Goal MSE=0.001905`，`none Goal MSE=0.065238`；Goal 方向 `100% vs 49.48%`。Short MSE `0.004230 vs 0.020606`。数据保存在 `runs/official/posenc_occluded_*_seed*.json`，汇总 `posenc_occluded_summary.json`。

**局限仍然严格成立：** 这是人工潜在坐标的条件任务、确定性零初始噪声的 FM 方向判别；不等于采样分布质量或真实机器人闭环效果；Teacher 尚未接入正式训练，不代表完整 TTT-RM 复现。

## Iteration 5 — 真正 DINOv3 运行环境阻塞

- 代码使用用户指定 `timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=True, num_classes=0)`。
- 当前容器 `torch==2.10.0+cpu`，预装环境没有 `timm`，尝试安装时 PyPI 域名 DNS 解析失败。因此真实预训练权重、256x256 输入吞吐、GPU 延迟与真实 patch 特征没有被执行。
- 已使用模拟同接口 `forward_features` 的 Stub 验证特殊 token 剥离、冻结参数与模型名称调用；**不把 Stub 测试称为真实 DINOv3 测试**。
- 下次在可联网 NVIDIA GPU 的用户环境通过 `uv sync --extra dev` 与 `scripts/check_dinov3.py` 解除阻塞。
