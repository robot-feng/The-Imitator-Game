# ResWAM v8.1 — Living Research Plan（动态实施方案）

> **本方案仅供研究与实现参考，非固定设计、非完成验收或性能承诺。**
> 后续必须根据真实实验结果、失败原因、效率评测与消融持续迭代完善优化。若发现新的设计不能提升闭环性能或导致训练/延迟恶化，应优先删除或回退，并在实验日志注明原因。该文件持续更新，历史方案保留在 `PLAN_v8_REFERENCE.md`。

日期：2026-11-11 ｜ 原型版本：0.1

## 1. 当前已锁定的 V1 约束

- 单次原生视频演示输入与后续机器人真实观测构成统一**因果**历史前缀。
- 视觉骨干是 `timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=True, num_classes=0)`，训练中冻结。
- 帧内 Spatial Attention 可双向，跨帧因果；跨人类演示/机器人实际历史**禁止相同 Patch Index 的时间注意力直接链接**，允许帧摘要跨段因果交互。
- 不在 DINO 后过早进行极端无条件 Resampler 压缩；Writer 接受真实 Patch 事件，Reader + 显式 Context 参与**后置任务条件化 Resampler**。
- TTT 只写已观测/确认事件；`Short/Goal Delta` 与未执行动作**不入记忆**。
- 视觉预测恰好两个目标帧：`Delta_short=Y(o_{t+h})-Y(o_t)` 与 `Delta_goal=Y(o_goal)-Y(o_t)`，不加其他视觉帧。
- 视觉监督坐标 `Y` 是稳定、与任务 Query 无关的固定空间投影。新任务演示与机器人画面不可直接按 Patch 做差。
- 下层使用标准 Flow Matching，默认 VTA/IDM；不使用 RollFlow、大型视频 Diffusion/VAE、VLM、预测历史缓存。
- 用机器人新的真实观测驱动下一轮重预测；V1 **没有**对历史预测误差执行在线梯度校准。

## 2. 当前阶段与已完成的实现

- `vision.py`：模型名与冻结 timm 官方接口（真实权重仍待外部 GPU 联网环境验证）。
- `causal_context.py`：帧内 Spatial + 同 Segment Patch Temporal + 跨 Segment Global Temporal。
- `memory.py`：单一 normalized Delta Fast Weight 和等量显式上下文；`memory_mode=none` 对照。
- `history_teacher.py`：仅离线可使用的历史 Decoder 和互补历史目标；目前还没有正式 Stage-G3 训练。
- `resampler.py`：后置、Short/Goal Query-Conditioned Cross-Attention。
- `residual.py`：共享 Transformer，一次前向只输出两帧 latent delta。
- `action_fm.py`：标准 FM `x_t=(1-t)epsilon+t*action`，Euler 积分采样。
- `model.py`：统一 observe/predict/reset，禁止错误 DEMO 数据字段、时间戳重复、机器人前预测、未来标签泄漏。
- `data.py` + 脚本：现实数据 manifest 与特征缓存入口；`train_encoded.py` 作为 G1/G2 最小训练接口。

当前不声称完成真实相机/执行器的 70ms 在线闭环；小型 toy 结果是功能性指标。

## 3. 动态迭代 Gate（不可跳过）

### G0：可信数据与执行接口

接入现有 SynthICL/Imitator Game / LIBERO，排查相机 resize、动作归一化、执行时间步、任务评估，尤其对已有 0/60 现象提供明确原因。通过前不得宣称新策略成功率。

### G1：可靠 Video ICL

测试单次演示、同场景不同演示、相机/背景更换、演示交换，确认任务信息确实影响 FM 动作。检查 Prefix 与 Streaming 一致性。

### G2：双帧 ResWAM

对照 Goal only / Short only / 2 delta / No delta / Oracle Delta；保证 FM 训练和部署都使用模型预测的两个 Delta。视觉目标必须固定坐标并在训练时真实可得。

### G3：TTT-RM 原理验证

训练并冻结 History Decoder，在 Current+Explicit Context 无法恢复历史的条件下构造历史残差，使用与推理一致的历史 Query 训练 Memory Writer/Reader，比较单矩阵、普通 KV、等容量 Linear Attention、显式上下文；报告 TBPTT 代价与 Demo→Robot 梯度链。

### G4：真实闭环与信息流消融

先 VTA/IDM，再改轻量 Cross Attention；若实验支持再考虑真正联合 S/G/A 交互。报告成功率、动态扰动、p50/p95、峰值 VRAM、GPU-hours。

## 4. 决策与回退规则

1. 每轮保存 `seed/config/dataset/model checksum/test protocol/result`；不隐藏失败种子。
2. 修改模块前保留独立基线，并在同数据/预算下比较。
3. 发现时间泄漏、训练/推理不一致或失效的 Fast Weight 更新，立即回退并记录。
4. 如果 DINO Latent Delta 因遮挡/视角导致坐标无效，优先修正视觉监督定义，不急着加大网络。
5. 如果 online memory 没有显著优于 bounded explicit context，考虑更简单替代，而不是为了架构好看保留 TTT。
6. 如果上层目标准确但动作质量低，单独测试 FM Head；不要用动作成功率误判世界预测。
7. 保留本文件 `PLAN.md` 作为实时修订主文档；`PLAN_v8_REFERENCE.md` 为原始讨论快照。

## 5. 迭代实验入口

- G0 结构检验：`uv run pytest -q && uv run python scripts/smoke.py`
- G2 Toy：`uv run python scripts/train_toy.py --task occluded --steps 250 --memory delta`
- 无记忆对照：同上 `--memory none`
- 真实特征缓存：参考 `docs/DATA_PROTOCOL.md`。
- 实验结果记录：`docs/EXPERIMENT_LOG.md` 与 `runs/official/*.json`。

**所有方案应根据实验结果持续完善。当前版本只是可验证的工作假说。**
