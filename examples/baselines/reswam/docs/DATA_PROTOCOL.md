# 真实数据接入协议（V1）

本项目只提供抽象的数据接口，**没有擅自读取/覆盖用户本地的 SynthICL、LIBERO、Imitator Game 或机器人仓库**。
训练时由用户提供每一任务配对的参考演示、机器人状态和执行动作；可以从用户现有的数据适配器转换到下面格式。

## 1. manifest.json 示例

```json
{
  "episodes": [
    {
      "id": "task_001_ep0",
      "demo_frames": ["demo/000.png", "demo/001.png"],
      "robot_frames": ["robot/000.png", "robot/001.png", "robot/002.png"],
      "actions_path": "action.npy",
      "proprio_path": "proprio.npy"
    }
  ]
}
```

所有路径都相对 manifest 所在目录。

- `demo_frames`：至少 2 帧任务示范；可以是人类演示，不要求演示动作。
- `robot_frames`：训练时至少 `max(short_horizon+1, action_horizon)` 以上的机器人观测；默认以**成功轨迹末帧**为 Goal；复杂任务建议提供更严格的 Goal 标注与完成事件。
- `actions_path`：真实机器人标签 `[T, action_dim]`，与机器人每个观测时间步对齐。如果动作与相机不同频，需要在上游完成对齐，不能简单按索引拼接。
- `proprio_path`：机器人本体状态 `[T, proprio_dim]`。
- `Action Chunk`：训练代码取 `actions[t:t+H]`，仅使用正确的 robot 控制标签。
- `Short target`：`robot_features[t+h_s]`。
- `Goal target`：当前成功机器人轨迹末帧，与当前观察在同一稳定空间坐标中。

## 2. 生成真实 DINOv3 特征缓存

```bash
uv sync --extra dev
uv run python scripts/cache_features.py \
  --manifest /path/to/manifest.json \
  --output runs/official/dino_features.pt
```

将使用 `vit_small_patch16_dinov3.lvd1689m`，`pretrained=True`；首次运行需要从 Hugging Face 下载预训练权重，遵守模型许可证。RGB 统一交给 timm 官方预处理，不要改用自己猜测的 Normalize / Resize。输入 `256x256` 时输出剥离 5 个特殊 token 后的 `256x384` patch features。

`--no-pretrained` 仅用于测试代码结构，**不允许拿它的结果宣称 DINOv3 预训练成功**。

## 3. 从特征缓存训练

```bash
uv run python scripts/train_encoded.py \
  --data runs/official/dino_features.pt \
  --steps 1000 \
  --grad-accum 4 \
  --output runs/official/reswam_g2.pt
```

当下版本 `train_encoded.py` 是最小 G1/G2 训练管线，并不包含历史教师完整训练及 LIBERO 闭环成功率。特征缓存读取只接受可信本地文件。

## 4. 必须由真实数据适配器明确的数据语义

- 演示与机器人历史属于不同 segment，**不按相同 Patch ID 强行做跨本体运动对应**。
- 未执行的预测动作不能进入 TTT Writer。
- 不把未来图像当作本轮输入，只作为离线监督目标。
- `action[t]` 是当前时刻以后实际要执行/已记录的标签，`previous_executed_action` 是进入当前观测之前已发生的动作；两者绝不可混用。
- 本 V1 无预测历史缓存、无推理时在线预测误差反传。
