"""ResWAM configuration; one source of truth for experiments."""
from dataclasses import dataclass, asdict


@dataclass
class ResWAMConfig:
    # 默认使用用户指定的 timm DINOv3-S 模型，不更换骨干。
    vision_model: str = "vit_small_patch16_dinov3.lvd1689m"
    vision_pretrained: bool = True
    vision_dim: int = 384
    hidden_dim: int = 384
    heads: int = 6
    target_hw: int = 4                 # 4x4=16 个空间坐标固定的目标 tokens
    causal_depth: int = 2             # 每层 Spatial -> Temporal -> Global
    prediction_depth: int = 4
    fm_depth: int = 3
    ff_mult: int = 4
    demo_context_limit: int = 8
    robot_context_limit: int = 4
    n_memory_reads: int = 8
    memory_dim: int = 96
    memory_lr: float = 0.08
    memory_mode: str = "delta"         # "delta" | "none"
    fm_steps: int = 8
    action_dim: int = 7
    action_horizon: int = 8
    proprio_dim: int = 7
    short_horizon: int = 4             # 训练时短期未来标签的帧间距
    weight_fm: float = 1.0
    weight_short: float = 1.0
    weight_goal: float = 1.0
    weight_memory: float = 0.0         # G3 开始使用离线 teacher 监督
    dropout: float = 0.0

    def validate(self) -> None:
        if self.hidden_dim != self.vision_dim:
            raise ValueError("第一版要求 hidden_dim == vision_dim，以免残差监督坐标发生隐式漂移")
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim 必须能被 attention heads 整除")
        if self.target_hw < 1 or self.action_horizon < 1 or self.fm_steps < 1:
            raise ValueError("target_hw/action_horizon/fm_steps 必须 > 0")
        if self.demo_context_limit < 1 or self.robot_context_limit < 1:
            raise ValueError("显式上下文长度必须 > 0")
        if self.memory_mode not in {"delta", "none"}:
            raise ValueError("不支持的 memory_mode")
        if self.memory_dim < 1 or self.n_memory_reads < 1:
            raise ValueError("memory_dim/n_memory_reads 必须 > 0")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def tiny(cls) -> "ResWAMConfig":
        """只用于离线功能测试。正式默认仍使用 DINOv3-S / d=384。"""
        return cls(
            vision_pretrained=False, vision_dim=32, hidden_dim=32, heads=4,
            target_hw=2, causal_depth=1, prediction_depth=2, fm_depth=2,
            demo_context_limit=3, robot_context_limit=3,
            n_memory_reads=4, memory_dim=16, action_dim=2,
            action_horizon=4, proprio_dim=2, fm_steps=3,
        )
