"""ResWAM V1: causal observed history -> TWO residual frames -> standard FM.

No prediction cache, no video generator, no VLM, no executed-action hallucination.
"""
from __future__ import annotations
import torch
from torch import Tensor,nn
import torch.nn.functional as F
from .config import ResWAMConfig
from .vision import FrozenDINOv3,FixedSpatialTarget
from .causal_context import BoundedFrameHistory,CausalSpatiotemporalEncoder
from .memory import ResidualFastMemory,NullMemory
from .resampler import TaskConditionedResampler
from .residual import TwoFrameResidualPredictor,TwoFrameDeltas
from .action_fm import StandardFlowMatchingPolicy


class ResWAM(nn.Module):
    def __init__(self,cfg:ResWAMConfig|None=None,*,load_vision:bool=True,vision_factory=None):
        super().__init__()
        self.cfg=cfg if cfg is not None else ResWAMConfig()
        self.cfg.validate()
        c=self.cfg
        self.vision=(FrozenDINOv3(c.vision_model,pretrained=c.vision_pretrained,
                                  model_factory=vision_factory) if load_vision else None)
        if self.vision is not None and self.vision.feature_dim!=c.vision_dim:
            raise ValueError(f"DINOv3 feature dim = {self.vision.feature_dim}, config={c.vision_dim}")
        self.fixed_target=FixedSpatialTarget(c.target_hw)
        self.encoder=CausalSpatiotemporalEncoder(c.hidden_dim,c.heads,
                   depth=c.causal_depth,mlp_ratio=c.ff_mult,dropout=c.dropout)
        if c.memory_mode=="delta":
            self.memory=ResidualFastMemory(c.hidden_dim,c.memory_dim,c.proprio_dim,
                c.action_dim,c.n_memory_reads,c.memory_lr)
        else:
            self.memory=NullMemory(c.memory_dim,c.n_memory_reads)
        self.memory_to_hidden=nn.Sequential(nn.LayerNorm(c.memory_dim),nn.Linear(c.memory_dim,c.hidden_dim))
        self.resampler=TaskConditionedResampler(c.hidden_dim,c.heads,c.target_hw**2)
        self.predictor=TwoFrameResidualPredictor(c.hidden_dim,c.heads,c.prediction_depth,
                                                 c.target_hw**2,c.ff_mult,c.dropout)
        self.fm=StandardFlowMatchingPolicy(c.hidden_dim,c.heads,c.fm_depth,
                  c.action_dim,c.action_horizon,c.proprio_dim,c.fm_steps)
        self.reset_task()

    def reset_task(self):
        """New task; only confirmed external observations are allowed afterward."""
        self.history=BoundedFrameHistory(self.cfg.demo_context_limit,self.cfg.robot_context_limit)
        self.fast_state:Tensor|None=None
        self.seen_robot=False
        self.latest_proprio:Tensor|None=None
        self.last_observed_timestamp={"demo":None,"robot":None}
        self.n_observations=0

    def end_demo(self):
        """Declare demo segment completed; no fake transition from demo to robot."""
        self.history.segment_closed=True

    def observe_image(self,image:Tensor,*,segment:str,grid=None,
                      proprio:Tensor|None=None,executed_action:Tensor|None=None,
                      timestamp:float|None=None):
        if self.vision is None:
            raise RuntimeError("此模型以 load_vision=False 初始化，只能用 observe_features")
        patches,inferred_grid=self.vision(image)
        self.observe_features(patches,grid=inferred_grid,segment=segment,
             proprio=proprio,executed_action=executed_action,timestamp=timestamp)

    def observe_features(self,patches:Tensor,*,grid:tuple[int,int]|None=None,
                         segment:str,proprio:Tensor|None=None,
                         executed_action:Tensor|None=None,
                         timestamp:float|None=None,observed:bool=True):
        """Append ONE verified observation. No predicted features or future actions.

        `executed_action` is the actual previous action reported by the controller.
        It is NOT the current planned action chunk.
        """
        if observed is not True:
            raise ValueError("禁止把模型预测或未确认事件写入历史 (observed=False)")
        if segment not in {"demo","robot"}:
            raise ValueError("segment 必须是 demo/robot")
        if patches.ndim!=3 or patches.shape[-1]!=self.cfg.vision_dim:
            raise ValueError("patches 必须是 [B,P,vision_dim]")
        if grid is None:
            p=patches.shape[1]; side=int(p**0.5)
            if side*side!=p:
                raise ValueError("非正方 patch 数量需指定 grid")
            grid=(side,side)
        if grid[0]*grid[1]!=patches.shape[1]:
            raise ValueError("grid token count mismatch")
        if segment=="demo" and (proprio is not None or executed_action is not None):
            raise ValueError("首版演示为无动作标签视频，不允许传入机器人动作或状态")
        if segment=="robot" and not self.history.segment_closed:
            self.end_demo()
        prev=self.last_observed_timestamp[segment]
        if timestamp is not None and prev is not None and timestamp<=prev:
            raise ValueError("重复或倒退的观测时间戳；禁止重复写入")
        if self.fast_state is None:
            self.fast_state=self.memory.initial_state(patches.shape[0])
        elif self.history.batch_size!=patches.shape[0]:
            raise ValueError("batch size 不得在一个任务中变化")
        self.fast_state=self.memory.write(self.fast_state,patches,segment=segment,
                                     proprio=proprio,executed_action=executed_action)
        self.history.add(patches,segment=segment,grid=grid)
        self.latest_proprio=proprio if segment=="robot" else self.latest_proprio
        self.seen_robot = self.seen_robot or segment=="robot"
        self.last_observed_timestamp[segment]=timestamp
        self.n_observations+=1

    def _predict_with_context(self) -> TwoFrameDeltas:
        if not self.seen_robot:
            raise RuntimeError("必须先读取机器人当前的真实观测才可以预测动作")
        history,segments=self.history.sequence()
        encoded=self.encoder(history,segments)
        current=encoded[:,-1]
        raw_current=history[:,-1]
        # Stable target coordinates are independent of task/temporal resampling.
        current_target=self.fixed_target(raw_current,self.history.grid)
        memory_readout=self.memory.read(self.fast_state,current)
        memory_tokens=self.memory_to_hidden(memory_readout)
        task_queries=self.resampler(current,encoded,memory_tokens)
        return self.predictor(current_target,task_queries)

    def predict_two_deltas(self) -> TwoFrameDeltas:
        # Returned tensors are ephemeral; no internal pending-prediction buffer.
        return self._predict_with_context()

    @staticmethod
    def action_condition(deltas:TwoFrameDeltas) -> Tensor:
        return torch.cat([deltas.current,deltas.short,deltas.goal],dim=1)

    @torch.no_grad()
    def predict_actions(self,*,noise:Tensor|None=None,steps:int|None=None) -> Tensor:
        deltas=self.predict_two_deltas()
        condition=self.action_condition(deltas)
        return self.fm.sample(condition,self.latest_proprio,steps=steps,noise=noise)

    def loss_for_observed_robot_step(self,*,action_target:Tensor,
                                    short_future_features:Tensor,goal_features:Tensor,
                                    historical_target:Tensor|None=None,
                                    teacher=None,teacher_explicit:Tensor|None=None,
                                    teacher_addresses:Tensor|None=None) -> dict[str,Tensor]:
        """Training-only future labels: NEVER fed to model's current state.

        Call after observing real robot frame at t, but before observing future t+h.
        `goal_features` belongs to a successful robot target, NOT human demo.
        """
        deltas=self.predict_two_deltas()
        condition=self.action_condition(deltas)
        short_t,goal_t=self.fixed_target.residual_targets(
            self.history.robot[-1],short_future_features,goal_features,self.history.grid)
        loss_short=F.mse_loss(deltas.short,short_t)
        loss_goal=F.mse_loss(deltas.goal,goal_t)
        loss_fm=self.fm.loss(action_target,condition,self.latest_proprio)
        total=self.cfg.weight_fm*loss_fm + self.cfg.weight_short*loss_short + self.cfg.weight_goal*loss_goal
        zero=total.new_zeros(())
        loss_mem=zero
        if self.cfg.weight_memory!=0:
            if teacher is None or historical_target is None or teacher_explicit is None or teacher_addresses is None:
                raise ValueError("G3 memory loss 需要已训练 teacher、历史 target 与地址")
            if not isinstance(self.memory,ResidualFastMemory):
                raise ValueError("memory_mode='none' 不能开启 TTT-RM loss")
            from .history_teacher import complementary_memory_loss
            loss_mem=complementary_memory_loss(self.memory,self.fast_state,teacher,
                          self.history.robot[-1],teacher_explicit,teacher_addresses,historical_target)
            total=total+self.cfg.weight_memory*loss_mem
        return {"total":total,"fm":loss_fm,"short":loss_short,"goal":loss_goal,"memory":loss_mem}

    def detach_stream_state(self):
        """Truncated BPTT between training windows; keeps numeric history state."""
        if self.fast_state is not None:
            self.fast_state=self.fast_state.detach()
        self.history.demo=[x.detach() for x in self.history.demo]
        self.history.robot=[x.detach() for x in self.history.robot]

    def snapshot_demo(self) -> tuple:
        """Evaluation-only copy for repeat trials. NOT a concurrently updated memory."""
        if self.seen_robot:
            raise RuntimeError("演示快照必须在机器人执行之前创建")
        return (None if self.fast_state is None else self.fast_state.detach().clone(),
                [x.detach().clone() for x in self.history.demo],self.history.grid)

    def restore_demo(self,snapshot):
        self.reset_task()
        w,demo,grid=snapshot
        self.fast_state=None if w is None else w.detach().clone()
        self.history.demo=[x.detach().clone() for x in demo]
        self.history.grid=grid
        self.history.batch_size=None if not demo else demo[0].shape[0]
        self.end_demo()
