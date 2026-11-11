"""Exactly TWO visual latent frames: short/goal, both expressed as deltas to now."""
from __future__ import annotations
from dataclasses import dataclass
from torch import Tensor,nn
import torch


@dataclass
class TwoFrameDeltas:
    short: Tensor
    goal: Tensor
    current: Tensor


class TwoFrameResidualPredictor(nn.Module):
    def __init__(self,dim:int,heads:int,depth:int,n_targets:int,ff_mult:int=4,dropout:float=0.):
        super().__init__()
        self.n_targets=n_targets
        self.current_type=nn.Parameter(torch.randn(1,1,dim)*0.02)
        self.short_type=nn.Parameter(torch.randn(1,1,dim)*0.02)
        self.goal_type=nn.Parameter(torch.randn(1,1,dim)*0.02)
        layer=nn.TransformerEncoderLayer(dim,heads,dim*ff_mult,dropout,
                                         batch_first=True,activation="gelu",norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,depth,enable_nested_tensor=False)
        self.final=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim))

    def forward(self,current:Tensor,queries:Tensor) -> TwoFrameDeltas:
        b,k,d=current.shape
        if k!=self.n_targets or queries.shape!=(b,2,k,d):
            raise ValueError("期望 current[B,K,D] 与 queries[B,2,K,D]")
        short=queries[:,0]+self.short_type
        goal=queries[:,1]+self.goal_type
        seq=torch.cat([current+self.current_type,short,goal],dim=1)
        out=self.transformer(seq)[:,k:]
        short_delta=self.final(out[:,:k])
        goal_delta=self.final(out[:,k:])
        return TwoFrameDeltas(short=short_delta,goal=goal_delta,current=current)
