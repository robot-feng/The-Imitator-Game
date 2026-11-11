"""Offline-only TTT-RM-style historical reconstruction teacher.

This teacher does NOT run in online control and NEVER supplies online writes.
The explicit-context baseline must not contain the selected hidden target.
"""
from __future__ import annotations
import torch
from torch import Tensor,nn
import torch.nn.functional as F


class HistoryDecoder(nn.Module):
    def __init__(self,visual_dim:int,mem_dim:int,heads:int=4):
        super().__init__()
        if visual_dim%heads:
            raise ValueError("visual_dim % heads must be zero")
        self.to_query=nn.Linear(visual_dim+3,visual_dim)
        self.cross=nn.MultiheadAttention(visual_dim,heads,batch_first=True)
        self.pred=nn.Sequential(nn.LayerNorm(visual_dim),nn.Linear(visual_dim,mem_dim))

    def forward(self,current:Tensor,explicit:Tensor,addresses:Tensor) -> Tensor:
        if current.ndim!=3 or explicit.ndim!=3 or addresses.ndim!=3:
            raise ValueError("teacher shapes: current[B,P,D], explicit[B,Q,D], addr[B,R,3]")
        q=current.mean(dim=1,keepdim=True).expand(-1,addresses.shape[1],-1)
        q=self.to_query(torch.cat([q,addresses],dim=-1))
        cat=torch.cat([current,explicit],dim=1)
        ret,_=self.cross(q,cat,cat,need_weights=False)
        return self.pred(q+ret)

    def reconstruction_loss(self,current:Tensor,explicit:Tensor,
                            addresses:Tensor,target_projected:Tensor) -> Tensor:
        return F.mse_loss(self(current,explicit,addresses),target_projected.detach())


def complementary_memory_loss(memory,fast_state:Tensor,teacher:HistoryDecoder,
                              current:Tensor,explicit:Tensor,addresses:Tensor,
                              history_target:Tensor) -> Tensor:
    """When teacher is pretrained/frozen, train online memory readout on residual.

    Inputs: historical target [B,Q,visual_dim]; addresses [B,Q,3].
    """
    with torch.no_grad():
        recovered=teacher(current.detach(),explicit.detach(),addresses)
        residual=memory.project_value(history_target)-recovered
    prediction=memory.read(fast_state,current,addresses)
    return F.mse_loss(prediction,residual)
