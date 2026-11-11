"""Factorized spatial-temporal causal encoder with safe segment boundaries.

Temporal patch-index attention is limited to the SAME physical trajectory.
Across demo/robot segments, only causal global frame summaries communicate.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SelfAttentionFFN(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio=4, dropout=0.):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3*dim)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio*dim), nn.GELU(), nn.Linear(mlp_ratio*dim, dim))
        self.dropout = float(dropout)

    def forward(self, x: Tensor, allowed: Tensor | None = None) -> Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        def split(t):
            return t.reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = map(split, (q, k, v))
        if allowed is not None:
            if allowed.shape != (b, n, n):
                raise ValueError(f"允许矩阵应为 {(b,n,n)}，实际 {tuple(allowed.shape)}")
            mask = allowed[:, None, :, :]
        else:
            mask = None
        out = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,
                                             dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1,2).reshape(b,n,d)
        x = x + self.proj(out)
        return x + self.mlp(self.ln2(x))


class FactorizedCausalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio=4, dropout=0.):
        super().__init__()
        self.spatial = SelfAttentionFFN(dim,heads,mlp_ratio,dropout)
        self.temporal = SelfAttentionFFN(dim,heads,mlp_ratio,dropout)
        self.global_temporal = SelfAttentionFFN(dim,heads,mlp_ratio,dropout)
        self.global_gate = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def temporal_allow_mask(segment_ids: Tensor) -> Tensor:
        """Only causal SAME-segment local patch mixing across time."""
        b,t = segment_ids.shape
        causal = torch.ones(t,t,device=segment_ids.device,dtype=torch.bool).tril()
        same = segment_ids[:, :, None] == segment_ids[:, None, :]
        return same & causal[None]

    def forward(self, x: Tensor, segments: Tensor) -> Tensor:
        b,t,p,d=x.shape
        x = self.spatial(x.reshape(b*t,p,d)).reshape(b,t,p,d)
        patch_streams = x.transpose(1,2).reshape(b*p,t,d)
        mask = self.temporal_allow_mask(segments).repeat_interleave(p, dim=0)
        patch_streams=self.temporal(patch_streams,mask)
        x=patch_streams.reshape(b,p,t,d).transpose(1,2)
        # Only frame-level summary exchanges information across segment boundaries.
        global_allow=torch.ones((b,t,t),dtype=torch.bool,device=x.device).tril()
        summaries = self.global_temporal(x.mean(dim=2),global_allow)
        return x + torch.tanh(self.global_gate) * summaries[:, :, None, :]


class CausalSpatiotemporalEncoder(nn.Module):
    def __init__(self, dim: int, heads: int, depth=2, mlp_ratio=4, dropout=0.):
        super().__init__()
        self.layers=nn.ModuleList([
            FactorizedCausalBlock(dim,heads,mlp_ratio,dropout)
            for _ in range(depth)
        ])
        self.final_ln=nn.LayerNorm(dim)
        self.segment_embedding=nn.Embedding(2,dim)
        nn.init.normal_(self.segment_embedding.weight,std=0.02)
        self.temporal_scale=nn.Parameter(torch.tensor(0.1))

    def forward(self, frames: Tensor, segments: Tensor) -> Tensor:
        if frames.ndim != 4:
            raise ValueError("frames 必须 [B,T,P,D]")
        b,t,p,d=frames.shape
        if segments.shape!=(b,t):
            raise ValueError("segment ids 必须 [B,T]")
        # Ordinal time encoding on the bounded causal prefix, same for demo/robot.
        # Real sensor timestamps are validated in the event API but not yet used
        # to rescale this sinusoid; irregular FPS needs a later data-alignment gate.
        half=d//2
        positions=torch.arange(t,device=frames.device,dtype=frames.dtype)
        freq=torch.exp(torch.arange(half,device=frames.device,dtype=frames.dtype)*(-math.log(10000)/max(1,half-1)))
        angle=positions[:,None]*freq[None,:]
        temporal=torch.cat((angle.sin(),angle.cos()),dim=-1)
        if temporal.shape[-1]<d:
            temporal=F.pad(temporal,(0,d-temporal.shape[-1]))
        x=(frames + self.temporal_scale*temporal[None,:,None,:]
                   + self.segment_embedding(segments)[:,:,None,:])
        for blk in self.layers:
            x=blk(x,segments)
        return self.final_ln(x)


class BoundedFrameHistory:
    """Bounded *observed* feature store; never stores predicted delta tensors."""
    def __init__(self, max_demo:int, max_robot:int):
        self.max_demo,self.max_robot=max_demo,max_robot
        self.reset()

    def reset(self):
        self.demo:list[Tensor]=[]
        self.robot:list[Tensor]=[]
        self.segment_closed=False
        self.grid=None
        self.batch_size=None

    def add(self, patches: Tensor, *, segment:str, grid:tuple[int,int]):
        if segment not in {"demo","robot"}:
            raise ValueError("segment 必须是 demo 或 robot")
        if segment == "demo" and self.segment_closed:
            raise RuntimeError("机器人段开始后，不允许继续向演示段追加")
        if self.batch_size is None:
            self.batch_size=patches.shape[0]
            self.grid=grid
        if patches.shape[0]!=self.batch_size or patches.shape[1]!=self.grid[0]*self.grid[1] or grid!=self.grid:
            raise ValueError("原型要求同 batch/camera 的 patch grid 固定；跨视角需外部适配")
        if segment=="robot":
            self.segment_closed=True
            self.robot.append(patches)
            self.robot=self.robot[-self.max_robot:]
        else:
            self.demo.append(patches)
            if len(self.demo)>self.max_demo:
                # 保留早期演示和最新演示，不要求保留全长 patch 缓存。
                inds=torch.linspace(0,len(self.demo)-1,steps=self.max_demo).round().long().tolist()
                self.demo=[self.demo[i] for i in inds]

    def sequence(self) -> tuple[Tensor,Tensor]:
        if not self.demo and not self.robot:
            raise RuntimeError("当前没有观测历史")
        frames=self.demo+self.robot
        seg=[0]*len(self.demo)+[1]*len(self.robot)
        return torch.stack(frames,dim=1),torch.tensor(seg,device=frames[0].device).unsqueeze(0).expand(self.batch_size,-1)
