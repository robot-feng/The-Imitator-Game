"""Standard conditional Flow Matching; NOT Rolling Flow or video diffusion."""
from __future__ import annotations
import math
import torch
from torch import Tensor,nn
import torch.nn.functional as F


class ActionFlowBlock(nn.Module):
    def __init__(self,dim:int,heads:int):
        super().__init__()
        self.ln1=nn.LayerNorm(dim)
        self.self_attn=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.ln2=nn.LayerNorm(dim)
        self.cross_attn=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.ln3=nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Linear(4*dim,dim))

    def forward(self,actions:Tensor,condition:Tensor) -> Tensor:
        x=self.ln1(actions)
        u,_=self.self_attn(x,x,x,need_weights=False)
        actions=actions+u
        u,_=self.cross_attn(self.ln2(actions),condition,condition,need_weights=False)
        actions=actions+u
        return actions+self.ff(self.ln3(actions))


class StandardFlowMatchingPolicy(nn.Module):
    def __init__(self,dim:int,heads:int,depth:int,action_dim:int,
                 horizon:int,proprio_dim:int,steps:int=8):
        super().__init__()
        self.action_dim,self.horizon,self.steps,self.proprio_dim=(
            action_dim,horizon,steps,proprio_dim)
        self.action_embed=nn.Linear(action_dim,dim)
        self.time_embed=nn.Sequential(nn.Linear(dim,dim),nn.SiLU(),nn.Linear(dim,dim))
        self.proprio_embed=nn.Linear(proprio_dim,dim)
        self.position=nn.Parameter(torch.randn(1,horizon,dim)*0.01)
        self.blocks=nn.ModuleList([ActionFlowBlock(dim,heads) for _ in range(depth)])
        self.out=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,action_dim))
        self.dim=dim

    def _time_embedding(self,time:Tensor) -> Tensor:
        # time [B] -> [B,D]; pure sinusoidal embedding
        half=self.dim//2
        f=torch.exp(torch.arange(half,device=time.device,dtype=time.dtype)*(-math.log(10000)/max(1,half-1)))
        v=time[:,None]*f[None,:]
        emb=torch.cat([v.sin(),v.cos()],dim=-1)
        if emb.shape[-1]<self.dim:
            emb=F.pad(emb,(0,self.dim-emb.shape[-1]))
        return self.time_embed(emb)

    def forward(self,noisy_action:Tensor,t:Tensor,condition:Tensor,
                proprio:Tensor|None=None) -> Tensor:
        b,h,a=noisy_action.shape
        if h!=self.horizon or a!=self.action_dim:
            raise ValueError("Action Chunk shape mismatch")
        if t.ndim==0:
            t=t.expand(b)
        if t.shape!=(b,):
            raise ValueError("t 应为 [B]")
        if proprio is None:
            proprio=noisy_action.new_zeros((b,self.proprio_dim))
        if proprio.shape!=(b,self.proprio_dim):
            raise ValueError("proprio shape mismatch")
        context=torch.cat([condition,self.proprio_embed(proprio)[:,None,:]],dim=1)
        x=self.action_embed(noisy_action)+self.position+self._time_embedding(t)[:,None,:]
        for block in self.blocks:
            x=block(x,context)
        return self.out(x)

    def loss(self,actions:Tensor,condition:Tensor,proprio:Tensor|None=None) -> Tensor:
        eps=torch.randn_like(actions)
        t=torch.rand((actions.shape[0],),device=actions.device,dtype=actions.dtype)
        interpolated=(1-t[:,None,None])*eps + t[:,None,None]*actions
        v_target=actions-eps
        v=self(interpolated,t,condition,proprio)
        return F.mse_loss(v,v_target)

    @torch.no_grad()
    def sample(self,condition:Tensor,proprio:Tensor|None=None,
               *,steps:int|None=None,noise:Tensor|None=None) -> Tensor:
        b=condition.shape[0]
        n_steps=self.steps if steps is None else int(steps)
        if n_steps<=0:
            raise ValueError("steps must > 0")
        x=torch.randn((b,self.horizon,self.action_dim),device=condition.device,dtype=condition.dtype) if noise is None else noise.clone()
        for i in range(n_steps):
            t=torch.full((b,),i/n_steps,device=x.device,dtype=x.dtype)
            x=x+self(x,t,condition,proprio)/n_steps
        return x
