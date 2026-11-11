"""Late query-conditioned resampling; never used as the target coordinate system."""
from __future__ import annotations
from torch import Tensor,nn
import torch


class TaskConditionedResampler(nn.Module):
    def __init__(self,dim:int,heads:int,n_targets:int):
        super().__init__()
        self.n_targets=n_targets
        self.queries=nn.Parameter(torch.randn(2,n_targets,dim)*0.02)
        self.current_query=nn.Linear(dim,dim)
        self.memory_query=nn.Linear(dim,dim)
        self.cross=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.ln=nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim*2),nn.GELU(),nn.Linear(dim*2,dim))

    def forward(self,current:Tensor,history:Tensor,memory:Tensor) -> Tensor:
        """current [B,P,D], history [B,T,P,D], memory [B,Q,D].

        Output [B,2,K,D] corresponds to short/goal query groups.
        """
        b,t,p,d=history.shape
        q=self.queries.reshape(1,2*self.n_targets,d).expand(b,-1,-1)
        q=q+self.current_query(current.mean(1))[:,None,:]
        q=q+self.memory_query(memory.mean(1))[:,None,:]
        kv=torch.cat([history.reshape(b,t*p,d),memory],dim=1)
        retrieved,_=self.cross(q,kv,kv,need_weights=False)
        output=self.ln(q+retrieved)
        output=output+self.ff(output)
        return output.reshape(b,2,self.n_targets,d)
