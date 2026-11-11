"""TTT-RM-inspired residual Fast Weights. Not an official TTT-RM reproduction.

V1 writes only actually observed visual features and confirmed robot state/actions.
No predicted short/goal values or imagined actions ever enter the state.
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def patch_xy(n:int,device, dtype) -> Tensor:
    side=int(n**0.5)
    if side*side!=n:
        raise ValueError("原型仅支持正方形 patch grid")
    axes=torch.linspace(-1.,1.,side,device=device,dtype=dtype)
    yy,xx=torch.meshgrid(axes,axes,indexing="ij")
    return torch.stack([xx.flatten(),yy.flatten()],dim=-1)


class ResidualFastMemory(nn.Module):
    """Single differentiable matrix state, normalized key-value delta update.

    Inner write target is fixed visual value minus key, resembling KVB residual
    self-supervision. The history-reconstruction residual is for OFFLINE outer
    teacher supervision, not an online label. All operations are differentiable
    through a short prefill+robot unroll; never update slow nn.Parameters online.
    """

    def __init__(self, visual_dim:int, memory_dim:int, proprio_dim:int,
                 action_dim:int, n_reads:int, lr:float=0.08):
        super().__init__()
        self.memory_dim,self.proprio_dim,self.action_dim,self.lr=(
            memory_dim,proprio_dim,action_dim,float(lr))
        meta_dim=2+1+proprio_dim+action_dim+2
        self.key=nn.Sequential(nn.Linear(visual_dim+meta_dim,memory_dim),nn.LayerNorm(memory_dim),nn.Tanh())
        self.query=nn.Sequential(nn.Linear(visual_dim+3,memory_dim),nn.LayerNorm(memory_dim),nn.Tanh())
        self.w0=nn.Parameter(torch.zeros(memory_dim,memory_dim))
        # Fixed value target: same shared frozen visual coordinates on every frame.
        rng=torch.Generator(device="cpu").manual_seed(1701)
        mat=torch.randn(memory_dim,visual_dim,generator=rng)
        q,_=torch.linalg.qr(mat.T,mode="reduced") if memory_dim<=visual_dim else (None,None)
        if q is None:
            raise ValueError("memory_dim 必须 <= visual_dim")
        self.register_buffer("value_projection",q.T.contiguous())
        addresses=[]
        for i in range(n_reads):
            # [relative past interval, x spatial address, y spatial address]
            addresses.append([min(1.0, (2**(i//4))/16.0), (-1.0+2*(i%2)), (-1.0+2*((i//2)%2))])
        self.register_buffer("default_addresses",torch.tensor(addresses,dtype=torch.float32))

    def initial_state(self,batch:int) -> Tensor:
        return self.w0.unsqueeze(0).expand(batch,-1,-1).clone()

    def project_value(self,features:Tensor) -> Tensor:
        return F.linear(features.detach().float(),self.value_projection)

    def write(self,w:Tensor,patches:Tensor,*,segment:str,
              proprio:Tensor|None=None,executed_action:Tensor|None=None) -> Tensor:
        if segment not in {"demo","robot"}:
            raise ValueError("segment 必须是 demo 或 robot")
        b,n,d=patches.shape
        if w.shape!=(b,self.memory_dim,self.memory_dim):
            raise ValueError("Fast Weight batch/dimension mismatch")
        visual=patches.detach().float()
        coords=patch_xy(n,visual.device,visual.dtype).unsqueeze(0).expand(b,-1,-1)
        def optional(value,dim):
            if value is None:
                return torch.zeros((b,dim),device=visual.device,dtype=visual.dtype),torch.zeros((b,1),device=visual.device,dtype=visual.dtype)
            if value.shape!=(b,dim):
                raise ValueError(f"metadata 应是 {(b,dim)}，实际 {tuple(value.shape)}")
            return value.detach().to(dtype=visual.dtype),torch.ones((b,1),device=visual.device,dtype=visual.dtype)
        prop,prop_present=optional(proprio,self.proprio_dim)
        act,act_present=optional(executed_action,self.action_dim)
        seg=torch.full((b,1),float(segment=="robot"),device=visual.device,dtype=visual.dtype)
        meta=torch.cat((seg,prop,act,prop_present,act_present),dim=-1)
        meta=meta.unsqueeze(1).expand(-1,n,-1)
        key=self.key(torch.cat((visual,coords,meta),dim=-1))
        value=self.project_value(visual)
        target=value-key
        error=target-torch.bmm(key,w.transpose(1,2))
        norm=key.square().sum(-1,keepdim=True).clamp_min(1e-4)
        update=torch.bmm((error/norm).transpose(1,2),key)/n
        return w + self.lr*update

    def read(self,w:Tensor,current:Tensor,addresses:Tensor|None=None) -> Tensor:
        b,_,d=current.shape
        if addresses is None:
            addresses=self.default_addresses.unsqueeze(0).expand(b,-1,-1)
        if addresses.shape[:1]!=(b,) or addresses.shape[-1]!=3:
            raise ValueError("memory read addresses 应为 [B,Q,3]")
        cur=current.detach().float().mean(dim=1,keepdim=True).expand(-1,addresses.shape[1],-1)
        q=self.query(torch.cat((cur,addresses.to(cur.dtype)),dim=-1))
        return torch.bmm(q,(w-self.w0.unsqueeze(0)).transpose(1,2))


class NullMemory(nn.Module):
    """Equivalent fixed zero memory tokens for controlled ablation."""
    def __init__(self,mem_dim:int,n_reads:int):
        super().__init__()
        self.memory_dim,self.n_reads=mem_dim,n_reads

    def initial_state(self,batch:int):
        return None

    def write(self,w,patches,**kwargs):
        return None

    def read(self,w,current,addresses=None):
        q=self.n_reads if addresses is None else addresses.shape[1]
        return current.new_zeros((current.shape[0],q,self.memory_dim))
