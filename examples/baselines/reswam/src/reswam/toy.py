"""Offline synthetic latent ICL data. For debug only, not a robot benchmark.

Two tasks have exactly the same current robot observation but opposite demo
outcomes, so only historical demonstration can identify the required action.
"""
from __future__ import annotations
import torch
from torch import Tensor


def make_toy_features(position:Tensor,dim:int=32,patch_hw:int=2) -> Tensor:
    """position [B] -> [B,P,D]. Stable artificial coordinate for all frames."""
    b=position.shape[0]
    p=patch_hw*patch_hw
    v=torch.zeros((b,p,dim),device=position.device)
    coords=torch.linspace(-1,1,p,device=position.device)
    for i in range(p):
        v[:,i,0]=(1+0.2*coords[i])*position
        v[:,i,1]=position.square()
        v[:,i,2]=torch.sin(position*1.5)
        v[:,i,3]=coords[i]
        v[:,i,4]=1.0
        v[:,i,5]=float(i%2)
    return v


def toy_batch(batch:int,config,*,generator:torch.Generator|None=None,
              device:str|torch.device="cpu",occluded:bool=False) -> dict[str,Tensor]:
    # Robot starts from zero in every task; demo ends at +/- 1.
    goal_sign=(torch.randint(0,2,(batch,),generator=generator)*2-1).float().to(device)
    initial=torch.zeros_like(goal_sign)
    demo_start=make_toy_features(initial,config.vision_dim,config.target_hw)
    demo_end=make_toy_features(initial if occluded else goal_sign,config.vision_dim,config.target_hw)
    demo_middle=make_toy_features(goal_sign,config.vision_dim,config.target_hw)
    robot=make_toy_features(initial,config.vision_dim,config.target_hw)
    short=make_toy_features(goal_sign*0.45,config.vision_dim,config.target_hw)
    goal=make_toy_features(goal_sign,config.vision_dim,config.target_hw)
    action=torch.zeros((batch,config.action_horizon,config.action_dim),device=device)
    action[...,0]=goal_sign[:,None]*0.8
    action[...,1]=goal_sign[:,None]*0.35
    proprio=torch.zeros((batch,config.proprio_dim),device=device)
    return {"demo_start":demo_start,"demo_middle":demo_middle if occluded else None,
            "demo_end":demo_end,"robot":robot,
            "short":short,"goal":goal,"action":action,"proprio":proprio,"goal_sign":goal_sign}


def fill_toy_history(model,batch):
    p=model.cfg.target_hw
    model.reset_task()
    model.observe_features(batch["demo_start"],grid=(p,p),segment="demo",timestamp=0.)
    if batch.get("demo_middle") is not None:
        model.observe_features(batch["demo_middle"],grid=(p,p),segment="demo",timestamp=1.)
    model.observe_features(batch["demo_end"],grid=(p,p),segment="demo",timestamp=2.)
    model.end_demo()
    model.observe_features(batch["robot"],grid=(p,p),segment="robot",
                           proprio=batch["proprio"],timestamp=0.)
