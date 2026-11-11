"""Feature-cache data protocol for adapting real robot datasets to this prototype.

No assumptions about a specific LIBERO/SynthICL episode file format.
Caller provides aligned RGBs, executed actions, and proprioception.
"""
from __future__ import annotations
from pathlib import Path
import torch
from torch import Tensor


REQUIRED=("id","demo","robot","actions","proprio","grid")


def validate_episode(ep:dict,*,action_dim:int,proprio_dim:int,vision_dim:int):
    missing=[k for k in REQUIRED if k not in ep]
    if missing:raise ValueError(f"Episode missing fields: {missing}")
    demo,robot,actions,proprio=(ep[k] for k in ("demo","robot","actions","proprio"))
    if demo.ndim!=3 or robot.ndim!=3 or demo.shape[1:]!=robot.shape[1:]:
        raise ValueError("demo/robot must be [T,P,C] with a shared token shape")
    if demo.shape[2]!=vision_dim:
        raise ValueError("Cached features do not match chosen visual encoder")
    t=robot.shape[0]
    if actions.shape!=(t,action_dim):
        raise ValueError(f"actions must [T,{action_dim}], got {tuple(actions.shape)}")
    if proprio.shape!=(t,proprio_dim):
        raise ValueError(f"proprio must [T,{proprio_dim}], got {tuple(proprio.shape)}")
    h,w=ep["grid"]
    if h*w!=robot.shape[1]:
        raise ValueError("Patch grid does not match feature cache")
    if demo.shape[0]<2 or t<2:
        raise ValueError("At least two demo/robot observations required")
    return True


def load_feature_cache(path:str|Path,*,action_dim:int,proprio_dim:int,vision_dim:int):
    # The cache must originate from a trusted local source. weights_only reduces
    # pickle attack surface but this should not be treated as complete sandboxing.
    data=torch.load(path,map_location="cpu",weights_only=True)
    if not isinstance(data,dict) or "episodes" not in data:
        raise ValueError("Expected {'episodes': [...]}")
    for ep in data["episodes"]:
        validate_episode(ep,action_dim=action_dim,proprio_dim=proprio_dim,vision_dim=vision_dim)
    return data


def evenly_spaced_indices(total:int,limit:int):
    if total<=limit:
        return list(range(total))
    inds=torch.linspace(0,total-1,steps=limit).round().long().tolist()
    return sorted(set(inds))
