from pathlib import Path
import torch
import pytest
from reswam.data import validate_episode,load_feature_cache,evenly_spaced_indices


def episode():
    return {"id":"a","demo":torch.zeros(2,4,32),
            "robot":torch.zeros(12,4,32),"actions":torch.zeros(12,2),
            "proprio":torch.zeros(12,2),"grid":(2,2)}

def test_feature_cache_contract(tmp_path):
    ep=episode()
    assert validate_episode(ep,action_dim=2,proprio_dim=2,vision_dim=32)
    path=tmp_path/"episodes.pt"
    torch.save({"episodes":[ep]},path)
    data=load_feature_cache(path,action_dim=2,proprio_dim=2,vision_dim=32)
    assert len(data["episodes"])==1
    bad=dict(ep);bad["actions"]=torch.zeros(8,2)
    with pytest.raises(ValueError):validate_episode(bad,action_dim=2,proprio_dim=2,vision_dim=32)


def test_demo_sampling_preserves_start_and_goal():
    assert evenly_spaced_indices(20,3)==[0,10,19]
    assert evenly_spaced_indices(2,3)==[0,1]
