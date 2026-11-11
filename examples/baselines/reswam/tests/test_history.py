import torch
import pytest
from reswam.toy import toy_batch,fill_toy_history
from reswam.causal_context import BoundedFrameHistory


def test_reset_memory_and_history(tiny_model):
    cfg=tiny_model.cfg
    batch=toy_batch(2,cfg)
    fill_toy_history(tiny_model,batch)
    assert tiny_model.n_observations==3
    assert len(tiny_model.history.demo)==2 and len(tiny_model.history.robot)==1
    assert tiny_model.fast_state is not None
    tiny_model.reset_task()
    assert tiny_model.fast_state is None and tiny_model.n_observations==0
    assert not tiny_model.history.demo and not tiny_model.history.robot


def test_demo_snapshot_restoration(tiny_model):
    cfg=tiny_model.cfg
    batch=toy_batch(2,cfg)
    tiny_model.reset_task()
    tiny_model.observe_features(batch["demo_start"],grid=(2,2),segment="demo",timestamp=0.)
    tiny_model.observe_features(batch["demo_end"],grid=(2,2),segment="demo",timestamp=1.)
    snap=tiny_model.snapshot_demo()
    tiny_model.end_demo()
    tiny_model.observe_features(batch["robot"],grid=(2,2),segment="robot",proprio=batch["proprio"],timestamp=0.)
    result1=tiny_model.predict_two_deltas().goal.detach().clone()
    tiny_model.restore_demo(snap)
    tiny_model.observe_features(batch["robot"],grid=(2,2),segment="robot",proprio=batch["proprio"],timestamp=0.)
    result2=tiny_model.predict_two_deltas().goal.detach().clone()
    assert torch.allclose(result1,result2,atol=1e-6)


def test_only_verified_events_and_timestamps(tiny_model):
    cfg=tiny_model.cfg
    batch=toy_batch(2,cfg)
    tiny_model.reset_task()
    tiny_model.observe_features(batch["demo_start"],segment="demo",grid=(2,2),timestamp=1.)
    with pytest.raises(ValueError):
        tiny_model.observe_features(batch["demo_start"],segment="demo",grid=(2,2),timestamp=1.)
    with pytest.raises(ValueError):
        tiny_model.observe_features(batch["demo_start"],segment="demo",grid=(2,2),timestamp=2.,observed=False)
    with pytest.raises(ValueError):
        tiny_model.observe_features(batch["demo_start"],segment="demo",grid=(2,2),executed_action=batch["action"][:,0])
    tiny_model.end_demo()
    with pytest.raises(RuntimeError):
        tiny_model.observe_features(batch["demo_end"],segment="demo",grid=(2,2),timestamp=2.)


def test_bounded_explicit_history():
    hist=BoundedFrameHistory(3,2)
    for i in range(8):
        hist.add(torch.full((1,4,32),float(i)),segment="demo",grid=(2,2))
    assert len(hist.demo)==3
    assert float(hist.demo[-1][0,0,0])==7
    for i in range(4):
        hist.add(torch.full((1,4,32),float(i)),segment="robot",grid=(2,2))
    assert len(hist.robot)==2
    stacked,segments=hist.sequence()
    assert stacked.shape==(1,5,4,32)
    assert segments.tolist()==[[0,0,0,1,1]]
