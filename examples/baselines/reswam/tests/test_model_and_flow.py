import torch
from reswam.toy import toy_batch,fill_toy_history,make_toy_features
from reswam import ResWAM,ResWAMConfig


def test_exactly_two_predicted_visual_frames_and_standard_fm(tiny_model):
    batch=toy_batch(2,tiny_model.cfg)
    fill_toy_history(tiny_model,batch)
    before=tiny_model.n_observations
    d=tiny_model.predict_two_deltas()
    assert d.short.shape==(2,4,32)
    assert d.goal.shape==(2,4,32)
    x=torch.zeros(2,tiny_model.cfg.action_horizon,tiny_model.cfg.action_dim)
    act=tiny_model.predict_actions(noise=x,steps=2)
    assert act.shape==(2,4,2)
    assert tiny_model.n_observations==before
    assert not hasattr(tiny_model,"pending_predictions")
    assert not hasattr(tiny_model,"prediction_cache")


def test_future_labels_never_enter_context_or_memory(tiny_model):
    batch=toy_batch(2,tiny_model.cfg)
    fill_toy_history(tiny_model,batch)
    n=tiny_model.n_observations
    w=tiny_model.fast_state.detach().clone()
    o1=tiny_model.predict_two_deltas().goal.detach().clone()
    loss1=tiny_model.loss_for_observed_robot_step(action_target=batch["action"],
        short_future_features=batch["short"],goal_features=batch["goal"])
    loss2=tiny_model.loss_for_observed_robot_step(action_target=batch["action"],
        short_future_features=-batch["short"],goal_features=-batch["goal"])
    o2=tiny_model.predict_two_deltas().goal.detach().clone()
    assert torch.allclose(o1,o2)
    assert torch.allclose(w,tiny_model.fast_state)
    assert tiny_model.n_observations==n
    assert loss1["goal"]!=loss2["goal"]


def test_fm_gradient_reaches_visual_residual_predictor(tiny_model):
    tiny_model.train()
    batch=toy_batch(2,tiny_model.cfg)
    fill_toy_history(tiny_model,batch)
    pred=tiny_model.predict_two_deltas()
    only_action_loss=tiny_model.fm.loss(batch["action"],tiny_model.action_condition(pred),batch["proprio"])
    only_action_loss.backward()
    grad=sum(float(p.grad.abs().sum()) for p in tiny_model.predictor.parameters() if p.grad is not None)
    assert grad>0


def test_demo_swap_changes_prediction_without_changing_current_robot_obs(tiny_model):
    cfg=tiny_model.cfg
    batch=toy_batch(2,cfg)
    fill_toy_history(tiny_model,batch)
    a=tiny_model.predict_two_deltas().goal.detach().clone()
    counterfact=dict(batch)
    counterfact["demo_end"]=make_toy_features(-batch["goal_sign"],cfg.vision_dim,cfg.target_hw)
    fill_toy_history(tiny_model,counterfact)
    b=tiny_model.predict_two_deltas().goal.detach().clone()
    assert (a-b).abs().max()>1e-8


def test_memory_disabled_ablation_runs():
    cfg=ResWAMConfig.tiny()
    cfg.memory_mode="none"
    model=ResWAM(cfg,load_vision=False)
    batch=toy_batch(2,cfg)
    fill_toy_history(model,batch)
    loss=model.loss_for_observed_robot_step(action_target=batch["action"],
         short_future_features=batch["short"],goal_features=batch["goal"])
    assert torch.isfinite(loss["total"])
