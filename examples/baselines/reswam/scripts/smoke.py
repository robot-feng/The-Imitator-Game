"""Quick import/shape/gradient smoke with off-line synthetic patch features."""
from reswam import ResWAM,ResWAMConfig
from reswam.toy import toy_batch,fill_toy_history
import torch


def main():
    torch.manual_seed(11)
    torch.set_num_threads(1)
    cfg=ResWAMConfig.tiny()
    model=ResWAM(cfg,load_vision=False)
    batch=toy_batch(3,cfg)
    fill_toy_history(model,batch)
    output=model.predict_two_deltas()
    losses=model.loss_for_observed_robot_step(
        action_target=batch["action"],short_future_features=batch["short"],goal_features=batch["goal"])
    losses["total"].backward()
    ngrad=sum(p.grad is not None and p.grad.abs().sum().item()>0 for p in model.parameters())
    actions=model.predict_actions(steps=2)
    print({"short":tuple(output.short.shape),"goal":tuple(output.goal.shape),
           "action":tuple(actions.shape),"total_loss":round(losses["total"].item(),5),
           "params_with_grad":ngrad,"observed_events":model.n_observations})
    assert ngrad>0 and model.n_observations==3

if __name__=="__main__": main()
