"""First real-data interface: train V1 from precomputed *actual* DINO patches.

One episode/task paired demo+robot, per-step expert actions, robot goal frame.
Does not yet claim reproduction of SynthICL or TTT-RM Stage-G3 training.
"""
from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import torch
from reswam import ResWAM,ResWAMConfig
from reswam.data import load_feature_cache,evenly_spaced_indices


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--steps",type=int,default=100)
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--grad-accum",type=int,default=1)
    p.add_argument("--output",default="runs/official/encoded_train_v1.pt")
    p.add_argument("--memory",choices=("none","delta"),default="delta")
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tiny",action="store_true",help="Only for synthetic cache debug")
    args=p.parse_args()
    if args.steps<1 or args.grad_accum<1: raise ValueError("steps and grad-accum must >0")
    random.seed(args.seed);torch.manual_seed(args.seed)
    if args.device=="cpu":torch.set_num_threads(1)
    cfg=ResWAMConfig.tiny() if args.tiny else ResWAMConfig()
    cfg.memory_mode=args.memory
    data=load_feature_cache(args.data,action_dim=cfg.action_dim,
                   proprio_dim=cfg.proprio_dim,vision_dim=cfg.vision_dim)
    episodes=[ep for ep in data["episodes"] if ep["robot"].shape[0]>max(cfg.short_horizon,cfg.action_horizon)]
    if not episodes:
        raise ValueError("No episodes long enough for configured short_horizon and action_horizon")
    model=ResWAM(cfg,load_vision=False).to(args.device).train()
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    step_log=[]
    start=time.perf_counter()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        metrics=[]
        for micro in range(args.grad_accum):
            ep=random.choice(episodes)
            t_max=ep["robot"].shape[0]-max(cfg.short_horizon+1,cfg.action_horizon)
            t=random.randrange(t_max+1)
            grid=tuple(ep["grid"])
            model.reset_task()
            for j in evenly_spaced_indices(ep["demo"].shape[0],cfg.demo_context_limit):
                model.observe_features(ep["demo"][j:j+1].to(args.device),grid=grid,
                                       segment="demo",timestamp=float(j))
            model.end_demo()
            # Initial G1/G2 protocol: short-window robot context. G3 must add full
            # trajectories and TBPTT to test genuine long-episode memory.
            for j in range(max(0,t-cfg.robot_context_limit+1),t+1):
                prev=ep["actions"][j-1:j].to(args.device) if j>0 else None
                model.observe_features(ep["robot"][j:j+1].to(args.device),
                      grid=grid,segment="robot",proprio=ep["proprio"][j:j+1].to(args.device),
                      executed_action=prev,timestamp=float(j))
            action_chunk=ep["actions"][t:t+cfg.action_horizon].unsqueeze(0).to(args.device)
            short=ep["robot"][t+cfg.short_horizon:t+cfg.short_horizon+1].to(args.device)
            goal=ep["robot"][-1:].to(args.device)
            loss=model.loss_for_observed_robot_step(action_target=action_chunk,
                       short_future_features=short,goal_features=goal)
            (loss["total"]/args.grad_accum).backward()
            metrics.append({k:float(v.detach().cpu()) for k,v in loss.items()})
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        vals={k:round(sum(m[k] for m in metrics)/len(metrics),6) for k in metrics[0]}
        step_log.append(vals)
        if (step+1)%max(1,args.steps//5)==0:
            print(f"{step+1}/{args.steps} total={vals['total']:.4f} fm={vals['fm']:.4f} short={vals['short']:.4f} goal={vals['goal']:.4f}")
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":model.state_dict(),"config":cfg.to_dict(),
                "feature_cache":str(args.data),"steps":args.steps,
                "seed":args.seed},out)
    out.with_suffix(".metrics.json").write_text(json.dumps({
        "status":"training_loss_only_not_closed_loop_evaluation",
        "steps":args.steps,"seed":args.seed,"memory":args.memory,
        "elapsed_s":round(time.perf_counter()-start,3),"first":step_log[0],
        "last":step_log[-1],"history":step_log},indent=2),encoding="utf8")
    print("Checkpoint:",out)

if __name__=="__main__":main()
