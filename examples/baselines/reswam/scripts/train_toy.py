"""Tiny repeatable video-ICL *latent-feature toy*; NOT DINO / robot benchmark.

The toy tests whether demo prefixes influence the two predicted delta frames
and the standard flow policy; true robotic success requires a separate dataset.
"""
from __future__ import annotations
import argparse,json,os,random,sys,time
from datetime import datetime,timezone
from pathlib import Path
import torch
from reswam import ResWAM,ResWAMConfig
from reswam.toy import toy_batch,fill_toy_history,make_toy_features


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--steps",type=int,default=60)
    ap.add_argument("--batch",type=int,default=12)
    ap.add_argument("--lr",type=float,default=3e-3)
    ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--output",default=None)
    ap.add_argument("--memory",choices=["none","delta"],default="delta")
    ap.add_argument("--task",choices=["standard","occluded"],default="standard")
    args=ap.parse_args()
    if args.steps<1: raise ValueError("steps must > 0")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed); random.seed(args.seed)
    cfg=ResWAMConfig.tiny(); cfg.memory_mode=args.memory
    if args.task=="occluded":
        cfg.demo_context_limit=2  # 仅保留首/末中性帧；中间目标事件被显式缓存移除
    model=ResWAM(cfg,load_vision=False).train()
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    gen=torch.Generator().manual_seed(args.seed+100)
    history=[]
    start=time.perf_counter()
    for step in range(args.steps):
        batch=toy_batch(args.batch,cfg,generator=gen,occluded=args.task=="occluded")
        fill_toy_history(model,batch)
        loss=model.loss_for_observed_robot_step(action_target=batch["action"],
              short_future_features=batch["short"],goal_features=batch["goal"])
        opt.zero_grad(set_to_none=True);loss["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        history.append({k:round(float(v.detach()),6) for k,v in loss.items()})
        if (step+1)%max(1,args.steps//5)==0:
            print(f"step={step+1:4d}/{args.steps} fm={history[-1]['fm']:.4f} short={history[-1]['short']:.4f} goal={history[-1]['goal']:.4f}")
    # Held-out synthetic combinations; no real visual encoder loaded.
    torch.manual_seed(args.seed+999)
    model.eval()
    val=toy_batch(64,cfg,generator=torch.Generator().manual_seed(4700),occluded=args.task=="occluded")
    fill_toy_history(model,val)
    with torch.no_grad():
        out=model.predict_two_deltas()
        s,g=model.fixed_target.residual_targets(val["robot"],val["short"],val["goal"],(cfg.target_hw,cfg.target_hw))
        metric={"short_mse":round(float((out.short-s).square().mean()),6),
                "goal_mse":round(float((out.goal-g).square().mean()),6)}
        # Counterfactual demo swap: a current scene identical across examples
        # should produce a different predicted goal when only demo changes.
        prev=out.goal.detach().clone()
        # 对立演示：只改变先前的目标事件，机器人当前观测完全相同。
        rev=dict(val)
        changed=make_toy_features(-val["goal_sign"],cfg.vision_dim,cfg.target_hw)
        if args.task=="occluded":
            rev["demo_middle"]=changed
        else:
            rev["demo_end"]=changed
        fill_toy_history(model,rev)
        altered=model.predict_two_deltas().goal
        metric["demo_swap_delta_l2"]=round(float((prev-altered).square().mean().sqrt()),6)
        # 预测的 Goal Delta 第 0 维对应本玩具的有符号位置变化。
        metric["goal_direction_accuracy"]=round(float((out.goal[...,0].mean(1).sign()==val["goal_sign"]).float().mean()),6)
        metric["swapped_goal_direction_accuracy"]=round(float((altered[...,0].mean(1).sign()==-val["goal_sign"]).float().mean()),6)
        # 标准 FM 每次生成使用不同噪声时，MSE 受到采样方差影响；此处用固定零噪声作配对测试。
        fill_toy_history(model,val)
        fixed_noise=torch.zeros((val["goal_sign"].shape[0],cfg.action_horizon,cfg.action_dim))
        sampled=model.predict_actions(noise=fixed_noise,steps=cfg.fm_steps)
        metric["fm_action_direction_accuracy"]=round(float((sampled[...,0].mean(1).sign()==val["goal_sign"]).float().mean()),6)
    secs=round(time.perf_counter()-start,3)
    data={"status":"toy_feature_only_not_robot_evaluation","seed":args.seed,"steps":args.steps,
          "batch":args.batch,"lr":args.lr,"memory":args.memory,"task":args.task,"elapsed_s":secs,
          "first_train_loss":history[0],"last_train_loss":history[-1],"heldout":metric,
          "history":history,"config":cfg.to_dict()}
    if args.output:
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
        print("Saved:",path)
    print("Held-out:",metric, "elapsed_s:",secs)

if __name__=="__main__":main()
