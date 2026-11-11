"""Produce a tiny synthetic encoded-episode fixture for train_encoded.py testing."""
import argparse
import torch
from pathlib import Path
from reswam.config import ResWAMConfig
from reswam.toy import make_toy_features


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",default="runs/tmp/toy_encoded.pt")
    args=p.parse_args()
    cfg=ResWAMConfig.tiny()
    episodes=[]
    for i in range(6):
        g=1.0 if i%2==0 else -1.0
        def feat(p):
            return make_toy_features(torch.tensor([p]),cfg.vision_dim,cfg.target_hw)[0]
        demo=torch.stack([feat(0.),feat(g)])
        positions=torch.linspace(0,g,12)
        robot=torch.stack([feat(float(pos)) for pos in positions])
        a=torch.zeros((12,cfg.action_dim));a[:,0]=g*0.8;a[:,1]=g*0.35
        proprio=torch.zeros((12,cfg.proprio_dim))
        episodes.append({"id":f"toy{i}","demo":demo,"robot":robot,
                         "actions":a,"proprio":proprio,"grid":(cfg.target_hw,cfg.target_hw)})
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"encoder":"synthetic_feature_fixture","episodes":episodes},target)
    print(target)

if __name__=="__main__":main()
