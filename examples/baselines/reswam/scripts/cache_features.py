"""Cache frozen DINOv3 patch features from a JSON episode manifest.

Example manifest in docs/DATA_PROTOCOL.md. Creates no learned action labels.
No training or evaluation statistics are inferred by this preprocessing step.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from reswam.vision import FrozenDINOv3


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--no-pretrained",action="store_true")
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch",type=int,default=8)
    args=ap.parse_args()
    root=Path(args.manifest).resolve().parent
    manifest=json.loads(Path(args.manifest).read_text(encoding="utf8"))
    vis=FrozenDINOv3(pretrained=not args.no_pretrained).to(args.device)
    transform=vis.make_transform()
    def encode_images(paths):
        pieces=[]
        grid=None
        for i in range(0,len(paths),args.batch):
            batch=[]
            for rel in paths[i:i+args.batch]:
                image=Image.open(root / rel).convert("RGB")
                batch.append(transform(image))
            z,cur_grid=vis(torch.stack(batch).to(args.device))
            if grid is not None and cur_grid!=grid:
                raise ValueError("Mixed image grid sizes")
            grid=cur_grid
            pieces.append(z.cpu())
        return torch.cat(pieces,dim=0),grid
    episodes=[]
    for entry in manifest["episodes"]:
        demo,grid1=encode_images(entry["demo_frames"])
        robot,grid2=encode_images(entry["robot_frames"])
        if grid1!=grid2:
            raise ValueError("Demo and robot patch grids must agree in the starter")
        actions=torch.from_numpy(np.load(root / entry["actions_path"])).float()
        proprio=torch.from_numpy(np.load(root / entry["proprio_path"])).float()
        episodes.append({"id":entry["id"],"demo":demo,"robot":robot,
                         "actions":actions,"proprio":proprio,"grid":grid1})
        print("Cached",entry["id"],tuple(demo.shape),tuple(robot.shape))
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"episodes":episodes,"encoder":vis.model_name,
                "pretrained":not args.no_pretrained},path)
    print("Saved feature cache:",path)

if __name__=="__main__":main()
