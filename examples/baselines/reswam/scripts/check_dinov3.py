"""Real default encoder check: requires timm and (first run) HF weight access."""
import argparse
import torch
from PIL import Image
from reswam.vision import FrozenDINOv3,FixedSpatialTarget

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--image",help="本地图像文件；未提供时仅测试随机 256x256 输入")
    ap.add_argument("--no-pretrained",action="store_true",help="仅检查结构，不下载权重")
    args=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    vis=FrozenDINOv3(pretrained=not args.no_pretrained).to(dev)
    if args.image:
        img=Image.open(args.image).convert("RGB")
        x=vis.make_transform()(img).unsqueeze(0).to(dev)
    else:
        x=torch.randn(1,3,256,256,device=dev)
    patches,grid=vis(x)
    latent=FixedSpatialTarget(4)(patches,grid)
    print({"model":vis.model_name,"device":dev,"grid":grid,
           "patch_tokens":tuple(patches.shape),"target_tokens":tuple(latent.shape),
           "pretrained":not args.no_pretrained})

if __name__=="__main__":main()
