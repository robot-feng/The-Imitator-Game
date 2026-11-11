import torch
import pytest
from torch import nn
from reswam.vision import FrozenDINOv3,FixedSpatialTarget

class FakeBackbone(nn.Module):
    num_features=384
    class PE:
        patch_size=(16,16)
    patch_embed=PE()
    def __init__(self):
        super().__init__()
        self.lin=nn.Linear(3,384)
    def forward_features(self,images):
        p=images.shape[-1]//16
        patch=images.mean((2,3))[:,None,:]
        tokens=self.lin(patch).expand(-1,p*p,-1)
        prefix=tokens.new_full((images.shape[0],5,384),77)
        return torch.cat((prefix,tokens),dim=1)

def test_exact_default_encoder_name_pretrained_and_special_tokens():
    calls=[]
    def factory(name,*,pretrained,num_classes):
        calls.append((name,pretrained,num_classes))
        return FakeBackbone()
    model=FrozenDINOv3(model_factory=factory)
    features,grid=model(torch.zeros(2,3,256,256))
    assert calls==[("vit_small_patch16_dinov3.lvd1689m",True,0)]
    assert features.shape==(2,256,384)
    assert grid==(16,16)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    model.train()
    assert not model.backbone.training

def test_fixed_target_is_time_invariant():
    proj=FixedSpatialTarget(2)
    x=torch.randn(2,16,32)
    a,b=proj.residual_targets(x,x,x,(4,4))
    assert torch.count_nonzero(a)==0
    assert torch.count_nonzero(b)==0
    assert proj(x,(4,4)).shape==(2,4,32)
    with pytest.raises(ValueError):proj(x,(5,4))
