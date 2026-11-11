import torch
from reswam.causal_context import FactorizedCausalBlock,CausalSpatiotemporalEncoder


def test_no_future_leakage_in_causal_spatiotemporal_encoder():
    torch.manual_seed(3)
    encoder=CausalSpatiotemporalEncoder(32,4,depth=2).eval()
    frames=torch.randn(2,4,4,32)
    # Demo first two frames; robot next two, in both batch elements.
    seg=torch.tensor([[0,0,1,1],[0,0,1,1]])
    before=encoder(frames,seg)
    modified=frames.clone()
    modified[:,3]+=100
    after=encoder(modified,seg)
    assert torch.allclose(before[:,:3],after[:,:3],atol=2e-5,rtol=2e-5)


def test_cross_segment_patch_index_not_temporally_connected():
    seg=torch.tensor([[0,0,1,1]])
    mask=FactorizedCausalBlock.temporal_allow_mask(seg)[0]
    assert not mask[2,0] and not mask[2,1]
    assert mask[3,2] and mask[3,3]
    assert mask[1,0] and not mask[0,1]
