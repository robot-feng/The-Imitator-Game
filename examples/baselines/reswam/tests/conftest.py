import pytest,torch
from reswam import ResWAM,ResWAMConfig

@pytest.fixture
def tiny_model():
    torch.manual_seed(19)
    model=ResWAM(ResWAMConfig.tiny(),load_vision=False)
    model.eval()
    return model
