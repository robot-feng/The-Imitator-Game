import torch
from reswam.history_teacher import HistoryDecoder,complementary_memory_loss
from reswam.memory import ResidualFastMemory


def test_teacher_offline_residual_and_runtime_writer_separation():
    torch.manual_seed(2)
    teacher=HistoryDecoder(32,16,heads=4)
    memory=ResidualFastMemory(32,16,2,2,4)
    b=2
    current=torch.randn(b,4,32)
    explicit=torch.randn(b,3,32)
    target=torch.randn(b,4,32)
    addrs=memory.default_addresses[None].expand(b,-1,-1)
    w=memory.initial_state(b)
    w=memory.write(w,current,segment="robot",proprio=torch.zeros(b,2))
    err=teacher.reconstruction_loss(current,explicit,addrs,memory.project_value(target))
    assert torch.isfinite(err)
    teacher.eval().requires_grad_(False)
    loss=complementary_memory_loss(memory,w,teacher,current,explicit,addrs,target)
    loss.backward()
    assert memory.key[0].weight.grad is not None
    assert teacher.to_query.weight.grad is None
