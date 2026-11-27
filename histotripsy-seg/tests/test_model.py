import torch
from src.model import UNet
from src.losses import DiceBCELoss


def test_forward_shape():
    m = UNet(base=8)
    x = torch.randn(2, 1, 64, 64)
    assert m(x).shape == (2, 1, 64, 64)


def test_loss_backward():
    m = UNet(base=8)
    x = torch.randn(2, 1, 64, 64)
    y = (torch.rand(2, 1, 64, 64) > 0.5).float()
    loss = DiceBCELoss()(m(x), y)
    loss.backward()
    assert loss.item() > 0
