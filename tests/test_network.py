import torch

from darwinchess.network import ChessNet


def test_network_shapes_and_zeroed_heads():
    model = ChessNet(channels=16, residual_blocks=1).eval()
    x = torch.zeros((2, 18, 8, 8))
    out = model(x)
    assert out["value"].shape == (2,)
    assert out["from_logits"].shape == (2, 64)
    assert out["to_logits"].shape == (2, 64)
    assert out["promo_logits"].shape == (2, 5)
    assert torch.allclose(out["value"], torch.zeros(2), atol=1e-6)
    assert torch.allclose(out["from_logits"], torch.zeros((2, 64)), atol=1e-6)
