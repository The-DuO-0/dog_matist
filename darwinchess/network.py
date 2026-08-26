from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .encoding import BOARD_PLANES


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class ChessNet(nn.Module):
    """Compact residual value + factorized policy network.

    Policy is factorized as from-square, to-square and promotion type. Search masks
    illegal combinations, so this remains compact while supporting promotions.
    Value is always from the side-to-move perspective in [-1, 1].
    """

    def __init__(self, channels: int = 64, residual_blocks: int = 4):
        super().__init__()
        self.channels = channels
        self.residual_blocks = residual_blocks
        self.stem = nn.Sequential(
            nn.Conv2d(BOARD_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(residual_blocks)])

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 8, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
        self.policy_shared = nn.Sequential(
            nn.Conv2d(channels, 8, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        policy_dim = 8 * 8 * 8
        self.from_head = nn.Linear(policy_dim, 64)
        self.to_head = nn.Linear(policy_dim, 64)
        self.promo_head = nn.Linear(policy_dim, 5)

        # Start generation 0 as a stable classical-search baseline: the neural
        # value contributes 0 and policy ordering is uniform until experience
        # gives the network evidence to change them.
        nn.init.zeros_(self.value_head[-2].weight)
        nn.init.zeros_(self.value_head[-2].bias)
        nn.init.zeros_(self.from_head.weight)
        nn.init.zeros_(self.from_head.bias)
        nn.init.zeros_(self.to_head.weight)
        nn.init.zeros_(self.to_head.bias)
        nn.init.zeros_(self.promo_head.weight)
        nn.init.zeros_(self.promo_head.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.body(self.stem(x))
        p = self.policy_shared(z)
        return {
            "value": self.value_head(z).squeeze(-1),
            "from_logits": self.from_head(p),
            "to_logits": self.to_head(p),
            "promo_logits": self.promo_head(p),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ChessNet":
        m = config["model"]
        return cls(channels=int(m["channels"]), residual_blocks=int(m["residual_blocks"]))


def save_checkpoint(
    path: str | Path,
    model: ChessNet,
    *,
    generation: int,
    optimizer: torch.optim.Optimizer | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "generation": generation,
        "model_kwargs": {"channels": model.channels, "residual_blocks": model.residual_blocks},
        "model_state": model.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, Path(path))


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[ChessNet, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model = ChessNet(**payload["model_kwargs"])
    model.load_state_dict(payload["model_state"])
    model.to(device)
    return model, payload
