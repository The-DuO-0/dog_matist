from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess
import torch
from torch import nn

from .encoding import encode_boards, move_targets
from .memory import MemoryStore
from .network import ChessNet


@dataclass
class TrainingStats:
    steps: int
    examples_used: int
    mean_loss: float
    value_loss: float
    policy_loss: float


class ContinualTrainer:
    def __init__(
        self,
        model: ChessNet,
        memory: MemoryStore,
        config: dict[str, Any],
        device: torch.device,
        optimizer_state: dict[str, Any] | None = None,
    ):
        self.model = model
        self.memory = memory
        self.config = config
        self.device = device
        t = config["training"]
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(t["learning_rate"]),
            weight_decay=float(t.get("weight_decay", 1e-5)),
        )
        if optimizer_state:
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except (ValueError, KeyError):
                pass

    def train(
        self,
        steps: int | None = None,
        *,
        opening_focus: list[str] | tuple[str, ...] | None = None,
        opening_fraction: float = 0.0,
        opening_generations: list[int] | tuple[int, ...] | None = None,
        recent_fraction_override: float | None = None,
    ) -> TrainingStats:
        tcfg = self.config["training"]
        steps = int(steps or tcfg["steps_per_cycle"])
        batch_size = int(tcfg["batch_size"])
        recent_fraction = float(tcfg.get("replay_recent_fraction", 0.35)) if recent_fraction_override is None else float(recent_fraction_override)
        value_weight = float(tcfg.get("value_loss_weight", 1.0))
        policy_weight = float(tcfg.get("policy_loss_weight", 0.35))
        grad_clip = float(tcfg.get("gradient_clip", 1.0))

        print(f"[dog_matist][stage=training][detail=0/{steps}]", flush=True)
        self.model.to(self.device)
        self.model.train()
        total_loss = total_v = total_p = 0.0
        used = 0
        completed = 0
        report_every = max(1, steps // 20)

        for step in range(steps):
            rows = self.memory.replay_sample(
                batch_size,
                recent_fraction,
                opening_names=opening_focus,
                opening_fraction=opening_fraction,
                generations=opening_generations,
            )
            if not rows:
                break
            boards: list[chess.Board] = []
            moves: list[chess.Move] = []
            values: list[float] = []
            weights: list[float] = []
            for row in rows:
                try:
                    board = chess.Board(row["fen"])
                    move = chess.Move.from_uci(row["move_uci"])
                    if move not in board.legal_moves:
                        continue
                except Exception:
                    continue
                boards.append(board)
                moves.append(move)
                values.append(float(row["value_target"]))
                weights.append(float(row["policy_weight"]) * min(3.0, float(row["priority"])))
            if not boards:
                continue

            x = encode_boards(boards, self.device)
            value_target = torch.tensor(values, dtype=torch.float32, device=self.device)
            from_target = torch.tensor([move_targets(m)[0] for m in moves], dtype=torch.long, device=self.device)
            to_target = torch.tensor([move_targets(m)[1] for m in moves], dtype=torch.long, device=self.device)
            promo_target = torch.tensor([move_targets(m)[2] for m in moves], dtype=torch.long, device=self.device)
            sample_weight = torch.tensor(weights, dtype=torch.float32, device=self.device)
            sample_weight = sample_weight / sample_weight.mean().clamp_min(1e-6)

            out = self.model(x)
            value_losses = (out["value"] - value_target).pow(2)
            vloss = (value_losses * sample_weight).mean()
            fl = nn.functional.cross_entropy(out["from_logits"], from_target, reduction="none")
            tl = nn.functional.cross_entropy(out["to_logits"], to_target, reduction="none")
            pl = nn.functional.cross_entropy(out["promo_logits"], promo_target, reduction="none")
            policy_losses = fl + tl + 0.35 * pl
            ploss = (policy_losses * sample_weight).mean()
            loss = value_weight * vloss + policy_weight * ploss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_v += float(vloss.detach().cpu())
            total_p += float(ploss.detach().cpu())
            used += len(boards)
            completed += 1
            if completed == 1 or completed % report_every == 0 or completed == steps:
                running_loss = total_loss / max(1, completed)
                print(
                    f"[dog_matist][stage=training][detail={completed}/{steps} loss={running_loss:.5f}]",
                    flush=True,
                )

        self.model.eval()
        denom = max(1, completed)
        return TrainingStats(
            steps=completed,
            examples_used=used,
            mean_loss=total_loss / denom,
            value_loss=total_v / denom,
            policy_loss=total_p / denom,
        )
