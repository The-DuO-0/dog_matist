from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AgentGenome:
    """Non-weight behavior parameters that evolve together with a checkpoint."""

    classical_mix: float = 1.0
    neural_cp_scale: float = 650.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, config: dict[str, Any]) -> "AgentGenome":
        data = data or {}
        return cls(
            classical_mix=float(data.get("classical_mix", 1.0)),
            neural_cp_scale=float(data.get("neural_cp_scale", config["model"].get("neural_cp_scale", 650.0))),
        )


def propose_child_genome(parent: AgentGenome, config: dict[str, Any], replay_examples: int) -> AgentGenome:
    ecfg = config.get("evolution", {})
    min_mix = float(ecfg.get("min_classical_mix", 0.20))
    step = float(ecfg.get("classical_mix_step", 0.05))
    min_examples = int(ecfg.get("mix_mutation_min_examples", 256))
    new_mix = parent.classical_mix
    if replay_examples >= min_examples:
        new_mix = max(min_mix, round(parent.classical_mix - step, 6))
    return AgentGenome(classical_mix=new_mix, neural_cp_scale=parent.neural_cp_scale)
