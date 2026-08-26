from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import chess
import torch

from .evaluator import HybridEvaluator
from .genome import AgentGenome
from .memory import MemoryStore
from .network import ChessNet
from .opening_curriculum import OpeningCurriculum
from .search import AlphaBetaSearcher
from .selfplay import play_game


@dataclass
class ArenaResult:
    games: int
    wins: int
    draws: int
    losses: int
    score: float
    wilson_lower: float
    promoted: bool


def _wilson_interval(score: float, n: int, z: float) -> tuple[float, float]:
    n = max(1, int(n))
    score = max(0.0, min(1.0, float(score)))
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (score + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt(score * (1.0 - score) / n + z2 / (4.0 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


class Arena:
    def __init__(self, config: dict[str, Any], memory: MemoryStore, device: torch.device):
        self.config = config
        self.memory = memory
        self.device = device

    def compare(
        self,
        challenger: ChessNet,
        champion: ChessNet,
        *,
        challenger_generation: int,
        champion_generation: int,
        challenger_genome: AgentGenome,
        champion_genome: AgentGenome,
        games: int | None = None,
    ) -> ArenaResult:
        acfg = self.config["arena"]
        explicit_games = games is not None
        requested_games = int(games or acfg["games"])
        depth = int(acfg.get("depth", self.config["search"]["depth"]))
        max_plies = int(acfg.get("max_game_plies", 220))
        threshold = float(acfg.get("promotion_score", 0.55))
        wilson_z = float(acfg.get("promotion_wilson_z", 1.2816))

        def even(n: int) -> int:
            n = max(2, int(n))
            return n if n % 2 == 0 else n + 1

        if explicit_games:
            min_games = max_games = even(requested_games)
            adaptive = False
        else:
            # V1 used a tiny fixed Arena plus a confidence guard. At 8-10 games
            # that guard effectively demanded an implausibly high score, causing
            # strong-but-not-dominant candidates to be rejected before they could
            # gather enough evidence. V2 evaluates sequentially: obvious losers
            # stop early, ambiguous candidates earn more paired games, and only a
            # statistically supported challenger can promote.
            min_games = even(int(acfg.get("min_games", requested_games)))
            max_games = even(int(acfg.get("max_games", max(min_games, requested_games))))
            max_games = max(min_games, max_games)
            adaptive = bool(acfg.get("adaptive", True))

        print(f"[dog_matist][stage=arena][detail=0/{max_games} min={min_games} adaptive={adaptive}]", flush=True)
        challenger.eval()
        champion.eval()
        ce = HybridEvaluator(challenger, self.config, self.device, challenger_genome)
        pe = HybridEvaluator(champion, self.config, self.device, champion_genome)
        cs = AlphaBetaSearcher(ce, self.config)
        ps = AlphaBetaSearcher(pe, self.config)

        seed = int(self.config["project"].get("seed", 0)) + challenger_generation * 1009 + champion_generation
        curriculum = OpeningCurriculum(seed=seed)
        pairs = curriculum.arena_pairs(max_games // 2)

        wins = draws = losses = 0
        played = 0
        stopped_early = False
        for pair_index, (start_board, opening_name) in enumerate(pairs):
            for challenger_white in (True, False):
                i = played
                if challenger_white:
                    record = play_game(
                        cs, ps, self.config,
                        white_name=f"challenger-g{challenger_generation}",
                        black_name=f"champion-g{champion_generation}",
                        stochastic=False,
                        seed=100000 + i,
                        depth=depth,
                        max_plies=max_plies,
                        starting_board=start_board,
                        opening_name=opening_name,
                        opening_family="arena",
                    )
                    if record.winner is chess.WHITE:
                        r = 1.0
                    elif record.winner is None:
                        r = 0.5
                    else:
                        r = 0.0
                    color = "white"
                else:
                    record = play_game(
                        ps, cs, self.config,
                        white_name=f"champion-g{champion_generation}",
                        black_name=f"challenger-g{challenger_generation}",
                        stochastic=False,
                        seed=100000 + i,
                        depth=depth,
                        max_plies=max_plies,
                        starting_board=start_board,
                        opening_name=opening_name,
                        opening_family="arena",
                    )
                    if record.winner is chess.BLACK:
                        r = 1.0
                    elif record.winner is None:
                        r = 0.5
                    else:
                        r = 0.0
                    color = "black"

                if r == 1.0:
                    wins += 1
                elif r == 0.5:
                    draws += 1
                else:
                    losses += 1

                metadata = dict(record.metadata)
                metadata.update({
                    "arena_pair": pair_index,
                    "opening_name": opening_name,
                    "paired_colors": True,
                    "requested_arena_games": requested_games,
                    "arena_min_games": min_games,
                    "arena_max_games": max_games,
                    "arena_adaptive": adaptive,
                })
                gid = self.memory.add_game(
                    source="arena",
                    generation=challenger_generation,
                    white_agent=f"challenger-g{challenger_generation}" if challenger_white else f"champion-g{champion_generation}",
                    black_agent=f"champion-g{champion_generation}" if challenger_white else f"challenger-g{challenger_generation}",
                    result=record.result,
                    termination=record.termination,
                    pgn=record.pgn,
                    plies=record.plies,
                    examples=[],
                    metadata=metadata,
                )
                self.memory.add_arena_match(challenger_generation, champion_generation, gid, color, r)
                played += 1
                print(
                    f"[dog_matist][stage=arena][detail={played}/{max_games}] opening={opening_name} pair={pair_index + 1}",
                    flush=True,
                )

            # Only inspect after a full color-swapped pair. This preserves the
            # paired-opening fairness guarantee even when evaluation stops early.
            if adaptive and played >= min_games:
                score_now = (wins + 0.5 * draws) / max(1, played)
                lower_now, upper_now = _wilson_interval(score_now, played, wilson_z)
                if score_now >= threshold and lower_now > 0.5:
                    stopped_early = True
                    print(
                        f"[dog_matist][arena] early accept evidence after {played} games "
                        f"(score={score_now:.3f}, lower={lower_now:.3f})", flush=True,
                    )
                    break
                if upper_now < threshold:
                    stopped_early = True
                    print(
                        f"[dog_matist][arena] early reject evidence after {played} games "
                        f"(score={score_now:.3f}, upper={upper_now:.3f})", flush=True,
                    )
                    break

        score = (wins + 0.5 * draws) / max(1, played)
        wilson_lower, _wilson_upper = _wilson_interval(score, max(1, played), wilson_z)
        promoted = score >= threshold and wilson_lower > 0.5
        final_stage = "promoted" if promoted else "rejected"
        print(f"[dog_matist][stage={final_stage}][detail=score {score:.3f}]", flush=True)
        return ArenaResult(played, wins, draws, losses, score, wilson_lower, promoted)
