from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
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


@dataclass(frozen=True)
class CandidatePlan:
    generation: int
    role: str
    focus_openings: tuple[str, ...] = ()
    opening_fraction: float = 0.0
    donor_generations: tuple[int, ...] = ()


@dataclass
class LeagueRow:
    generation: int
    games: int = 0
    points: float = 0.0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    rating: float = 1500.0
    opening_points: dict[str, float] = field(default_factory=dict)
    opening_games: dict[str, int] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.points / self.games if self.games else 0.0

    def opening_score(self, opening: str) -> float:
        n = self.opening_games.get(opening, 0)
        return self.opening_points.get(opening, 0.0) / n if n else 0.0


@dataclass(frozen=True)
class LeagueSummary:
    round_id: int
    ranking: list[dict[str, Any]]
    top_generation: int
    specialist_generations: dict[str, int]


class LeagueTable:
    def __init__(self, generations: list[int], *, k_factor: float = 20.0):
        self.rows = {int(g): LeagueRow(int(g)) for g in generations}
        self.k_factor = float(k_factor)

    @staticmethod
    def expected(a: float, b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((b - a) / 400.0))

    def add(self, white: int, black: int, white_score: float, opening: str) -> None:
        wr = self.rows[int(white)]
        br = self.rows[int(black)]
        ws = float(white_score)
        bs = 1.0 - ws
        we = self.expected(wr.rating, br.rating)
        be = 1.0 - we
        old_wr, old_br = wr.rating, br.rating
        for row, score in ((wr, ws), (br, bs)):
            row.games += 1
            row.points += score
            row.opening_games[opening] = row.opening_games.get(opening, 0) + 1
            row.opening_points[opening] = row.opening_points.get(opening, 0.0) + score
            if score == 1.0:
                row.wins += 1
            elif score == 0.5:
                row.draws += 1
            else:
                row.losses += 1
        wr.rating = old_wr + self.k_factor * (ws - we)
        br.rating = old_br + self.k_factor * (bs - be)

    def ranking(self) -> list[LeagueRow]:
        return sorted(self.rows.values(), key=lambda r: (r.rating, r.score, r.games), reverse=True)


class PopulationArena:
    """Cheap league screening before the held-out champion promotion gate.

    League games decide where to spend evaluation budget and identify niches. They
    never directly switch the champion. The existing Arena promotion gate remains
    the final safety boundary.
    """

    def __init__(self, config: dict[str, Any], memory: MemoryStore, device: torch.device):
        self.config = config
        self.memory = memory
        self.device = device

    def run(
        self,
        *,
        round_id: int,
        champion_generation: int,
        members: dict[int, tuple[ChessNet, AgentGenome]],
        candidate_plans: list[CandidatePlan],
    ) -> LeagueSummary:
        lcfg = self.config.get("league", {})
        anchor_pairs = max(1, int(lcfg.get("anchor_pairs", 1)))
        playoff_pairs = max(1, int(lcfg.get("playoff_pairs", 1)))
        depth = int(lcfg.get("depth", self.config["arena"].get("depth", self.config["search"]["depth"])))
        max_plies = int(lcfg.get("max_game_plies", self.config["arena"].get("max_game_plies", 220)))
        k_factor = float(lcfg.get("k_factor", 20.0))

        generations = [champion_generation] + [p.generation for p in candidate_plans]
        table = LeagueTable(generations, k_factor=k_factor)
        self._league_played = 0
        self._league_total = len(candidate_plans) * anchor_pairs * 2
        if len(candidate_plans) >= 2:
            self._league_total += playoff_pairs * 2
        print(
            f"[dog_matist][stage=league][detail=0/{self._league_total} depth={depth} max_plies={max_plies}]",
            flush=True,
        )
        searchers: dict[int, AlphaBetaSearcher] = {}
        for generation, (model, genome) in members.items():
            model.to(self.device).eval()
            searchers[generation] = AlphaBetaSearcher(
                HybridEvaluator(model, self.config, self.device, genome), self.config
            )

        # Phase 1: every candidate gets the same number of paired-opening anchor
        # games against the champion. This is much cheaper than full round robin.
        seed = int(self.config["project"].get("seed", 0)) + round_id * 10007
        curriculum = OpeningCurriculum(seed=seed)
        anchors = curriculum.arena_pairs(anchor_pairs * max(1, len(candidate_plans)))
        cursor = 0
        for plan in candidate_plans:
            pair_seeds = anchors[cursor:cursor + anchor_pairs]
            cursor += anchor_pairs
            self._play_paired_set(
                round_id, table, plan.generation, champion_generation,
                searchers[plan.generation], searchers[champion_generation],
                pair_seeds, depth, max_plies,
            )

        # Phase 2: only the two best challengers play each other. This Swiss-like
        # concentration of budget avoids O(n^2) weak-vs-weak games.
        challengers = [r for r in table.ranking() if r.generation != champion_generation]
        if len(challengers) >= 2:
            a, b = challengers[0].generation, challengers[1].generation
            playoff = OpeningCurriculum(seed=seed + 7919).arena_pairs(playoff_pairs)
            self._play_paired_set(round_id, table, a, b, searchers[a], searchers[b], playoff, depth, max_plies)

        ranking_rows = table.ranking()
        for row in ranking_rows:
            if row.generation == champion_generation:
                continue
            self.memory.update_population_member(
                round_id, row.generation, league_score=row.score, rating=row.rating,
                status="league_complete",
            )

        specialists = self._archive_specialists(round_id, table, champion_generation)
        top = next((r.generation for r in ranking_rows if r.generation != champion_generation), champion_generation)
        return LeagueSummary(
            round_id=round_id,
            ranking=[
                {
                    "generation": r.generation,
                    "games": r.games,
                    "score": r.score,
                    "rating": r.rating,
                    "wins": r.wins,
                    "draws": r.draws,
                    "losses": r.losses,
                }
                for r in ranking_rows
            ],
            top_generation=top,
            specialist_generations=specialists,
        )

    def _play_paired_set(
        self,
        round_id: int,
        table: LeagueTable,
        first_generation: int,
        second_generation: int,
        first_searcher: AlphaBetaSearcher,
        second_searcher: AlphaBetaSearcher,
        seeds: list[tuple[chess.Board, str]],
        depth: int,
        max_plies: int,
    ) -> None:
        for pair_index, (start_board, opening_name) in enumerate(seeds):
            for first_white in (True, False):
                if first_white:
                    white_g, black_g = first_generation, second_generation
                    white_s, black_s = first_searcher, second_searcher
                else:
                    white_g, black_g = second_generation, first_generation
                    white_s, black_s = second_searcher, first_searcher
                print(
                    f"[dog_matist][stage=league][detail={self._league_played}/{self._league_total} playing] "
                    f"g{white_g} vs g{black_g} opening={opening_name}", flush=True,
                )
                record = play_game(
                    white_s, black_s, self.config,
                    white_name=f"league-g{white_g}",
                    black_name=f"league-g{black_g}",
                    stochastic=False,
                    seed=round_id * 1_000_000 + pair_index * 2 + int(first_white),
                    depth=depth,
                    max_plies=max_plies,
                    starting_board=start_board,
                    opening_name=opening_name,
                    opening_family="league",
                )
                if record.winner is chess.WHITE:
                    white_score = 1.0
                elif record.winner is None:
                    white_score = 0.5
                else:
                    white_score = 0.0
                gid = self.memory.add_game(
                    source="league",
                    generation=first_generation,
                    white_agent=f"league-g{white_g}",
                    black_agent=f"league-g{black_g}",
                    result=record.result,
                    termination=record.termination,
                    pgn=record.pgn,
                    plies=record.plies,
                    examples=[],
                    metadata={**record.metadata, "population_round": round_id, "paired_colors": True},
                )
                self.memory.add_league_match(
                    round_id, white_g, black_g, gid, opening_name, record.result, white_score
                )
                table.add(white_g, black_g, white_score, opening_name)
                self._league_played += 1
                print(
                    f"[dog_matist][stage=league][detail={self._league_played}/{self._league_total}] "
                    f"result={record.result} opening={opening_name}", flush=True,
                )

    def _archive_specialists(self, round_id: int, table: LeagueTable, champion_generation: int) -> dict[str, int]:
        cfg = self.config.get("league", {})
        min_games = max(1, int(cfg.get("specialist_min_games", 2)))
        margin = float(cfg.get("specialist_margin", 0.15))
        floor = float(cfg.get("specialist_score", 0.65))
        champion = table.rows[champion_generation]
        winners: dict[str, int] = {}
        openings = {o for row in table.rows.values() for o in row.opening_games}
        for opening in openings:
            champ_n = champion.opening_games.get(opening, 0)
            champ_score = champion.opening_score(opening) if champ_n else 0.5
            best: tuple[float, int, int] | None = None
            for generation, row in table.rows.items():
                if generation == champion_generation:
                    continue
                n = row.opening_games.get(opening, 0)
                if n < min_games:
                    continue
                score = row.opening_score(opening)
                if score < floor or score < champ_score + margin:
                    continue
                key = (score, n, generation)
                if best is None or key > best:
                    best = key
            if best is None:
                continue
            score, games, generation = best
            self.memory.upsert_specialist(
                generation, opening, score, games,
                {"round_id": round_id, "champion_generation": champion_generation, "champion_score": champ_score},
            )
            winners[opening] = generation
        return winners


def choose_focus_openings(memory: MemoryStore, *, count: int = 4) -> tuple[str, ...]:
    """Prefer openings that lifetime self-play has under-explored.

    Existing active specialists are also included so useful niches remain in the
    curriculum instead of disappearing after a failed overall promotion.
    """
    from .opening_curriculum import CURATED_OPENINGS

    counts = memory.recent_opening_counts(600)
    all_names = [o.name for o in CURATED_OPENINGS]
    underplayed = sorted(all_names, key=lambda name: (counts.get(name, 0), name))
    specialist_names = [str(r["opening_name"]) for r in memory.active_specialists(limit=count)]
    out: list[str] = []
    for name in specialist_names + underplayed:
        if name not in out:
            out.append(name)
        if len(out) >= count:
            break
    return tuple(out)
