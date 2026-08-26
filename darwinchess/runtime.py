from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import monotonic
from typing import Any
import copy
import json
import os
import platform

import chess
import torch

from .arena import Arena, ArenaResult
from .config import apply_mode, choose_device, configure_runtime, ensure_state_dirs, load_config, seed_everything
from .evaluator import HybridEvaluator
from .genome import AgentGenome, propose_child_genome
from .locks import EvolutionLock
from .memory import MemoryStore
from .network import ChessNet, load_checkpoint, save_checkpoint
from .reflection import ReflectionEngine
from .search import AlphaBetaSearcher, SearchResult
from .teacher import StockfishTeacher, find_stockfish
from .trainer import ContinualTrainer, TrainingStats
from .selfplay import play_game
from .opening_curriculum import CURATED_OPENINGS
from .population import CandidatePlan, LeagueSummary, PopulationArena, choose_focus_openings
from .parallel_selfplay import SelfPlayTask, parallel_selfplay
from .resource_control import choose_worker_budget


class DarwinRuntime:
    def __init__(
        self, config_path: str | Path | None = None, *, mode: str | None = None,
        device: str | None = None, search_device: str | None = None,
        apply_nice: bool = True,
    ):
        base = load_config(config_path)
        mode = mode or base["resources"].get("default_mode", "normal")
        self.config = apply_mode(base, mode)
        configure_runtime(self.config, apply_nice=apply_nice)
        seed_everything(int(self.config["project"].get("seed", 0)))
        self.paths = ensure_state_dirs(self.config)
        self.memory = MemoryStore(self.paths["db"])
        self.device = torch.device(device) if device else choose_device(True)
        configured_search = search_device or self.config.get("runtime", {}).get("search_device", "cpu")
        self.search_device = torch.device(configured_search)
        self._last_resource_budget: dict[str, Any] | None = None
        self._ensure_initialized()

    def close(self) -> None:
        self.memory.close()

    def __enter__(self) -> "DarwinRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def mode(self) -> str:
        return self.config.get("runtime", {}).get("mode", "normal")

    def _ensure_initialized(self) -> None:
        champion = self.memory.champion_generation()
        if champion is not None:
            if Path(champion["checkpoint_path"]).exists():
                return
            raise RuntimeError(
                f"Champion checkpoint is missing: {champion['checkpoint_path']}. "
                "dog_matist refuses to silently reset lifetime state."
            )
        existing = self.memory.conn.execute("SELECT COUNT(*) AS n FROM generations").fetchone()["n"]
        if int(existing) > 0:
            raise RuntimeError("Generation history exists but no active champion is marked; refusing automatic reset.")
        model = ChessNet.from_config(self.config)
        initial_genome = AgentGenome(
            classical_mix=float(self.config.get("evolution", {}).get("initial_classical_mix", 1.0)),
            neural_cp_scale=float(self.config["model"].get("neural_cp_scale", 650.0)),
        )
        path = self.paths["checkpoints"] / "generation_000000_champion.pt"
        save_checkpoint(
            path,
            model,
            generation=0,
            metadata={
                "origin": "classical-baseline-with-zeroed-neural-heads",
                "created_by": "dog_matist",
                "genome": initial_genome.to_dict(),
            },
        )
        self.memory.add_generation(
            0, None, str(path), "champion",
            notes="Initial classical-search baseline; neural heads zeroed.",
            genome=initial_genome.to_dict(),
        )
        self.memory.set_meta("created", True)
        self.memory.set_meta("project_name", self.config["project"]["name"])

    def _retire_stale_challengers(self) -> None:
        """Retire only challengers left behind before this Evolution lock was acquired."""
        with self.memory.conn:
            self.memory.conn.execute(
                "UPDATE generations SET status='aborted' WHERE status IN ('challenger','candidate')"
            )

    def champion_info(self):
        row = self.memory.champion_generation()
        if row is None:
            raise RuntimeError("No champion generation exists")
        return row

    def load_champion(self, device: torch.device | None = None) -> tuple[ChessNet, dict[str, Any]]:
        row = self.champion_info()
        return load_checkpoint(row["checkpoint_path"], device or self.device)

    def genome_from_payload(self, payload: dict[str, Any]) -> AgentGenome:
        return AgentGenome.from_dict(payload.get("metadata", {}).get("genome"), self.config)

    def generation_genome(self, generation_id: int) -> AgentGenome:
        row = self.memory.get_generation(generation_id)
        if row is None:
            raise RuntimeError(f"Unknown generation {generation_id}")
        try:
            data = json.loads(row["genome_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            data = {}
        if data:
            return AgentGenome.from_dict(data, self.config)
        _, payload = load_checkpoint(row["checkpoint_path"], torch.device("cpu"))
        return self.genome_from_payload(payload)

    def make_searcher(
        self,
        model: ChessNet | None = None,
        *,
        genome: AgentGenome | None = None,
        device: torch.device | None = None,
    ) -> AlphaBetaSearcher:
        d = device or self.search_device
        if model is None:
            model, payload = self.load_champion(d)
            genome = genome or self.genome_from_payload(payload)
        if genome is None:
            genome = AgentGenome.from_dict(None, self.config)
        model.to(d).eval()
        return AlphaBetaSearcher(HybridEvaluator(model, self.config, d, genome), self.config)

    def selfplay(self, games: int | None = None) -> list[str]:
        champion = self.champion_info()
        generation = int(champion["id"])
        games = int(games or self.config["selfplay"]["games_per_cycle"])
        ids: list[str] = []
        base_seed = int(self.config["project"].get("seed", 0)) + self.memory.count_games() * 17
        runtime_cfg = self.config.get("runtime", {})
        budget = choose_worker_budget(self.config)
        self._last_resource_budget = budget.to_dict()
        # Persist the most recent budget so a separate Studio/status process can
        # report what the overnight worker controller actually chose.
        self.memory.set_meta("last_resource_budget", self._last_resource_budget)
        use_parallel = bool(runtime_cfg.get("parallel_selfplay", False)) and str(self.search_device) == "cpu"
        workers = max(1, int(budget.selfplay_workers))

        print(
            f"[dog_matist][stage=self-play][detail=0/{games} workers={workers if use_parallel else 1} "
            f"resource={budget.reason}]", flush=True,
        )
        records = None
        if use_parallel and workers > 1 and games > 1:
            tasks = [
                SelfPlayTask(
                    checkpoint_path=str(champion["checkpoint_path"]),
                    generation=generation,
                    config=self.config,
                    seed=base_seed + i,
                    depth=int(self.config["search"]["depth"]),
                )
                for i in range(games)
            ]
            try:
                records = parallel_selfplay(tasks, workers)
            except Exception as exc:
                # Parallelism is an optimization, never a reason to lose an
                # overnight run. Fall back to the proven sequential path.
                print(f"[dog_matist][self-play] parallel fallback: {exc}", flush=True)

        if records is None:
            model, payload = self.load_champion(self.search_device)
            genome = self.genome_from_payload(payload)
            evaluator = HybridEvaluator(model, self.config, self.search_device, genome)
            searcher = AlphaBetaSearcher(evaluator, self.config)
            records = [
                play_game(
                    searcher, searcher, self.config,
                    white_name=f"dog_matist-g{generation}",
                    black_name=f"dog_matist-g{generation}",
                    stochastic=True, seed=base_seed + i,
                    depth=int(self.config["search"]["depth"]),
                )
                for i in range(games)
            ]

        for i, record in enumerate(records):
            gid = self.memory.add_game(
                source="selfplay", generation=generation,
                white_agent=f"dog_matist-g{generation}", black_agent=f"dog_matist-g{generation}",
                result=record.result, termination=record.termination, pgn=record.pgn,
                plies=record.plies, examples=record.examples, metadata=record.metadata,
            )
            ids.append(gid)
            opening = record.metadata.get("opening_name", "Initial position")
            print(f"[dog_matist][stage=self-play][detail={i + 1}/{games}] opening={opening}", flush=True)
        self.memory.update_generation(
            generation, games_seen=self.memory.count_games(), examples_seen=self.memory.count_examples()
        )
        self.memory.prune_replay(int(self.config["training"].get("replay_capacity", 2_000_000)))
        return ids

    def train_challenger(self, steps: int | None = None) -> tuple[int, ChessNet, TrainingStats, Path] | None:
        min_examples = int(self.config["training"].get("min_examples_before_training", 256))
        if self.memory.count_examples() < min_examples:
            return None
        champion = self.champion_info()
        parent_id = int(champion["id"])
        challenger_id = self.memory.next_generation_id()
        champion_model, parent_payload = self.load_champion(self.device)
        parent_genome = self.genome_from_payload(parent_payload)
        child_genome = propose_child_genome(parent_genome, self.config, self.memory.count_examples())
        challenger = copy.deepcopy(champion_model).to(self.device)
        trainer = ContinualTrainer(
            challenger, self.memory, self.config, self.device,
            optimizer_state=parent_payload.get("optimizer_state"),
        )
        stats = trainer.train(steps)
        path = self.paths["checkpoints"] / f"generation_{challenger_id:06d}_challenger.pt"
        save_checkpoint(
            path,
            challenger,
            generation=challenger_id,
            optimizer=trainer.optimizer,
            metadata={
                "parent_generation": parent_id,
                "training": asdict(stats),
                "genome": child_genome.to_dict(),
            },
        )
        self.memory.add_generation(
            challenger_id,
            parent_id,
            str(path),
            "challenger",
            notes=f"Trained from champion g{parent_id}; proposed genome {child_genome.to_dict()}",
            training_loss=stats.mean_loss,
            genome=child_genome.to_dict(),
        )
        return challenger_id, challenger, stats, path

    def gate_challenger(self, challenger_id: int, challenger: ChessNet, *, games: int | None = None) -> ArenaResult:
        champion = self.champion_info()
        champion_id = int(champion["id"])
        champion_model, champion_payload = self.load_champion(self.search_device)
        champion_genome = self.genome_from_payload(champion_payload)
        challenger_genome = self.generation_genome(challenger_id)
        challenger = challenger.to(self.search_device)
        arena = Arena(self.config, self.memory, self.search_device)
        result = arena.compare(
            challenger,
            champion_model,
            challenger_generation=challenger_id,
            champion_generation=champion_id,
            challenger_genome=challenger_genome,
            champion_genome=champion_genome,
            games=games,
        )
        self.memory.update_generation(
            challenger_id,
            arena_score=result.score,
            arena_wins=result.wins,
            arena_draws=result.draws,
            arena_losses=result.losses,
        )
        if result.promoted:
            row = self.memory.get_generation(challenger_id)
            old_path = Path(row["checkpoint_path"])
            new_path = self.paths["checkpoints"] / f"generation_{challenger_id:06d}_champion.pt"
            try:
                old_path.replace(new_path)
            except OSError:
                new_path = old_path
            self.memory.promote_generation(champion_id, challenger_id, str(new_path))
            self.memory.add_insight(
                challenger_id,
                "promotion",
                f"Generation {challenger_id} 击败 generation {champion_id}，以 arena score {result.score:.3f} 成为新 champion；classical_mix={challenger_genome.classical_mix:.2f}。",
                asdict(result),
            )
        else:
            self.memory.update_generation(challenger_id, status="rejected")
            self.memory.add_insight(
                champion_id,
                "rejection",
                f"Generation {challenger_id} 的改动未通过 arena（score {result.score:.3f}），champion 保持为 generation {champion_id}。",
                asdict(result),
            )
        return result

    def train_population(
        self,
        *,
        round_id: int,
        total_steps: int | None = None,
    ) -> tuple[list[CandidatePlan], dict[int, ChessNet], dict[int, TrainingStats]] | None:
        """Train several candidate branches with roughly one old-cycle GPU budget.

        One shared update consumes most of the steps. Candidate roles then receive
        short sequential fine-tunes from that shared base, so MPS never has several
        independent trainers fighting for the same GPU.
        """
        min_examples = int(self.config["training"].get("min_examples_before_training", 256))
        if self.memory.count_examples() < min_examples:
            return None
        lcfg = self.config.get("league", {})
        population_size = max(2, int(lcfg.get("population_size", 3)))
        total_steps = int(total_steps or self.config["training"]["steps_per_cycle"])
        shared_fraction = max(0.25, min(0.85, float(lcfg.get("shared_step_fraction", 0.55))))
        shared_steps = max(1, int(round(total_steps * shared_fraction)))
        branch_steps = max(1, (max(population_size, total_steps - shared_steps)) // population_size)

        champion = self.champion_info()
        parent_id = int(champion["id"])
        champion_model, parent_payload = self.load_champion(self.device)
        parent_genome = self.genome_from_payload(parent_payload)
        child_genome = propose_child_genome(parent_genome, self.config, self.memory.count_examples())

        print(
            f"[dog_matist][stage=population-train][detail=shared {shared_steps} + "
            f"{population_size}x{branch_steps} steps]", flush=True,
        )
        shared_model = copy.deepcopy(champion_model).to(self.device)
        shared_trainer = ContinualTrainer(
            shared_model, self.memory, self.config, self.device,
            optimizer_state=parent_payload.get("optimizer_state"),
        )
        shared_trainer.train(shared_steps)
        shared_cpu = copy.deepcopy(shared_model).to(torch.device("cpu"))
        del shared_model

        focus = choose_focus_openings(self.memory, count=max(4, population_size + 1))
        specialist_rows = self.memory.active_specialists(limit=8)
        specialist_focus_list: list[str] = []
        specialist_donors: list[int] = []
        for row in specialist_rows:
            name = str(row["opening_name"])
            if name in specialist_focus_list:
                continue
            specialist_focus_list.append(name)
            specialist_donors.append(int(row["generation"]))
            if len(specialist_focus_list) >= 4:
                break
        specialist_focus = tuple(specialist_focus_list)
        specialist_donor_generations = tuple(specialist_donors)
        roles = ["balanced", "explorer", "specialist", "recent"]
        plans: list[CandidatePlan] = []
        models: dict[int, ChessNet] = {}
        stats_by_generation: dict[int, TrainingStats] = {}

        for idx in range(population_size):
            generation_id = self.memory.next_generation_id()
            role = roles[idx % len(roles)]
            if role == "balanced":
                role_focus: tuple[str, ...] = ()
                opening_fraction = 0.0
                donor_generations: tuple[int, ...] = ()
                recent_override = None
            elif role == "explorer":
                role_focus = tuple(focus[:4])
                opening_fraction = float(lcfg.get("explorer_opening_fraction", 0.65))
                donor_generations = ()
                recent_override = 0.25
            elif role == "specialist":
                role_focus = specialist_focus or tuple(focus[:3])
                opening_fraction = float(lcfg.get("specialist_opening_fraction", 0.75))
                donor_generations = specialist_donor_generations if specialist_focus else ()
                recent_override = 0.35
            else:
                role_focus = ()
                opening_fraction = 0.0
                donor_generations = ()
                recent_override = 0.70

            branch = copy.deepcopy(shared_cpu).to(self.device)
            trainer = ContinualTrainer(branch, self.memory, self.config, self.device)
            stats = trainer.train(
                branch_steps,
                opening_focus=list(role_focus),
                opening_fraction=opening_fraction,
                opening_generations=donor_generations,
                recent_fraction_override=recent_override,
            )
            path = self.paths["checkpoints"] / f"generation_{generation_id:06d}_candidate.pt"
            save_checkpoint(
                path, branch, generation=generation_id, optimizer=trainer.optimizer,
                metadata={
                    "parent_generation": parent_id,
                    "population_round": round_id,
                    "role": role,
                    "focus_openings": list(role_focus),
                    "donor_generations": list(donor_generations),
                    "shared_steps": shared_steps,
                    "branch_steps": branch_steps,
                    "training": asdict(stats),
                    "genome": child_genome.to_dict(),
                },
            )
            self.memory.add_generation(
                generation_id, parent_id, str(path), "candidate",
                notes=f"Population round {round_id}; role={role}; focus={list(role_focus)}",
                training_loss=stats.mean_loss, genome=child_genome.to_dict(),
            )
            self.memory.add_population_member(round_id, generation_id, role, role_focus)
            plans.append(CandidatePlan(generation_id, role, role_focus, opening_fraction, donor_generations))
            stats_by_generation[generation_id] = stats
            models[generation_id] = branch.to(torch.device("cpu")).eval()
            print(
                f"[dog_matist][stage=population-train][detail=g{generation_id} role={role} loss={stats.mean_loss:.5f}]",
                flush=True,
            )
        return plans, models, stats_by_generation

    def _harvest_specialist_experience(
        self,
        summary: LeagueSummary,
        models: dict[int, ChessNet],
    ) -> int:
        """Turn specialist checkpoints into replay so niche knowledge survives.

        This is the key difference from merely keeping a rejected file on disk:
        specialist moves become future training examples even when that generation
        never becomes overall champion.
        """
        harvest_games = max(0, int(self.config.get("league", {}).get("specialist_harvest_games", 1)))
        if harvest_games <= 0 or not summary.specialist_generations:
            return 0
        opening_map = {o.name: o for o in CURATED_OPENINGS}
        harvested = 0
        total_harvest = len(summary.specialist_generations) * harvest_games
        if total_harvest:
            print(f"[dog_matist][stage=specialist-harvest][detail=0/{total_harvest}]", flush=True)
        for opening_name, generation in summary.specialist_generations.items():
            seed = opening_map.get(opening_name)
            model = models.get(generation)
            if seed is None or model is None:
                continue
            genome = self.generation_genome(generation)
            searcher = self.make_searcher(model, genome=genome, device=torch.device("cpu"))
            for i in range(harvest_games):
                record = play_game(
                    searcher, searcher, self.config,
                    white_name=f"specialist-g{generation}", black_name=f"specialist-g{generation}",
                    stochastic=True,
                    seed=int(self.config["project"].get("seed", 0)) + generation * 313 + i,
                    depth=int(self.config["search"]["depth"]),
                    starting_board=seed.board(),
                    opening_name=opening_name, opening_family="specialist",
                )
                self.memory.add_game(
                    source="specialist_selfplay", generation=generation,
                    white_agent=f"specialist-g{generation}", black_agent=f"specialist-g{generation}",
                    result=record.result, termination=record.termination, pgn=record.pgn,
                    plies=record.plies, examples=record.examples,
                    metadata={**record.metadata, "population_round": summary.round_id, "inherited_specialist": True},
                )
                harvested += 1
                print(
                    f"[dog_matist][stage=specialist-harvest][detail={harvested}/{total_harvest}] "
                    f"g={generation} opening={opening_name}", flush=True,
                )
        if harvested:
            self.memory.prune_replay(int(self.config["training"].get("replay_capacity", 2_000_000)))
        return harvested

    def _population_evolve_cycle_unlocked(self) -> dict[str, Any]:
        before = int(self.champion_info()["id"])
        game_ids = self.selfplay()
        ReflectionEngine(self.memory).reflect_recent(before, limit=50)
        round_id = self.memory.start_population_round(before, {"mode": self.mode})
        try:
            trained = self.train_population(round_id=round_id)
            if trained is None:
                self.memory.finish_population_round(round_id, "insufficient_replay")
                return {
                    "champion_before": before, "champion_after": before,
                    "population_round": round_id, "selfplay_games": len(game_ids),
                    "trained": False, "reason": "not enough replay examples",
                }
            plans, candidate_models, stats_by_generation = trained
            champion_model, champion_payload = self.load_champion(torch.device("cpu"))
            members: dict[int, tuple[ChessNet, AgentGenome]] = {
                before: (champion_model, self.genome_from_payload(champion_payload))
            }
            for plan in plans:
                members[plan.generation] = (candidate_models[plan.generation], self.generation_genome(plan.generation))

            summary = PopulationArena(self.config, self.memory, torch.device("cpu")).run(
                round_id=round_id, champion_generation=before, members=members, candidate_plans=plans
            )
            harvested = self._harvest_specialist_experience(summary, candidate_models)

            top_id = int(summary.top_generation)
            top_model = candidate_models[top_id]
            final_gate = self.gate_challenger(top_id, top_model)
            after = int(self.champion_info()["id"])

            specialist_ids = set(summary.specialist_generations.values())
            for plan in plans:
                gid = plan.generation
                if gid == after:
                    self.memory.update_population_member(round_id, gid, status="champion")
                    continue
                if gid in specialist_ids:
                    self.memory.update_generation(gid, status="specialist")
                    self.memory.update_population_member(round_id, gid, status="specialist")
                elif gid != top_id:
                    self.memory.update_generation(gid, status="rejected")
                    self.memory.update_population_member(round_id, gid, status="rejected")
                else:
                    self.memory.update_population_member(round_id, gid, status="rejected")

            self.memory.finish_population_round(round_id, "complete")
            for gid, stats in stats_by_generation.items():
                self.memory.add_metric(gid, "training_loss", stats.mean_loss, {"round_id": round_id})
            top_row = next((r for r in summary.ranking if int(r["generation"]) == top_id), summary.ranking[0])
            self.memory.add_metric(after, "population_top_score", float(top_row["score"]), {"round_id": round_id, "top_generation": top_id})
            return {
                "champion_before": before, "champion_after": after,
                "population_round": round_id, "selfplay_games": len(game_ids),
                "candidates": [asdict(p) for p in plans],
                "league": {
                    "ranking": summary.ranking,
                    "top_generation": summary.top_generation,
                    "specialists": summary.specialist_generations,
                },
                "specialist_replay_games": harvested,
                "final_gate": asdict(final_gate),
                "trained": True,
            }
        except Exception:
            self.memory.finish_population_round(round_id, "failed")
            raise

    def _evolve_cycle_unlocked(self) -> dict[str, Any]:
        before = int(self.champion_info()["id"])
        game_ids = self.selfplay()
        ReflectionEngine(self.memory).reflect_recent(before, limit=50)
        trained = self.train_challenger()
        if trained is None:
            return {
                "champion_before": before,
                "champion_after": before,
                "selfplay_games": len(game_ids),
                "trained": False,
                "reason": "not enough replay examples",
            }
        challenger_id, challenger, stats, _ = trained
        arena = self.gate_challenger(challenger_id, challenger)
        after = int(self.champion_info()["id"])
        self.memory.add_metric(after, "training_loss", stats.mean_loss, {"challenger": challenger_id})
        self.memory.add_metric(after, "arena_score", arena.score, {"challenger": challenger_id, "champion_before": before})
        return {
            "champion_before": before,
            "champion_after": after,
            "challenger": challenger_id,
            "selfplay_games": len(game_ids),
            "training": asdict(stats),
            "arena": asdict(arena),
            "trained": True,
        }

    def evolve_cycle(self) -> dict[str, Any]:
        with EvolutionLock(self.paths["root"]):
            self._retire_stale_challengers()
            if bool(self.config.get("league", {}).get("enabled", True)):
                return self._population_evolve_cycle_unlocked()
            return self._evolve_cycle_unlocked()

    def evolve(self, *, hours: float | None = None, cycles: int | None = None, progress=print) -> list[dict[str, Any]]:
        if hours is None and cycles is None:
            cycles = 1
        deadline = None if hours is None else monotonic() + max(0.0, hours) * 3600.0
        completed: list[dict[str, Any]] = []
        i = 0
        with EvolutionLock(self.paths["root"]):
            self._retire_stale_challengers()
            while True:
                if cycles is not None and i >= cycles:
                    break
                if deadline is not None and monotonic() >= deadline:
                    break
                i += 1
                if bool(self.config.get("league", {}).get("enabled", True)):
                    progress(f"[cycle {i}] self-play → shared train → population league → final gate")
                    result = self._population_evolve_cycle_unlocked()
                else:
                    progress(f"[cycle {i}] self-play → replay → train → arena")
                    result = self._evolve_cycle_unlocked()
                completed.append(result)
                progress(json.dumps(result, ensure_ascii=False, indent=2))
                if deadline is not None and monotonic() >= deadline:
                    break
        return completed

    def analyze(self, fen: str | None = None, *, depth: int | None = None, top_n: int = 5) -> SearchResult:
        board = chess.Board(fen) if fen else chess.Board()
        return self.make_searcher().search(board, depth=depth, top_n=top_n)

    def teacher_distill(self, positions: int = 64) -> dict[str, Any]:
        status = find_stockfish(self.config)
        if not status.available or not status.path:
            raise RuntimeError(
                "Stockfish was not found. Install it or set teacher.stockfish_path in a config override."
            )
        rows = self.memory.replay_sample(max(1, int(positions)), recent_fraction=0.5)
        fens = list(dict.fromkeys(str(r["fen"]) for r in rows))[:positions]
        if not fens:
            raise RuntimeError("No replay positions exist yet. Generate self-play first.")
        teacher = StockfishTeacher(status.path, depth=int(self.config.get("teacher", {}).get("depth", 14)))
        examples = teacher.annotate(fens)
        generation = int(self.champion_info()["id"])
        gid = self.memory.add_game(
            source="teacher",
            generation=generation,
            white_agent="StockfishTeacher",
            black_agent=f"dog_matist-g{generation}",
            result="*",
            termination="teacher_distillation",
            pgn="",
            plies=len(examples),
            examples=examples,
            metadata={"stockfish_path": status.path, "depth": int(self.config.get("teacher", {}).get("depth", 14))},
        )
        self.memory.add_insight(
            generation, "teacher",
            f"Stockfish teacher 为 {len(examples)} 个长期记忆局面提供了策略/价值标签。",
            {"game_id": gid, "positions": len(examples)},
        )
        return {"game_id": gid, "positions_labeled": len(examples), "stockfish": status.path}

    def status(self) -> dict[str, Any]:
        snapshot = self.memory.status_snapshot()
        champion = snapshot["champion"] or {}
        teacher = find_stockfish(self.config)
        genome = self.generation_genome(int(champion["id"])) if champion.get("id") is not None else None
        recent_round = self.memory.recent_population_round()
        specialists = [dict(r) for r in self.memory.active_specialists(limit=8)]
        population = None
        if recent_round is not None:
            population = {
                "round": dict(recent_round),
                "members": [dict(r) for r in self.memory.population_members(int(recent_round["id"]))],
            }
        return {
            "project": self.config["project"]["name"],
            "mode": self.mode,
            "device": str(self.device),
            "training_device": str(self.device),
            "search_device": str(self.search_device),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "state_dir": str(self.paths["root"]),
            "champion_generation": champion.get("id"),
            "champion_checkpoint": champion.get("checkpoint_path"),
            "champion_genome": genome.to_dict() if genome else None,
            "games": snapshot["games"],
            "replay_examples": snapshot["examples"],
            "results": snapshot["results"],
            "recent_insights": [x["text"] for x in snapshot["insights"]],
            "population": population,
            "specialists": specialists,
            "resource_budget": self._last_resource_budget or self.memory.get_meta("last_resource_budget"),
            "stockfish_teacher": {"available": teacher.available, "path": teacher.path, "reason": teacher.reason},
        }

    def doctor(self) -> dict[str, Any]:
        return {
            "torch_version": torch.__version__,
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "cuda_available": bool(torch.cuda.is_available()),
            "selected_training_device": str(self.device),
            "selected_search_device": str(self.search_device),
            "database": str(self.paths["db"]),
            "database_writable": os.access(self.paths["root"], os.W_OK),
            "champion_exists": Path(self.champion_info()["checkpoint_path"]).exists(),
            "mode": self.mode,
        }
