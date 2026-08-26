from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
import signal
from typing import Any

import torch

from .evaluator import HybridEvaluator
from .genome import AgentGenome
from .network import load_checkpoint
from .search import AlphaBetaSearcher
from .selfplay import GameRecord, play_game


@dataclass(frozen=True)
class SelfPlayTask:
    checkpoint_path: str
    generation: int
    config: dict[str, Any]
    seed: int
    depth: int


_WORKER_SEARCHER: AlphaBetaSearcher | None = None
_WORKER_CONFIG: dict[str, Any] | None = None
_WORKER_GENERATION: int | None = None


def _worker_init(checkpoint_path: str, generation: int, config: dict[str, Any]) -> None:
    """Load one champion once per worker and leave graceful SIGINT to the parent.

    Terminal Ctrl-C on macOS is delivered to the foreground process group. The
    V2 parent converts the first SIGINT into a safe-drain request, so child
    self-play workers must not independently raise KeyboardInterrupt and tear down
    the current batch. The parent remains able to terminate the executor/processes
    through normal lifecycle or a second emergency interrupt.
    """

    global _WORKER_SEARCHER, _WORKER_CONFIG, _WORKER_GENERATION
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (AttributeError, ValueError):
        pass
    torch.set_num_threads(1)
    device = torch.device("cpu")
    model, payload = load_checkpoint(checkpoint_path, device)
    genome = AgentGenome.from_dict(payload.get("metadata", {}).get("genome"), config)
    _WORKER_SEARCHER = AlphaBetaSearcher(HybridEvaluator(model, config, device, genome), config)
    _WORKER_CONFIG = config
    _WORKER_GENERATION = int(generation)


def _play_initialized(seed_depth: tuple[int, int]) -> GameRecord:
    if _WORKER_SEARCHER is None or _WORKER_CONFIG is None or _WORKER_GENERATION is None:
        raise RuntimeError("self-play worker was not initialized")
    seed, depth = seed_depth
    return play_game(
        _WORKER_SEARCHER, _WORKER_SEARCHER, _WORKER_CONFIG,
        white_name=f"dog_matist-g{_WORKER_GENERATION}",
        black_name=f"dog_matist-g{_WORKER_GENERATION}",
        stochastic=True,
        seed=int(seed),
        depth=int(depth),
    )


def _play_one(task: SelfPlayTask) -> GameRecord:
    _worker_init(task.checkpoint_path, task.generation, task.config)
    return _play_initialized((task.seed, task.depth))


def parallel_selfplay(tasks: list[SelfPlayTask], workers: int) -> list[GameRecord]:
    if workers <= 1 or len(tasks) <= 1:
        return [_play_one(t) for t in tasks]
    first = tasks[0]
    if any(t.checkpoint_path != first.checkpoint_path or t.generation != first.generation for t in tasks):
        raise ValueError("parallel self-play batch must pin one champion checkpoint")
    ctx = mp.get_context("spawn")
    payloads = [(t.seed, t.depth) for t in tasks]
    with ProcessPoolExecutor(
        max_workers=min(int(workers), len(tasks)),
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(first.checkpoint_path, first.generation, first.config),
    ) as pool:
        return list(pool.map(_play_initialized, payloads, chunksize=1))
