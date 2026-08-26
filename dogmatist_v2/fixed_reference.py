from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from .live_parallel_league import LiveLeagueProcessPool, LiveLeagueWorkerTask
from .runtime import ColorPairing
from .strength_lab import RoundStrengthEvidence


@dataclass(frozen=True)
class FrozenStrengthReference:
    """Immutable checkpoint used as a long-lived strength ruler.

    The reference is intentionally independent from the current Champion. If
    Gen15 remains Champion for many rounds, later candidates can still move from
    48% -> 53% -> 57% against this exact frozen checkpoint and the Strength Lab can
    see real progress instead of mistaking a stable throne for a stagnant system.
    """

    reference_id: str
    generation: int
    checkpoint_path: str
    checkpoint_sha256: str
    created_at: str

    def as_dict(self) -> dict[str, object]:
        return {
            "reference_id": self.reference_id,
            "generation": self.generation,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class FixedReferenceResult:
    round_index: int
    subject_generation: int
    reference_id: str
    score: float
    games: int
    wins: int
    draws: int
    losses: int
    complete_colour_pairs: int
    drained: bool
    stop_reason: str | None

    def to_round_evidence(self, *, champion_generation: int, promoted: bool) -> RoundStrengthEvidence:
        return RoundStrengthEvidence(
            round_index=self.round_index,
            champion_generation=int(champion_generation),
            promoted=bool(promoted),
            fixed_reference_score=float(self.score),
            paired_games=int(self.games),
        )

    def ui_payload(self) -> dict[str, object]:
        return {
            "round_index": self.round_index,
            "subject_generation": self.subject_generation,
            "reference_id": self.reference_id,
            "score": self.score,
            "games": self.games,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "complete_colour_pairs": self.complete_colour_pairs,
            "drained": self.drained,
            "stop_reason": self.stop_reason,
        }


def checkpoint_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class FrozenReferenceManager:
    """Create/verify one copied checkpoint and a tiny JSON manifest.

    No live checkpoint is edited. Freezing copies bytes into a dedicated directory
    and hashes them. Existing manifests are verified before reuse, so an accidental
    overwrite cannot silently move the ruler.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "reference.json"

    def load(self) -> FrozenStrengthReference | None:
        if not self.manifest_path.is_file():
            return None
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return FrozenStrengthReference(
            reference_id=str(payload["reference_id"]),
            generation=int(payload["generation"]),
            checkpoint_path=str(payload["checkpoint_path"]),
            checkpoint_sha256=str(payload["checkpoint_sha256"]),
            created_at=str(payload["created_at"]),
        )

    def verify(self, reference: FrozenStrengthReference) -> bool:
        path = Path(reference.checkpoint_path)
        return path.is_file() and checkpoint_sha256(path) == reference.checkpoint_sha256

    def freeze(
        self,
        source_checkpoint: str | Path,
        *,
        generation: int,
        created_at: datetime,
        reference_id: str | None = None,
    ) -> FrozenStrengthReference:
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        source = Path(source_checkpoint).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)

        existing = self.load()
        if existing is not None:
            if not self.verify(existing):
                raise RuntimeError("frozen strength reference failed checksum verification")
            return existing

        ref_id = reference_id or f"g{int(generation)}-{created_at:%Y%m%dT%H%M%S%z}"
        suffix = source.suffix or ".pt"
        target = self.root / f"{ref_id}{suffix}"
        if target.exists():
            raise FileExistsError(target)
        shutil.copy2(source, target)
        reference = FrozenStrengthReference(
            reference_id=ref_id,
            generation=int(generation),
            checkpoint_path=str(target.resolve()),
            checkpoint_sha256=checkpoint_sha256(target),
            created_at=created_at.isoformat(),
        )
        temp = self.manifest_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(reference.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.manifest_path)
        return reference


class FixedReferenceEvaluator:
    """Paired-colour evaluation against one immutable checkpoint.

    This reuses the real process-backed League worker. The session clock is only
    an admission gate: if time expires while a reference game is active, that game
    and its reverse-colour mate finish. Only the independent bug watchdog may kill
    a worker.

    The frozen reference receives a negative synthetic participant id inside the
    worker tasks. This matters when the live subject is the same generation that
    originally became the reference (for example Gen15 vs frozen Gen15): identical
    generation numbers must not make both sides accidentally load the live file.
    """

    def __init__(
        self,
        *,
        clock: Any,
        parallel_games: int = 2,
        hard_game_timeout_seconds: float = 2 * 60 * 60,
        stall_timeout_seconds: float = 30 * 60,
        kill_grace_seconds: float = 2.0,
        status_callback: Any | None = None,
        mp_context: Any | None = None,
        worker_target: Any | None = None,
    ) -> None:
        self.clock = clock
        self.parallel_games = int(parallel_games)
        self.hard_game_timeout_seconds = float(hard_game_timeout_seconds)
        self.stall_timeout_seconds = float(stall_timeout_seconds)
        self.kill_grace_seconds = float(kill_grace_seconds)
        self.status_callback = status_callback
        self.mp_context = mp_context
        self.worker_target = worker_target

    def evaluate(
        self,
        *,
        round_index: int,
        subject_generation: int,
        subject_checkpoint: str,
        reference: FrozenStrengthReference,
        config: dict[str, Any],
        openings: Iterable[tuple[str, str]],
        depth: int,
        max_plies: int,
        torch_threads: int = 1,
    ) -> FixedReferenceResult:
        if not Path(subject_checkpoint).is_file():
            raise FileNotFoundError(subject_checkpoint)
        if not Path(reference.checkpoint_path).is_file():
            raise FileNotFoundError(reference.checkpoint_path)

        # Production generations are positive. Keep the immutable ruler as a
        # synthetic negative participant so Gen15-live and Gen15-frozen remain two
        # distinct players even though they share historical ancestry.
        reference_player = -abs(int(reference.generation)) - 1
        if reference_player == int(subject_generation):
            reference_player -= 1

        pairings: list[ColorPairing] = []
        tasks: dict[str, LiveLeagueWorkerTask] = {}
        for index, (fen, opening_name) in enumerate(openings):
            pair_id = f"ref-r{int(round_index)}-{index}"
            pair = ColorPairing(
                pair_id,
                str(int(subject_generation)),
                str(reference_player),
                opening_name,
            )
            pairings.append(pair)
            for spec in pair.games():
                subject_is_white = int(spec.white_id) == int(subject_generation)
                tasks[spec.game_id] = LiveLeagueWorkerTask(
                    game_id=spec.game_id,
                    pairing_id=pair_id,
                    leg=int(spec.leg),
                    round_id=int(round_index),
                    white_generation=int(spec.white_id),
                    black_generation=int(spec.black_id),
                    white_checkpoint=subject_checkpoint if subject_is_white else reference.checkpoint_path,
                    black_checkpoint=reference.checkpoint_path if subject_is_white else subject_checkpoint,
                    config=config,
                    start_fen=fen,
                    opening_name=opening_name,
                    depth=int(depth),
                    max_plies=int(max_plies),
                    seed=int(round_index) * 1_000_003 + index * 2 + int(spec.leg),
                    torch_threads=max(1, int(torch_threads)),
                )

        if not pairings:
            return FixedReferenceResult(
                int(round_index), int(subject_generation), reference.reference_id,
                0.5, 0, 0, 0, 0, 0, False, None,
            )

        kwargs: dict[str, Any] = {
            "clock": self.clock,
            "parallel_games": self.parallel_games,
            "hard_game_timeout_seconds": self.hard_game_timeout_seconds,
            "stall_timeout_seconds": self.stall_timeout_seconds,
            "kill_grace_seconds": self.kill_grace_seconds,
            "status_callback": self.status_callback,
        }
        if self.mp_context is not None:
            kwargs["mp_context"] = self.mp_context
        if self.worker_target is not None:
            kwargs["worker_target"] = self.worker_target
        execution = LiveLeagueProcessPool(pairings, tasks, **kwargs).run()
        rows = execution.result_by_game

        wins = draws = losses = complete_pairs = 0
        for pair in pairings:
            specs = pair.games()
            legs = [rows.get(spec.game_id) for spec in specs]
            if any(row is None for row in legs):
                continue
            complete_pairs += 1
            for row in legs:
                assert row is not None
                if row.white_generation == int(subject_generation):
                    subject_score = row.white_score
                else:
                    subject_score = 1.0 - row.white_score
                if subject_score > 0.75:
                    wins += 1
                elif subject_score < 0.25:
                    losses += 1
                else:
                    draws += 1

        games = wins + draws + losses
        score = (wins + 0.5 * draws) / games if games else 0.5
        return FixedReferenceResult(
            round_index=int(round_index),
            subject_generation=int(subject_generation),
            reference_id=reference.reference_id,
            score=score,
            games=games,
            wins=wins,
            draws=draws,
            losses=losses,
            complete_colour_pairs=complete_pairs,
            drained=bool(execution.draining),
            stop_reason=execution.stop_reason,
        )
