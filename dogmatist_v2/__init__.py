"""DogMatist v2: population/league/OpenTree evolution primitives."""

from .archive import (
    ArchiveEntry,
    ArchivePolicy,
    ArchiveTier,
    CompactCheckpointPlan,
    choose_archive_tier,
)
from .chronicle_store import ChronicleStore
from .dynasty import (
    ChampionReign,
    GenerationChronicle,
    GenerationLife,
    HistoricalEvent,
    HistoricalRole,
    build_lineage_path,
)
from .fixed_reference import (
    FixedReferenceEvaluator,
    FixedReferenceResult,
    FrozenReferenceManager,
    FrozenStrengthReference,
    checkpoint_sha256,
)
from .hard_positions import HardPositionCandidate, HardPositionMiner
from .league import Candidate, MatchResult, LeagueTable, select_survivors
from .live_arena_guard import (
    LiveArenaDrainOverride,
    LiveArenaDrainState,
    build_budget_aware_arena,
)
from .live_bridge import (
    AlphaBetaTeacherAdapter,
    CapturedHardPosition,
    LiveGameEvidenceBridge,
    TeacherReplayTarget,
    TeacherSearchBudget,
    cp_to_value,
)
from .live_compute import ComputeSnapshot, HeartbeatComputeClock
from .live_cycle_override import LiveStrengthCycleOverride
from .live_fixed_reference import (
    LiveFixedReferenceCoordinator,
    LiveFixedReferenceCycleOverride,
    LiveFixedReferenceReport,
)
from .live_game_watchdog import LiveGameWatchdogPolicy, install_live_game_watchdog_policy
from .live_league_guard import (
    DrainedArenaResult,
    LiveLeagueDrainOverride,
    LiveLeagueDrainState,
    build_budget_aware_population_arena,
)
from .live_parallel_league import (
    LiveLeagueProcessPool,
    LiveLeagueWorkerResult,
    LiveLeagueWorkerTask,
    LiveParallelLeagueExecution,
    choose_live_league_parallelism,
    league_worker_threads,
)
from .live_parallel_population import LiveParallelLeagueOverride, build_parallel_population_arena
from .live_replay import LiveReplayMixSampler, LiveReplayOverride, ReplayBatchQuota
from .live_runner import LiveEvolutionRunReport, LiveEvolutionRunner
from .live_runtime_overlay import LiveReplayExample, LiveStrengthCoordinator, LiveStrengthRoundReport
from .mac_preflight import (
    MacPreflightReport,
    PreflightCheck,
    audit_copied_state_after_run,
    load_snapshot_manifest,
    run_spawn_probe,
    validate_copied_state,
)
from .opening_lab import OpeningBucketSignal, OpeningRepairPlan, OpeningWeaknessController
from .opening_search_revision import (
    CandidateScore,
    OpeningDeepeningDecision,
    OpeningSearchEvidence,
    OpeningSearchR2Policy,
    OpeningSearchR2Session,
    OpeningSearchRevisionPlan,
    absolute_game_ply,
    candidate_scores,
    select_verification_candidates,
)
from .opening_stability import OpeningSearchObservation, OpeningSearchStabilityReport, build_stability_report
from .opentree_guard import GuardDecision, OpenTreeStrengthGuard, TrialEvidence
from .opentree_policy import (
    CurriculumMix,
    OpenTreeCurriculumController,
    OpenTreePolicy,
    TreeHealth,
)
from .opentree_promotion import (
    OpenTreePromotionCoordinator,
    PromotionDecision,
    PromotionEvidence,
)
from .opentree_report import (
    OpenTreeExperimentReport,
    OpenTreeExperimentSummary,
    OpenTreeRoundTrace,
)
from .opentree_trials import OpenTreePolicyTrialManager, PolicyTrial, TrialResult
from .promotion_bridge import ChampionCheckpoint, PromotionChronicleBridge
from .resource import ResourceBudget, ResourceController, ResourceSample
from .runtime import (
    ColorPairing,
    ComputeBudgetClock,
    GameState,
    GameWatchdog,
    LeagueGameSpec,
    LeagueGameStatus,
    LeaguePairScheduler,
    WatchdogTrip,
)
from .search_forensics import SearchForensicRow, SearchForensicsSummary, summarize_search_forensics
from .search_handoff import HandoffBranchEvidence, HandoffDiagnosis
from .specialist_bridge import (
    SpecialistCheckpoint,
    SpecialistChronicleBridge,
    parse_generation_id,
)
from .specialists import OpeningBucket, SpecialistArchive, SpecialistRecord
from .strength_bridge import PositionObservation, StrengthCapturePolicy, StrengthEvidenceBridge
from .strength_lab import (
    EngineGateAction,
    EngineGateDecision,
    EngineRevisionGate,
    EngineTrialEvidence,
    PlateauDetector,
    RoundStrengthEvidence,
    StrengthCurriculumMix,
    StrengthLabController,
    StrengthLabPlan,
    StrengthMode,
    TrainingBatchBudget,
)
from .strength_pipeline import (
    DeepSearchTeacherRequest,
    EngineABTrialPlan,
    StrengthPipelinePlanner,
    StrengthRoundRecipe,
)
from .strength_store import HardPositionEvidence, StrengthStore
from .ui_flow import (
    EvolutionFlowSnapshot,
    EvolutionStage,
    build_evolution_flow_snapshot,
    encode_ui_event,
)
from .validation_telemetry import ValidationTelemetry
from .worker_supervisor import KillableWorker, LeagueWorkerSupervisor, WorkerTermination

__all__ = [
    "ArchiveEntry",
    "ArchivePolicy",
    "ArchiveTier",
    "CompactCheckpointPlan",
    "choose_archive_tier",
    "ChronicleStore",
    "ChampionReign",
    "GenerationChronicle",
    "GenerationLife",
    "HistoricalEvent",
    "HistoricalRole",
    "build_lineage_path",
    "FixedReferenceEvaluator",
    "FixedReferenceResult",
    "FrozenReferenceManager",
    "FrozenStrengthReference",
    "checkpoint_sha256",
    "HardPositionCandidate",
    "HardPositionMiner",
    "Candidate",
    "MatchResult",
    "LeagueTable",
    "select_survivors",
    "LiveArenaDrainOverride",
    "LiveArenaDrainState",
    "build_budget_aware_arena",
    "AlphaBetaTeacherAdapter",
    "CapturedHardPosition",
    "LiveGameEvidenceBridge",
    "TeacherReplayTarget",
    "TeacherSearchBudget",
    "cp_to_value",
    "ComputeSnapshot",
    "HeartbeatComputeClock",
    "LiveStrengthCycleOverride",
    "LiveFixedReferenceCoordinator",
    "LiveFixedReferenceCycleOverride",
    "LiveFixedReferenceReport",
    "LiveGameWatchdogPolicy",
    "install_live_game_watchdog_policy",
    "DrainedArenaResult",
    "LiveLeagueDrainOverride",
    "LiveLeagueDrainState",
    "build_budget_aware_population_arena",
    "LiveLeagueProcessPool",
    "LiveLeagueWorkerResult",
    "LiveLeagueWorkerTask",
    "LiveParallelLeagueExecution",
    "choose_live_league_parallelism",
    "league_worker_threads",
    "LiveParallelLeagueOverride",
    "build_parallel_population_arena",
    "LiveReplayMixSampler",
    "LiveReplayOverride",
    "ReplayBatchQuota",
    "LiveEvolutionRunReport",
    "LiveEvolutionRunner",
    "LiveReplayExample",
    "LiveStrengthCoordinator",
    "LiveStrengthRoundReport",
    "MacPreflightReport",
    "PreflightCheck",
    "audit_copied_state_after_run",
    "load_snapshot_manifest",
    "run_spawn_probe",
    "validate_copied_state",
    "OpeningBucketSignal",
    "OpeningRepairPlan",
    "OpeningWeaknessController",
    "CandidateScore",
    "OpeningDeepeningDecision",
    "OpeningSearchEvidence",
    "OpeningSearchR2Policy",
    "OpeningSearchR2Session",
    "OpeningSearchRevisionPlan",
    "absolute_game_ply",
    "candidate_scores",
    "select_verification_candidates",
    "OpeningSearchObservation",
    "OpeningSearchStabilityReport",
    "build_stability_report",
    "GuardDecision",
    "OpenTreeStrengthGuard",
    "TrialEvidence",
    "CurriculumMix",
    "OpenTreeCurriculumController",
    "OpenTreePolicy",
    "TreeHealth",
    "OpenTreePromotionCoordinator",
    "PromotionDecision",
    "PromotionEvidence",
    "OpenTreeExperimentReport",
    "OpenTreeExperimentSummary",
    "OpenTreeRoundTrace",
    "OpenTreePolicyTrialManager",
    "PolicyTrial",
    "TrialResult",
    "ChampionCheckpoint",
    "PromotionChronicleBridge",
    "ResourceBudget",
    "ResourceController",
    "ResourceSample",
    "ColorPairing",
    "ComputeBudgetClock",
    "GameState",
    "GameWatchdog",
    "LeagueGameSpec",
    "LeagueGameStatus",
    "LeaguePairScheduler",
    "WatchdogTrip",
    "SearchForensicRow",
    "SearchForensicsSummary",
    "summarize_search_forensics",
    "HandoffBranchEvidence",
    "HandoffDiagnosis",
    "SpecialistCheckpoint",
    "SpecialistChronicleBridge",
    "parse_generation_id",
    "OpeningBucket",
    "SpecialistArchive",
    "SpecialistRecord",
    "PositionObservation",
    "StrengthCapturePolicy",
    "StrengthEvidenceBridge",
    "EngineGateAction",
    "EngineGateDecision",
    "EngineRevisionGate",
    "EngineTrialEvidence",
    "PlateauDetector",
    "RoundStrengthEvidence",
    "StrengthCurriculumMix",
    "StrengthLabController",
    "StrengthLabPlan",
    "StrengthMode",
    "TrainingBatchBudget",
    "DeepSearchTeacherRequest",
    "EngineABTrialPlan",
    "StrengthPipelinePlanner",
    "StrengthRoundRecipe",
    "HardPositionEvidence",
    "StrengthStore",
    "EvolutionFlowSnapshot",
    "EvolutionStage",
    "build_evolution_flow_snapshot",
    "encode_ui_event",
    "ValidationTelemetry",
    "KillableWorker",
    "LeagueWorkerSupervisor",
    "WorkerTermination",
]
