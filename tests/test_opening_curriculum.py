import chess

from darwinchess.opening_curriculum import CURATED_OPENINGS, OpeningCurriculum, validate_curriculum
from darwinchess.selfplay import _sample_selfplay_opening


def test_curated_openings_are_legal_and_distinct():
    assert validate_curriculum() == []
    assert len(CURATED_OPENINGS) >= 16


def test_arena_pairs_are_usable_positions():
    curriculum = OpeningCurriculum(seed=7)
    pairs = curriculum.arena_pairs(12)
    assert len(pairs) == 12
    assert len({name for _, name in pairs}) >= 10
    for board, _name in pairs:
        assert isinstance(board, chess.Board)
        assert not board.is_game_over()


def test_sampling_has_multiple_families():
    curriculum = OpeningCurriculum(seed=22)
    families = {curriculum.sample()[2] for _ in range(120)}
    assert "standard" in families
    assert "controlled_random" in families
    assert len(families) >= 4


def test_selfplay_uses_configurable_curriculum():
    config = {
        "selfplay": {
            "opening_curriculum": {
                "enabled": True,
                "standard": 0.0,
                "curated": 1.0,
                "uncommon": 0.0,
                "controlled_random": 0.0,
            }
        }
    }
    starts = [_sample_selfplay_opening(config, seed=i) for i in range(20)]
    assert all(family != "standard" for _board, _name, family in starts)
    assert len({name for _board, name, _family in starts}) >= 5
