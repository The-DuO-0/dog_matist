from darwinchess.memory import MemoryStore, ReplayExample


def test_memory_persists_game_and_examples(tmp_path):
    db = tmp_path / "memory.sqlite3"
    with MemoryStore(db) as mem:
        gid = mem.add_game(
            source="test",
            generation=0,
            white_agent="a",
            black_agent="b",
            result="1-0",
            termination="test",
            pgn="[Result \"1-0\"]\n\n1. e4 1-0",
            plies=1,
            examples=[ReplayExample("fen", "e2e4", 1.0)],
        )
        assert mem.count_games() == 1
        assert mem.count_examples() == 1
        assert mem.replay_sample(1)[0]["game_id"] == gid
    with MemoryStore(db) as mem:
        assert mem.count_games() == 1


def test_generation_promotion_is_atomic_in_memory(tmp_path):
    db = tmp_path / "promotion.sqlite3"
    with MemoryStore(db) as mem:
        mem.add_generation(0, None, "/tmp/g0.pt", "champion")
        mem.add_generation(1, 0, "/tmp/g1_challenger.pt", "challenger")
        mem.promote_generation(0, 1, "/tmp/g1_champion.pt")
        assert mem.get_generation(0)["status"] == "retired"
        assert mem.champion_generation()["id"] == 1
        assert mem.champion_generation()["checkpoint_path"] == "/tmp/g1_champion.pt"
