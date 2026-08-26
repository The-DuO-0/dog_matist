from studio.backend import find_state_value, flatten_mapping, normalize_move


def test_flatten_mapping():
    assert flatten_mapping({"champion": {"generation": 7}})["champion.generation"] == 7


def test_find_state_value():
    s = {"champion": {"generation": 12}, "replay_size": 44}
    assert find_state_value(s, "champion_generation", "generation") == 12
    assert find_state_value(s, "replay_size") == 44


def test_normalize_move():
    assert normalize_move({"best_move": "e2e4"}) == "e2e4"
    assert normalize_move("bestmove g1f3 ponder g8f6") == "g1f3"
