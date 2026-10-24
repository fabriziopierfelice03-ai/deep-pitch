import pytest
from src.scanner import norm_team, get_elo, strip_accents, find_matched_odds

def test_norm_team_and_elo():
    assert norm_team("Arsenal") == "arsenal"
    assert norm_team("Atlético de Madrid") == "atleticodemadrid"
    assert get_elo("Arsenal") > 1900.0
    assert get_elo("Real Madrid") > 1900.0
    assert get_elo("Inter") > 1850.0

def test_strip_accents():
    assert strip_accents("Málaga") == "Malaga"
    assert strip_accents("Deportivo Alavés") == "Deportivo Alaves"
    assert strip_accents("1. FC Köln") == "1. FC Koln"

def test_find_matched_odds():
    odds_map = {
        "arsenal_chelsea": (1.80, 3.60, 4.20),
        "barcelona_realmadrid": (1.61, 5.48, 3.87)
    }
    match = find_matched_odds("Arsenal", "Chelsea", odds_map)
    assert match == (1.80, 3.60, 4.20)
    match_barca = find_matched_odds("FC Barcelona", "Real Madrid", odds_map)
    assert match_barca == (1.61, 5.48, 3.87)
    assert find_matched_odds("Unknown", "Team", odds_map) is None
