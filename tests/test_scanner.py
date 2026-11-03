import os
import pytest
from src.scanner import (
    norm_team,
    get_elo,
    strip_accents,
    find_matched_odds,
    format_date_for_tracking,
    append_bets_to_tracking,
    load_existing_tracking_keys
)

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

def test_format_date_for_tracking():
    assert format_date_for_tracking("2026-11-07T15:00:00.000Z") == "07/11/26"
    assert format_date_for_tracking("2026-11-07") == "07/11/26"
    assert format_date_for_tracking("07/11/26") == "07/11/26"

def test_append_bets_to_tracking_and_idempotency(tmp_path):
    test_csv = tmp_path / "tracking_test.csv"
    test_csv.write_text(
        "Date,League,Match,Bet,Bookmaker_odds,Model_probability,Fair_odds,EV_Perc,Stake_euro,Result,Profit,Bankroll\n"
        "02/11/26,E0,Newcastle-Everton,1,2.29,50.9%,1.96,16.65,15.17,2,-15.17,1050.47\n",
        encoding="utf-8"
    )

    sample_bets = [
        {
            "date": "2026-11-07T15:00:00.000Z",
            "league": "E0",
            "match": "Arsenal - Chelsea",
            "selection": "1",
            "bookmaker_odds": 2.10,
            "fair_odds": 1.91,
            "model_prob": 52.4,
            "ev_perc": 10.04,
            "stake_eur": 15.20,
            "applied_fraction": 1.45
        },
        # Existing duplicate that must be skipped
        {
            "date": "2026-11-02",
            "league": "E0",
            "match": "Newcastle-Everton",
            "selection": "1",
            "bookmaker_odds": 2.29,
            "fair_odds": 1.96,
            "model_prob": 50.9,
            "ev_perc": 16.65,
            "stake_eur": 15.17,
            "applied_fraction": 1.44
        }
    ]

    # First append: only Arsenal - Chelsea should be added (1 new)
    newly_added = append_bets_to_tracking(sample_bets, str(test_csv))
    assert len(newly_added) == 1
    assert newly_added[0]["match_clean"] == "Arsenal-Chelsea"
    assert newly_added[0]["date_csv"] == "07/11/26"

    # Second append: exact same list -> 0 should be added (idempotent!)
    again = append_bets_to_tracking(sample_bets, str(test_csv))
    assert len(again) == 0

    # Verify CSV file lines
    content = test_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 3 # Header + Newcastle + Arsenal
    assert "07/11/26,E0,Arsenal-Chelsea,1,2.10,52.4%,1.91,10.04,15.20,,," in content[2]

