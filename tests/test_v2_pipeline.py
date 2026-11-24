"""
tests/test_v2_pipeline.py
Unit tests for Deep-Pitch v2:
  - LiveV2StateEngine & exact motivation math
  - predict_v2 5-Head Ensemble probability sanity & calibration
  - injury_lineup fresh-missing starter detection, logit adjustment, and trap guard
  - decision_engine max_ev anti-adverse-selection guard
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.decision_engine import calculate_quarter_kelly
from src.injury_lineup import (
    build_match_injury_report,
    apply_injury_lineup_adjustment,
    check_injury_trap_guard,
)
from src.v2_state_engine import compute_live_motivation_features, canon_team, LiveV2StateEngine
from src.predict_v2 import predici_partita_v2


def test_decision_engine_max_ev_guard():
    # EV = 0.50 * 2.10 - 1 = +5.0% -> inside [1%, 8%) -> actionable
    ok = calculate_quarter_kelly(prob=0.50, bookmaker_odds=2.10, bankroll=1000.0, min_ev=0.01, max_ev=0.08, min_odds=1.35, max_odds=6.0)
    assert ok["actionable"] is True

    # EV = 0.55 * 2.10 - 1 = +15.5% -> above max_ev 8% -> blocked as toxic trap!
    trap = calculate_quarter_kelly(prob=0.55, bookmaker_odds=2.10, bankroll=1000.0, min_ev=0.01, max_ev=0.08, min_odds=1.35, max_odds=6.0)
    assert trap["actionable"] is False
    assert "anti-trappola" in trap["reason"]


def test_injury_lineup_fresh_vs_longterm():
    # Build 4 completed match JSONs where StarA (id=101) and StarB (id=102) started every match
    recent_home = []
    for _ in range(4):
        recent_home.append({
            "content": {
                "lineup": {
                    "homeTeam": {
                        "id": 1, "name": "Arsenal",
                        "starters": [
                            {"id": 101, "name": "Bukayo Saka", "performance": {"rating": 7.8}},
                            {"id": 102, "name": "William Saliba", "performance": {"rating": 7.5}},
                            {"id": 103, "name": "Declan Rice", "performance": {"rating": 7.4}},
                        ],
                    },
                    "awayTeam": {"id": 2, "name": "Opponent", "starters": []},
                }
            }
        })

    # Upcoming match: Saka & Saliba are FRESHLY injured, BenchGuy (id=999) was long-term injured (never started!)
    upcoming = {
        "content": {
            "lineup": {
                "lineupType": "lastStarting11",
                "homeTeam": {
                    "id": 1, "name": "Arsenal", "totalStarterMarketValue": 600_000_000,
                    "starters": [{"id": 103, "name": "Declan Rice", "marketValue": 100_000_000}],
                    "unavailable": [
                        {"id": 101, "name": "Bukayo Saka", "marketValue": 120_000_000, "unavailability": {"type": "injury", "expectedReturn": "Mid December"}},
                        {"id": 102, "name": "William Saliba", "marketValue": 80_000_000, "unavailability": {"type": "suspension", "expectedReturn": ""}},
                        {"id": 999, "name": "LongTerm Bench", "marketValue": 40_000_000, "unavailability": {"type": "injury", "expectedReturn": "March"}},
                    ],
                },
                "awayTeam": {
                    "id": 9, "name": "Chelsea", "totalStarterMarketValue": 500_000_000,
                    "starters": [],
                    "unavailable": [],
                },
            }
        }
    }

    rep = build_match_injury_report(upcoming, "Arsenal", "Chelsea", recent_home, [])
    # LongTerm Bench must NOT be counted as fresh! Only Saka + Saliba = 2.0 fresh missing!
    assert rep["home"]["fresh_missing_count"] == 2.0
    assert rep["away"]["fresh_missing_count"] == 0.0
    assert rep["net_logit_shift_home"] < -0.10

    # Probabilities must shift away from Home
    p_orig = [0.55, 0.25, 0.20]
    p_adj = apply_injury_lineup_adjustment(p_orig, rep)
    assert abs(sum(p_adj) - 1.0) < 1e-6
    assert p_adj[0] < p_orig[0]
    assert p_adj[2] > p_orig[2]

    # Trap guard must block betting on Home ("1"), but NOT block Away ("2")
    blk_h, reason_h = check_injury_trap_guard("1", rep)
    blk_a, _ = check_injury_trap_guard("2", rep)
    assert blk_h is True
    assert "Saka" in reason_h
    assert blk_a is False


def test_exact_motivation_champion_vs_relegation():
    # Simulated MD37 table (1 game remaining): Team1 already champion by 6 pts, Team18 fighting relegation!
    rows = []
    for i in range(1, 21):
        pts = 92 if i == 1 else (84 if i == 2 else max(20, 75 - i * 3))
        if i == 17:
            pts = 34
        if i == 18:
            pts = 33
        rows.append({"idx": i, "name": f"Team{i}", "pts": pts, "played": 37})
    mot = compute_live_motivation_features(rows, "Team1", "Team18", total_games=38, rel_pos_1idx=18)
    # Home is champion (-1), Away is desperate for relegation -> ex_champ_diff = -1.0, ex_mot_diff < 0
    assert mot["ex_champ_diff"] == -1.0
    assert mot["ex_mot_diff"] < -0.5


def test_v2_end_to_end_inference():
    assert canon_team("Manchester City") == "mancity"
    assert canon_team("Paris Saint-Germain") == "parissg"

    eng = LiveV2StateEngine()
    payload = eng.build_match_payload(
        div="E0", home_team="Arsenal", away_team="Chelsea",
        home_elo=1950.0, away_elo=1820.0,
        rest_days_h=6.0, rest_days_a=4.0,
        matches_14d_h=2.0, matches_14d_a=3.0,
        home_pts=28.0, away_pts=21.0, matchday=12.0,
        standings_rows=[], total_games=38, rel_pos_1idx=18,
    )
    out = predici_partita_v2(payload, market_odds=[1.85, 3.60, 4.20])
    probs = out["Final_1X2"]["Probs_Float"]
    assert len(probs) == 3
    assert abs(sum(probs) - 1.0) < 1e-6
    assert probs[0] > probs[2]
    assert "Market_Reference" in out
