from src.decision_engine import calculate_quarter_kelly, scan_match_markets


def test_quarter_kelly_default_cap_3_percent():
    # EV = +17.07%, raw Quarter-Kelly = 3.05% -> must cap at 3.0%
    res = calculate_quarter_kelly(prob=0.4878, bookmaker_odds=2.40, bankroll=1000.0)
    assert res["actionable"] is True
    assert res["applied_fraction_perc"] == 3.0
    assert res["stake_eur"] == 30.0


def test_quarter_kelly_uncapped_below_3_percent():
    # EV = +10.0%, b = 1.20 -> f* = 8.33%, quarter = 2.083% (< 3.0%)
    res = calculate_quarter_kelly(prob=0.50, bookmaker_odds=2.20, bankroll=1000.0)
    assert res["actionable"] is True
    assert res["applied_fraction_perc"] == 2.08
    assert res["stake_eur"] == 20.83


def test_scan_match_markets_uses_3_percent_cap():
    model_output = {
        "Bayesian_Shrinkage_1X2": {
            "Probabilita_%": {"1": 48.78, "X": 25.0, "2": 26.22}
        }
    }
    market_odds = {"1": 2.40, "X": 3.40, "2": 3.00}
    opps = scan_match_markets(model_output, market_odds, bankroll=2000.0)
    assert len(opps) == 1
    assert opps[0]["applied_fraction_perc"] == 3.0
    assert opps[0]["stake_eur"] == 60.0
