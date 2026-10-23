import unittest
import datetime
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.prematch_scanner import run_prematch_scan, get_active_bankroll, FIXTURES_WEEKEND

class TestPrematchScanner(unittest.TestCase):
    def test_fixtures_count(self):
        self.assertEqual(len(FIXTURES_WEEKEND), 48)
        leagues = [f["league"] for f in FIXTURES_WEEKEND]
        self.assertEqual(leagues.count("E0"), 10)
        self.assertEqual(leagues.count("I1"), 10)
        self.assertEqual(leagues.count("SP1"), 10)
        self.assertEqual(leagues.count("D1"), 9)
        self.assertEqual(leagues.count("F1"), 9)

    def test_tuesday_execution(self):
        tuesday_date = datetime.date(2026, 10, 20)
        res = run_prematch_scan(current_date=tuesday_date)
        self.assertEqual(res["status"], "NO_MIDWEEK_FIXTURES")
        self.assertEqual(len(res["qualified_bets"]), 0)

    def test_friday_execution(self):
        friday_date = datetime.date(2026, 10, 23)
        res = run_prematch_scan(current_date=friday_date)
        self.assertEqual(res["status"], "COMPLETED")
        self.assertEqual(res["total_matches"], 48)
        self.assertEqual(len(res["qualified_bets"]), 10)
        
        bankroll = get_active_bankroll()
        self.assertGreater(bankroll, 0.0)
        for bet in res["qualified_bets"]:
            self.assertGreaterEqual(bet["ev_perc"], 5.0)
            self.assertGreater(bet["bookmaker_odds"], 1.60)
            self.assertLess(bet["bookmaker_odds"], 3.10)
            self.assertLessEqual(bet["stake_eur"], round(bankroll * 0.015, 2) + 0.01)

if __name__ == "__main__":
    unittest.main()
