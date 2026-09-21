import sys
import unittest
sys.path.insert(0, r"c:\Users\fabri\Desktop\deep-pitch")

from src.feature_builder import (
    MatchRecord, H2HRecord, StandingsContext,
    build_56_features, build_57_features, validate_feature_scaling
)
from src.predict import predici_partita

class TestFeatureBuilder(unittest.TestCase):
    def test_lazio_milan_feature_vector(self):
        # 1. Lazio Match Records (Exact FotMob Data)
        lazio_matches = [
            MatchRecord(
                date="07/09/2026", opponent="Udinese", venue="A",
                goals_for=2, goals_against=1, xg_for=1.89, xg_against=0.76,
                shots=13, shots_on_target=5, corners=5, cards=3, points=3,
                source="FotMob/Opta"
            ),
            MatchRecord(
                date="30/08/2026", opponent="Genoa", venue="H",
                goals_for=1, goals_against=0, xg_for=1.62, xg_against=0.42,
                shots=14, shots_on_target=5, corners=6, cards=2, points=3,
                source="FotMob/Opta"
            ),
            MatchRecord(
                date="24/08/2026", opponent="Bologna", venue="A",
                goals_for=1, goals_against=0, xg_for=0.63, xg_against=1.78,
                shots=11, shots_on_target=4, corners=4, cards=2, points=3,
                source="FotMob/Opta"
            ),
        ]

        # 2. AC Milan Match Records (Exact FotMob Data)
        milan_matches = [
            MatchRecord(
                date="06/09/2026", opponent="Juventus", venue="A",
                goals_for=1, goals_against=1, xg_for=0.09, xg_against=1.25,
                shots=12, shots_on_target=4, corners=4, cards=3, points=1,
                source="FotMob/Opta"
            ),
            MatchRecord(
                date="28/08/2026", opponent="Venezia", venue="H",
                goals_for=2, goals_against=0, xg_for=1.60, xg_against=0.75,
                shots=17, shots_on_target=7, corners=7, cards=1, points=3,
                source="FotMob/Opta"
            ),
            MatchRecord(
                date="23/08/2026", opponent="Torino", venue="A",
                goals_for=2, goals_against=1, xg_for=2.08, xg_against=1.15,
                shots=15, shots_on_target=6, corners=6, cards=2, points=3,
                source="FotMob/Opta"
            ),
        ]

        # 3. H2H Records (FBref official last 5 meetings)
        h2h = H2HRecord(
            home_avg_points=2.0,
            away_avg_points=0.8,
            home_avg_goals_for=1.2,
            home_avg_goals_against=0.8,
            home_avg_xg=1.35,
            away_avg_xg=1.25,
            venue_home_avg_points=2.3333,
            venue_home_avg_goals_for=1.3333,
            source="FBref"
        )

        # 4. Standings Context (Official Lega Serie A after MD3)
        standings = StandingsContext(
            home_pos=3.0,
            away_pos=5.0,
            home_points=9.0,
            away_points=7.0,
            home_dist_ucl=2.0,
            away_dist_ucl=0.0,
            home_dist_uel=2.0,
            away_dist_uel=0.0,
            home_dist_rel=9.0,
            away_dist_rel=7.0,
            matchday=4.0,
            season_half=1.0
        )

        # 5. Build 56-feature vector
        feat_56 = build_56_features(
            home_elo=1785.0,
            away_elo=1820.0,
            home_matches=lazio_matches,
            away_matches=milan_matches,
            h2h=h2h,
            standings=standings,
            home_rest_days=5.0,
            away_rest_days=6.0,
            home_matches_14d=2.0,
            away_matches_14d=2.0
        )

        self.assertEqual(len(feat_56), 56)
        self.assertFalse(any(v is None for v in feat_56))

        # 6. Test scaling
        scaling = validate_feature_scaling(feat_56)
        self.assertTrue(scaling["is_sane"], f"Scaling anomaly: min={scaling['min_scaled']}, max={scaling['max_scaled']}")
        self.assertLess(scaling["max_scaled"], 3.0)
        self.assertGreater(scaling["min_scaled"], -3.0)

        # 7. Test prediction call (Raw Softmax & Bayesian Shrinkage)
        market_odds = [2.70, 3.30, 2.60]
        res = predici_partita("I1", feat_56, market_odds=market_odds)
        self.assertIn("Rete_Softmax_1X2", res)
        self.assertIn("Bayesian_Shrinkage_1X2", res)
        
        # Verify Bayesian Shrinkage probabilities sum to 100%
        bayes_probs = res["Bayesian_Shrinkage_1X2"]["Probabilita_%"]
        total_p = bayes_probs["1"] + bayes_probs["X"] + bayes_probs["2"]
        self.assertAlmostEqual(total_p, 100.0, delta=0.5)

        print("\nTest passed successfully! Model output:")
        print("Softmax 1X2 Prob:", res["Rete_Softmax_1X2"]["Probabilita_%"])
        print("Bayesian Shrinkage Prob:", res["Bayesian_Shrinkage_1X2"]["Probabilita_%"])

if __name__ == "__main__":
    unittest.main()
