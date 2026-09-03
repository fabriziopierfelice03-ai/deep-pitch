"""
DeepPitch: Quick Inference Test Script
Verifies that weights, embedding layers, and Poisson modules load and predict correctly.
"""

from src.predict import predici_partita


def run_verification():
    print("=" * 60)
    print(" DeepPitch: Testing Model Inference Pipeline")
    print("=" * 60)

    target_league = "I1"

    sample_vector = [
        1845.0, 1720.0,
        2.0, 7.0, 48.0, 36.0,
        2.1, 1.3, 2.4, 1.0,
        1.65, 1.12, 1.80, 0.95, 0.90, 1.45, 0.85, 1.60,
        1.80, 1.20, 2.20, 1.00, 0.80, 1.40, 0.60, 1.60,
        0.15, -0.08,
        1.8, 1.0, 1.6, 1.0, 1.55, 1.10, 2.0, 1.8,
        6.0, 3.0, 2.0, 4.0, 1.0,
        12.0, -2.0, 18.0, 4.0, 32.0, 18.0,
        22.0, 2.0,
        14.2, 11.8, 5.4, 3.8, 6.2, 4.5, 1.8, 2.4,
    ]

    print(f"[*] Input vector dimension: {len(sample_vector)} features")
    print(f"[*] Target division: {target_league}")
    print("[*] Executing forward pass...\n")

    try:
        results = predici_partita(target_league, sample_vector)

        print("[+] SUCCESS: Pipeline executed without errors!\n")
        print("--- INFERENCE REPORT ---")
        print(f"Predicted xG: {results.get('xG_Predetti')}")

        if "Rete_Softmax_1X2" in results:
            print("\n1X2 Head (Neural Logits):")
            print(
                f"  Probabilities: {results['Rete_Softmax_1X2'].get('Probabilita_%')}"
            )
            print(
                f"  Fair Odds:     {results['Rete_Softmax_1X2'].get('Fair_Odds')}"
            )

        if "Poisson_Stats" in results:
            print("\nDerivative Markets (Poisson Distribution):")
            ou = results["Poisson_Stats"].get("Under_Over_2.5", {})
            btts = results["Poisson_Stats"].get("Gol_NoGol", {})
            print(
                f"  Over 2.5: Prob {ou.get('Prob_Over_%')}% | Fair {ou.get('Quota_Over')}"
            )
            print(
                f"  BTTS:     Prob {btts.get('Prob_Gol_%')}% | Fair {btts.get('Quota_Gol')}"
            )

        print("\n" + "=" * 60)

    except Exception as err:
        print(f"[-] Test failed: {err}")


if __name__ == "__main__":
    run_verification()
        print(f"[-] Test failed: {err}")


if __name__ == "__main__":
    run_verification()
