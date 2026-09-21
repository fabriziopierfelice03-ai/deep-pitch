"""
DeepPitch: Decision & Money Management Engine
Calculates Expected Value (EV), Quarter-Kelly Staking, and risk-management caps
aligned with the C++ backtest and Bayesian Shrinkage simulation engine.
"""

from typing import Dict, Any, List


def calculate_quarter_kelly(
    prob: float,
    bookmaker_odds: float,
    bankroll: float = 1000.0,
    min_ev: float = 0.055,
    min_odds: float = 1.60,
    max_odds: float = 3.10,
    max_cap: float = 0.015,
) -> Dict[str, Any]:
    """
    Calcola l'Expected Value e il dimensionamento dello stake con Quarter-Kelly.

    Parametri:
    - prob: probabilità stimata dal modello / calibrazione bayesiana (compresa tra 0.0 e 1.0)
    - bookmaker_odds: quota decimale offerta dal bookmaker
    - bankroll: capitale totale attuale (default 1000.0€)
    - min_ev: soglia minima di valore atteso (default 5.5% = 0.055)
    - min_odds: quota minima ammissibile (default 1.60)
    - max_odds: quota massima ammissibile (default 3.10)
    - max_cap: quota massima del bankroll allocabile su una singola scommessa (default 1.5% = 0.015)
    """
    if bookmaker_odds <= 1.0 or prob <= 0.0 or prob > 1.0:
        return {
            "actionable": False,
            "reason": "Quota o probabilita non valide",
            "ev_perc": 0.0,
            "stake_eur": 0.0,
        }

    # Filtro quota: esclusione quote schiacciate o longshot ad alta varianza
    if not (min_odds < bookmaker_odds < max_odds):
        return {
            "actionable": False,
            "reason": f"Quota ({bookmaker_odds}) fuori dal range ammissibile ({min_odds} - {max_odds})",
            "ev_perc": round(((prob * bookmaker_odds) - 1.0) * 100, 2),
            "stake_eur": 0.0,
        }

    # Calcolo Expected Value: (P * Quota) - 1
    ev = (prob * bookmaker_odds) - 1.0

    # Filtro valore atteso minimo
    if ev <= min_ev:
        return {
            "actionable": False,
            "reason": f"EV ({round(ev * 100, 2)}%) inferiore o pari alla soglia minima ({round(min_ev * 100, 2)}%)",
            "ev_perc": round(ev * 100, 2),
            "stake_eur": 0.0,
        }

    # Formula di Kelly: f* = EV / (Quota - 1)
    b = bookmaker_odds - 1.0
    kelly_star = ev / b

    # Quarter-Kelly: riduce drasticamente il drawdown e la varianza
    quarter_kelly = kelly_star * 0.25

    # Hard Cap di portafoglio: max 1.5% del bankroll per singola posizione
    capped_fraction = max(0.0, min(quarter_kelly, max_cap))
    stake_eur = round(bankroll * capped_fraction, 2)

    fair_odds = round(1.0 / prob, 2)

    return {
        "actionable": True,
        "prob_perc": round(prob * 100, 2),
        "fair_odds": fair_odds,
        "bookmaker_odds": bookmaker_odds,
        "ev_perc": round(ev * 100, 2),
        "kelly_raw_perc": round(quarter_kelly * 100, 2),
        "applied_fraction_perc": round(capped_fraction * 100, 2),
        "stake_eur": stake_eur,
    }


# Alias retrocompatibile
def calculate_half_kelly(
    prob: float,
    bookmaker_odds: float,
    bankroll: float = 1000.0,
    min_ev: float = 0.055,
    max_cap: float = 0.015,
) -> Dict[str, Any]:
    """Alias di retrocompatibilità che invoca calculate_quarter_kelly."""
    return calculate_quarter_kelly(
        prob=prob,
        bookmaker_odds=bookmaker_odds,
        bankroll=bankroll,
        min_ev=min_ev,
        max_cap=max_cap,
    )


def scan_match_markets(
    model_output: Dict[str, Any],
    market_odds: Dict[str, float],
    bankroll: float = 1000.0,
    min_ev: float = 0.055,
    min_odds: float = 1.60,
    max_odds: float = 3.10,
    max_cap: float = 0.015,
) -> List[Dict[str, Any]]:
    """
    Confronta le probabilità calcolate da predict.py con le quote di mercato fornite.
    Privilegia le probabilità calibrate tramite Bayesian Shrinkage se disponibili.
    Ritorna la lista delle selezioni con valore positivo (EV > 5.5%, Quote in 1.60-3.10).
    """
    opportunities = []

    # Determina la sorgente di probabilità: Bayesian Shrinkage (consigliata) o Softmax pura
    probs_source = None
    if "Bayesian_Shrinkage_1X2" in model_output:
        probs_source = model_output["Bayesian_Shrinkage_1X2"].get("Probabilita_%", {})
    elif "Rete_Softmax_1X2" in model_output:
        probs_source = model_output["Rete_Softmax_1X2"].get("Probabilita_%", {})

    if probs_source:
        mapping = {"1": "1", "X": "X", "2": "2"}
        for outcome, key in mapping.items():
            if key in market_odds and outcome in probs_source:
                p = probs_source[outcome] / 100.0
                decision = calculate_quarter_kelly(
                    p,
                    market_odds[key],
                    bankroll=bankroll,
                    min_ev=min_ev,
                    min_odds=min_odds,
                    max_odds=max_odds,
                    max_cap=max_cap,
                )
                if decision["actionable"]:
                    decision["market"] = f"1X2 - Esito {outcome}"
                    opportunities.append(decision)

    return opportunities


if __name__ == "__main__":
    print("=" * 65)
    print(" DeepPitch: Test Modulo Decision Engine (Quarter-Kelly)")
    print("=" * 65)

    # Test: Quota Bookmaker 2.40 vs Quota Fair stimata 2.05 (P = 48.78%, EV = +17.07%)
    test_prob = 0.4878
    test_quota = 2.40
    current_bankroll = 1000.0

    res = calculate_quarter_kelly(
        test_prob, test_quota, bankroll=current_bankroll
    )

    print(f"Bankroll attuale:       €{current_bankroll}")
    print(f"Probabilita Modello:    {res['prob_perc']}% (Fair: {res['fair_odds']})")
    print(f"Quota Bookmaker:        {res['bookmaker_odds']}")
    print(f"Expected Value (EV):    +{res['ev_perc']}%")
    print(f"Frazione Quarter-Kelly: {res['applied_fraction_perc']}% (Cap max: 1.5%)")
    print(f"Stake Calcolato:        €{res['stake_eur']}")
    print(f"Giocabile:              {res['actionable']}")
    print("=" * 65)