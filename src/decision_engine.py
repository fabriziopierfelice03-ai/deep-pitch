"""
DeepPitch: Decision & Money Management Engine
Calculates Expected Value (EV), Half-Kelly Staking, and risk-management caps.
"""

from typing import Dict, Any, List


def calculate_half_kelly(
    prob: float,
    bookmaker_odds: float,
    bankroll: float = 1117.0,
    min_ev: float = 0.05,
    max_cap: float = 0.05,
) -> Dict[str, Any]:
    """
    Calcola l'Expected Value e il dimensionamento dello stake con Half-Kelly.

    Parametri:
    - prob: probabilità stimata dal modello (compresa tra 0.0 e 1.0)
    - bookmaker_odds: quota decimale offerta dal bookmaker
    - bankroll: capitale totale attuale (default 1117.0€)
    - min_ev: soglia minima di valore atteso (default 5% = 0.05)
    - max_cap: quota massima del bankroll allocabile su una singola scommessa (default 5% = 0.05)
    """
    if bookmaker_odds <= 1.0 or prob <= 0.0 or prob > 1.0:
        return {
            "actionable": False,
            "reason": "Quota o probabilita non valide",
            "ev_perc": 0.0,
            "stake_eur": 0.0,
        }

    # Calcolo Expected Value: (P * Quota) - 1
    ev = (prob * bookmaker_odds) - 1.0

    # Se non c'è margine sufficiente rispetto alla quota fair, scartiamo la bet
    if ev < min_ev:
        return {
            "actionable": False,
            "reason": f"EV ({round(ev * 100, 2)}%) inferiore alla soglia minima ({round(min_ev * 100, 2)}%)",
            "ev_perc": round(ev * 100, 2),
            "stake_eur": 0.0,
        }

    # Formula di Kelly: f* = (b*p - q) / b = EV / (Quota - 1)
    b = bookmaker_odds - 1.0
    kelly_full = ev / b

    # Half-Kelly per ridurre la varianza e proteggere il bankroll dai drawdown
    half_kelly = kelly_full * 0.5

    # Circuit Breaker: Applichiamo il tetto massimo prudenziale (max 5%)
    capped_fraction = max(0.0, min(half_kelly, max_cap))
    stake_eur = round(bankroll * capped_fraction, 2)

    fair_odds = round(1.0 / prob, 2)

    return {
        "actionable": True,
        "prob_perc": round(prob * 100, 2),
        "fair_odds": fair_odds,
        "bookmaker_odds": bookmaker_odds,
        "ev_perc": round(ev * 100, 2),
        "kelly_raw_perc": round(half_kelly * 100, 2),
        "applied_fraction_perc": round(capped_fraction * 100, 2),
        "stake_eur": stake_eur,
    }


def scan_match_markets(
    model_output: Dict[str, Any],
    market_odds: Dict[str, float],
    bankroll: float = 1117.0,
) -> List[Dict[str, Any]]:
    """
    Confronta tutte le quote calcolate da predict.py con le quote di mercato fornite.
    Ritorna la lista delle sole selezioni con valore positivo (EV >= 5%).
    """
    opportunities = []

    # 1. Mercato 1X2 (Dalla Rete Softmax)
    if "Rete_Softmax_1X2" in model_output:
        probs = model_output["Rete_Softmax_1X2"].get("Probabilita_%", {})
        mapping = {"1": "1", "X": "X", "2": "2"}
        for outcome, key in mapping.items():
            if key in market_odds and outcome in probs:
                p = probs[outcome] / 100.0
                decision = calculate_half_kelly(p, market_odds[key], bankroll)
                if decision["actionable"]:
                    decision["market"] = f"1X2 - Esito {outcome}"
                    opportunities.append(decision)

    # 2. Mercato Over/Under 2.5 (Da Poisson)
    if "Poisson_Stats" in model_output:
        poisson = model_output["Poisson_Stats"]
        if "Under_Over_2.5" in poisson:
            p_over = poisson["Under_Over_2.5"].get("Prob_Over_%", 0.0) / 100.0
            if "Over_2.5" in market_odds:
                decision = calculate_half_kelly(
                    p_over, market_odds["Over_2.5"], bankroll
                )
                if decision["actionable"]:
                    decision["market"] = "Over 2.5 Gol"
                    opportunities.append(decision)

        # 3. Mercato Gol/NoGol (BTTS da Poisson)
        if "Gol_NoGol" in poisson:
            p_btts = poisson["Gol_NoGol"].get("Prob_Gol_%", 0.0) / 100.0
            if "BTTS_Si" in market_odds:
                decision = calculate_half_kelly(
                    p_btts, market_odds["BTTS_Si"], bankroll
                )
                if decision["actionable"]:
                    decision["market"] = "Entrambe le Squadre Segnano (Gol)"
                    opportunities.append(decision)

    return opportunities


if __name__ == "__main__":
    print("=" * 65)
    print(" DeepPitch: Test Modulo Decision Engine (Half-Kelly)")
    print("=" * 65)

    # Test di verifica: Quota Bookmaker 2.10 vs Quota Fair stimata 1.75 (P = 57.14%)
    test_prob = 0.5714
    test_quota = 2.10
    current_bankroll = 1117.0

    res = calculate_half_kelly(
        test_prob, test_quota, bankroll=current_bankroll
    )

    print(f"Bankroll attuale:       €{current_bankroll}")
    print(f"Probabilita Modello:    {res['prob_perc']}% (Fair: {res['fair_odds']})")
    print(f"Quota Bookmaker:        {res['bookmaker_odds']}")
    print(f"Expected Value (EV):    +{res['ev_perc']}%")
    print(f"Frazione Half-Kelly:    {res['applied_fraction_perc']}%")
    print(f"Stake Calcolato:        €{res['stake_eur']}")
    print(f"Giocabile:              {res['actionable']}")
    print("=" * 65)