import math
import torch
import torch.nn.functional as F
import json

# 1. Carica il checkpoint del modello salvato
checkpoint = torch.load("models/modello_calcio_v1.pt")

emb = checkpoint["emb"]
w1, b1 = checkpoint["w1"], checkpoint["b1"]
w2, b2 = checkpoint["w2"], checkpoint["b2"]
w3, b3 = checkpoint["w3"], checkpoint["b3"]
w4, b4 = checkpoint["w4"], checkpoint["b4"]
median, iqr = checkpoint["median"], checkpoint["iqr"]
campionati_map = checkpoint["campionati"]


def poisson_pmf(k, lambda_val):
    """Calcola la probabilità P(X = k) per una variabile di Poisson con media lambda_val."""
    return (lambda_val**k * math.exp(-lambda_val)) / math.factorial(k)


def calcola_poisson_stats(xg_casa, xg_trasf, max_gol=7):
    """
    Genera la matrice dei risultati esatti e ricava le probabilità per:
    1X2, Under/Over 2.5, Goal/No Goal.
    """
    p_1 = 0.0
    p_X = 0.0
    p_2 = 0.0
    p_under25 = 0.0
    p_goal = 0.0

    for i in range(max_gol):  # Gol Casa
        prob_i = poisson_pmf(i, xg_casa)
        for j in range(max_gol):  # Gol Trasferta
            prob_j = poisson_pmf(j, xg_trasf)
            prob_punteggio = prob_i * prob_j

            # 1X2
            if i > j:
                p_1 += prob_punteggio
            elif i == j:
                p_X += prob_punteggio
            else:
                p_2 += prob_punteggio

            # Under / Over 2.5
            if (i + j) < 2.5:
                p_under25 += prob_punteggio

            # Goal / No Goal
            if i > 0 and j > 0:
                p_goal += prob_punteggio

    p_over25 = 1.0 - p_under25
    p_nogoal = 1.0 - p_goal

    return {
        "1X2": {
            "1": round(p_1 * 100, 1),
            "X": round(p_X * 100, 1),
            "2": round(p_2 * 100, 1),
        },
        "Fair_Odds_1X2": {
            "1": round(1.0 / (p_1 + 1e-8), 2),
            "X": round(1.0 / (p_X + 1e-8), 2),
            "2": round(1.0 / (p_2 + 1e-8), 2),
        },
        "Under_Over_2.5": {
            "Under_2.5_%": round(p_under25 * 100, 1),
            "Over_2.5_%": round(p_over25 * 100, 1),
            "Quota_Under": round(1.0 / (p_under25 + 1e-8), 2),
            "Quota_Over": round(1.0 / (p_over25 + 1e-8), 2),
        },
        "Goal_NoGoal": {
            "Goal_%": round(p_goal * 100, 1),
            "NoGoal_%": round(p_nogoal * 100, 1),
            "Quota_Goal": round(1.0 / (p_goal + 1e-8), 2),
            "Quota_NoGoal": round(1.0 / (p_nogoal + 1e-8), 2),
        },
    }


def predici_partita(codice_campionato, feature_57_grezze):
    """
    Input:
      - codice_campionato (str): 'E0', 'F1', 'SP1', 'I1', oppure 'D1'
      - feature_57_grezze (list/Tensor): 57 float non normalizzati
    """
    with torch.no_grad():
        # 1. Normalizzazione feature numeriche tramite lo scaler del Training Set
        x_num = torch.tensor(feature_57_grezze, dtype=torch.float32)
        x_num_scaled = (x_num - median) / (iqr + 1e-8)

        # 2. Lookup Embedding Campionato
        div_id = campionati_map[codice_campionato]
        div_emb = emb[div_id]

        # 3. Concatenazione (Dimensione = 60)
        x_input = torch.cat([x_num_scaled, div_emb]).unsqueeze(0)

        # 4. Forward Pass
        h = torch.tanh(x_input @ w1 + b1)
        k = torch.tanh(h @ w2 + b2)

        output_xg = F.softplus(k @ w3 + b3).squeeze(0)
        logits_1x2 = (k @ w4 + b4).squeeze(0)
        probs_1x2 = F.softmax(logits_1x2, dim=0)

        xg_c, xg_t = output_xg[0].item(), output_xg[1].item()
        p1_net, px_net, p2_net = (
            probs_1x2[0].item(),
            probs_1x2[1].item(),
            probs_1x2[2].item(),
        )

        # 5. Calcolo Poisson dagli xG predetti
        poisson_res = calcola_poisson_stats(xg_c, xg_t)

        return {
            "xG_Predetti": {"Casa": round(xg_c, 2), "Trasferta": round(xg_t, 2)},
            "Rete_Softmax_1X2": {
                "Probabilita_%": {
                    "1": round(p1_net * 100, 1),
                    "X": round(px_net * 100, 1),
                    "2": round(p2_net * 100, 1),
                },
                "Fair_Odds": {
                    "1": round(1.0 / (p1_net + 1e-8), 2),
                    "X": round(1.0 / (px_net + 1e-8), 2),
                    "2": round(1.0 / (p2_net + 1e-8), 2),
                },
            },
            "Poisson_Stats": poisson_res,
        }

def stampa_report_partita(res, nome_casa="Casa", nome_trasf="Trasferta"):
    print("=" * 60)
    print(f"   REPORT PREDIZIONE: {nome_casa.upper()} vs {nome_trasf.upper()}")
    print("=" * 60)

    # 1. xG
    xg_c = res["xG_Predetti"]["Casa"]
    xg_t = res["xG_Predetti"]["Trasferta"]
    print(f"\n📊 xG PREDETTI:  {nome_casa} {xg_c} - {xg_t} {nome_trasf}")

    # 2. Rete Neurale 1X2
    s_prob = res["Rete_Softmax_1X2"]["Probabilita_%"]
    s_odds = res["Rete_Softmax_1X2"]["Fair_Odds"]
    print("\n🧠 RETE NEURALE (1X2 Direct):")
    print(
        f"   • [1]: {s_prob['1']}%  (Quota Fair: {s_odds['1']})"
    )
    print(
        f"   • [X]: {s_prob['X']}%  (Quota Fair: {s_odds['X']})"
    )
    print(
        f"   • [2]: {s_prob['2']}%  (Quota Fair: {s_odds['2']})"
    )

    # 3. Poisson 1X2
    p_prob = res["Poisson_Stats"]["1X2"]
    p_odds = res["Poisson_Stats"]["Fair_Odds_1X2"]
    print("\n🎲 SIMULAZIONE POISSON (1X2):")
    print(
        f"   • [1]: {p_prob['1']}%  (Quota Fair: {p_odds['1']})"
    )
    print(
        f"   • [X]: {p_prob['X']}%  (Quota Fair: {p_odds['X']})"
    )
    print(
        f"   • [2]: {p_prob['2']}%  (Quota Fair: {p_odds['2']})"
    )

    # 4. Mercati accessori
    uo = res["Poisson_Stats"]["Under_Over_2.5"]
    gn = res["Poisson_Stats"]["Goal_NoGoal"]
    print("\n🎯 MERCATI ACCESSORI (Poisson):")
    print(
        f"   • Over 2.5:  {uo['Over_2.5_%']}%  | Quota Fair: {uo['Quota_Over']}"
    )
    print(
        f"   • Under 2.5: {uo['Under_2.5_%']}%  | Quota Fair: {uo['Quota_Under']}"
    )
    print(
        f"   • Goal:      {gn['Goal_%']}%  | Quota Fair: {gn['Quota_Goal']}"
    )
    print(
        f"   • No Goal:   {gn['NoGoal_%']}%  | Quota Fair: {gn['Quota_NoGoal']}"
    )
    print("=" * 60)





res=predici_partita("I1", [1793.93, 1728.45, 5.0, 14.0, 3.0, 0.0, 2.1333, 0.5333, 0.3333, 2.8, 1.5907, 1.0827, 1.2213, 0.8267, 1.0193, 1.364, 0.628, 0.9533, 1.5333, 1.0667, 0.0, 1.6, 1.0, 1.6, 0.6667, 0.8, 0.0573, 0.016, 1.8, 1.2, 1.2, 0.4, 1.084, 0.842, 1.2, 1.0, 8.0, 7.0, 2.0, 2.0, 0.0, 0.0, -3.0, 0.0, -3.0, 3.0, 0.0, 2.0, 1.0, 13.7333, 13.3333, 4.9333, 3.6, 5.0, 7.9333, 0.8667, 2.8667]
)
stampa_report_partita(res, "Atalanta", "Bologna")