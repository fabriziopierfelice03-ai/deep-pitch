import os
import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class BettingNet(nn.Module):
    def __init__(self, input_dim=56):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 25)
        self.fc2 = nn.Linear(25, 16)
        self.fc3 = nn.Linear(16, 3)
        self.drop = nn.Dropout(p=0.10)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        h = self.drop(h)
        h = torch.tanh(self.fc2(h))
        return self.fc3(h)


# Bayesian Shrinkage Logistic Regression Hyperparameters
# Fitted on 14,400 training fixtures via maximum likelihood
BAYES_ALPHA = 0.0806
BAYES_BETA_MODEL = 0.6173   # 55.1% Relative Weight
BAYES_BETA_BOOK = 0.5035    # 44.9% Relative Weight


def _find_checkpoint():
    candidates = [
        "models/modello_calcio_v1.pt",
        "modello_calcio_v1.pt",
        os.path.join(os.path.dirname(__file__), "..", "models", "modello_calcio_v1.pt"),
        os.path.join(os.path.dirname(__file__), "modello_calcio_v1.pt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return "models/modello_calcio_v1.pt"


# Load model checkpoint
checkpoint_path = _find_checkpoint()
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

median = checkpoint["median"]
iqr = checkpoint["iqr"]
n_features = checkpoint.get("n_features", 56)

model = BettingNet(n_features)
model.load_state_dict(checkpoint["model_state"])
model.eval()


def logit(p, eps=1e-5):
    """Numerically stable logit transformation."""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def apply_bayesian_shrinkage(raw_probs, market_odds):
    """
    Combines raw neural network probabilities with de-vigged bookmaker odds
    using empirical Bayesian Shrinkage via logistic regression:
      z_i = alpha + beta_model * logit(p_model,i) + beta_book * logit(p_book,i)
      p_calib,i = sigma(z_i) / sum(sigma(z_j))

    Parameters:
      - raw_probs: list or array of 3 floats [p1, pX, p2] summing to 1.0
      - market_odds: list or array of 3 floats [q1, qX, q2] decimal odds

    Returns:
      - dict with calibrated probabilities, fair odds, and de-vigged book probabilities.
    """
    q1, qx, q2 = market_odds
    invs = [1.0 / q1, 1.0 / qx, 1.0 / q2]
    margin_sum = sum(invs)
    p_book = [inv / margin_sum for inv in invs]

    logits_model = [logit(p) for p in raw_probs]
    logits_book = [logit(p) for p in p_book]

    z = [
        BAYES_ALPHA + BAYES_BETA_MODEL * lm + BAYES_BETA_BOOK * lb
        for lm, lb in zip(logits_model, logits_book)
    ]

    p_raw = [1.0 / (1.0 + np.exp(-zi)) for zi in z]
    total_p = sum(p_raw)
    p_calib = [p / total_p for p in p_raw]

    return {
        "calibrated_probs": p_calib,
        "book_probs": p_book,
        "fair_odds": [round(1.0 / p, 2) for p in p_calib],
    }


def predici_partita(arg1, arg2=None, market_odds=None):
    """
    Universal Match Prediction Engine.
    Accepts either:
      - predici_partita(features_56, market_odds=None)
      - predici_partita(codice_campionato, features_56, market_odds=None) (backwards-compatible)

    Returns dictionary with raw Softmax 1X2 probabilities, fair odds,
    and optional Bayesian Shrinkage calibrated distribution if market_odds are provided.
    """
    if isinstance(arg1, str) and arg2 is not None:
        features = arg2
    else:
        features = arg1

    if len(features) != 56:
        raise ValueError(f"Feature vector must have exactly 56 elements, got {len(features)}")

    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        x_scaled = (x - median) / (iqr + 1e-8)
        logits = model(x_scaled.unsqueeze(0)).squeeze(0)
        probs = F.softmax(logits, dim=0).tolist()

    p1, px, p2 = probs

    result = {
        "Rete_Softmax_1X2": {
            "Probabilita_%": {
                "1": round(p1 * 100, 2),
                "X": round(px * 100, 2),
                "2": round(p2 * 100, 2),
            },
            "Fair_Odds": {
                "1": round(1.0 / (p1 + 1e-8), 2),
                "X": round(1.0 / (px + 1e-8), 2),
                "2": round(1.0 / (p2 + 1e-8), 2),
            },
        }
    }

    if market_odds is not None and len(market_odds) == 3:
        shrunk = apply_bayesian_shrinkage([p1, px, p2], market_odds)
        p1_c, px_c, p2_c = shrunk["calibrated_probs"]
        q1, qx, q2 = market_odds

        result["Bayesian_Shrinkage_1X2"] = {
            "Probabilita_%": {
                "1": round(p1_c * 100, 2),
                "X": round(px_c * 100, 2),
                "2": round(p2_c * 100, 2),
            },
            "Fair_Odds": {
                "1": shrunk["fair_odds"][0],
                "X": shrunk["fair_odds"][1],
                "2": shrunk["fair_odds"][2],
            },
            "Market_Odds": {"1": q1, "X": qx, "2": q2},
            "De_Vigged_Market_%": {
                "1": round(shrunk["book_probs"][0] * 100, 2),
                "X": round(shrunk["book_probs"][1] * 100, 2),
                "2": round(shrunk["book_probs"][2] * 100, 2),
            },
            "Relative_Weights": {"Model": "55.1%", "Market": "44.9%"},
        }

    return result


def stampa_report_partita(res, nome_casa="Casa", nome_trasf="Trasferta"):
    print("=" * 65)
    print(f"   DEEPPITCH PREDICTION REPORT: {nome_casa.upper()} vs {nome_trasf.upper()}")
    print("=" * 65)

    s_prob = res["Rete_Softmax_1X2"]["Probabilita_%"]
    s_odds = res["Rete_Softmax_1X2"]["Fair_Odds"]
    print("\n[🧠] RAW NEURAL NETWORK (Direct 1X2 Softmax):")
    print(f"   • [1]: {s_prob['1']}%  (Fair Odd: {s_odds['1']})")
    print(f"   • [X]: {s_prob['X']}%  (Fair Odd: {s_odds['X']})")
    print(f"   • [2]: {s_prob['2']}%  (Fair Odd: {s_odds['2']})")

    if "Bayesian_Shrinkage_1X2" in res:
        b_prob = res["Bayesian_Shrinkage_1X2"]["Probabilita_%"]
        b_odds = res["Bayesian_Shrinkage_1X2"]["Fair_Odds"]
        m_odds = res["Bayesian_Shrinkage_1X2"]["Market_Odds"]
        m_prob = res["Bayesian_Shrinkage_1X2"]["De_Vigged_Market_%"]

        print("\n[⚖️] BAYESIAN SHRINKAGE CALIBRATION (Model 55.1% | Market 44.9%):")
        print(f"   • [1]: {b_prob['1']}% (Fair Odd: {b_odds['1']} | Book: {m_odds['1']} [Imp: {m_prob['1']}%])")
        print(f"   • [X]: {b_prob['X']}% (Fair Odd: {b_odds['X']} | Book: {m_odds['X']} [Imp: {m_prob['X']}%])")
        print(f"   • [2]: {b_prob['2']}% (Fair Odd: {b_odds['2']} | Book: {m_odds['2']} [Imp: {m_prob['2']}%])")
    print("=" * 65)


if __name__ == "__main__":
    sample_56 = [
        1620.0, 1740.0, 8.0, 10.0, 3.0, 4.0, 1.0, 1.133, 1.667, 0.0, 1.637,
        1.343, 1.978, 0.35, 1.91, 1.398, 1.944, 3.07, 1.267, 1.467, 1.667,
        0.0, 2.067, 1.933, 2.0, 4.0, 0.37, -0.124, 1.3, 1.3, 1.4, 1.2, 1.33,
        1.14, 1.5, 1.5, 7.0, 7.0, 2.0, 2.0, -2.0, -3.0, -1.0, -2.0, 3.0,
        3.0, 4.0, 1.0, 17.0, 13.8, 6.4, 4.933, 5.667, 4.6, 1.467, 1.733
    ]
    sample_odds = [2.70, 3.30, 2.60]
    out = predici_partita(sample_56, market_odds=sample_odds)
    stampa_report_partita(out, "Lazio", "Milan")