"""
src/predict_v2.py
Production inference engine for Deep-Pitch v2 (5-Head Residual-Offset Ensemble +
Live FotMob Fresh-Injury/Lineup Adjustment + Shin De-Vigged Market Reference).
"""

import os
import pickle
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.predict import devig_shin
from src.injury_lineup import apply_injury_lineup_adjustment

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BUNDLE_PATH = os.path.join(BASE_DIR, "models", "deep_pitch_v2_bundle.pkl")


class OffsetResidualMLP(nn.Module):
    def __init__(self, d: int, hidden: int = 24, dropout: float = 0.35):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.zeros(3))
        self.fc1 = nn.Linear(d, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden, 3)

    def forward(self, base_logits, x):
        h = self.drop(torch.tanh(self.ln1(self.fc1(x))))
        return self.alpha * base_logits + self.bias + self.fc2(h)


def _softmax_1d(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def _apply_temp_1d(p: np.ndarray, T: float) -> np.ndarray:
    logits = np.log(np.clip(p, 1e-7, 1.0))
    return _softmax_1d(logits / float(T))


_V2_BUNDLE_CACHE: Optional[Dict[str, Any]] = None
_V2_MLP_CACHE: Optional[List[OffsetResidualMLP]] = None


def load_v2_bundle(bundle_path: str = DEFAULT_BUNDLE_PATH) -> Dict[str, Any]:
    global _V2_BUNDLE_CACHE, _V2_MLP_CACHE
    if _V2_BUNDLE_CACHE is not None:
        return _V2_BUNDLE_CACHE

    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Deep-Pitch v2 bundle not found at {bundle_path}. Run src/train_v2.py first.")

    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    d_both = len(bundle["mot_cols"]) + len(bundle["res_cols"])
    mlp_models = []
    for st in bundle["mlp_states"]:
        m = OffsetResidualMLP(d_both, hidden=24, dropout=0.35)
        m.load_state_dict(st)
        m.eval()
        mlp_models.append(m)

    _V2_BUNDLE_CACHE = bundle
    _V2_MLP_CACHE = mlp_models
    return bundle


def predici_partita_v2(
    v2_payload: Dict[str, Any],
    market_odds: Optional[List[float]] = None,
    injury_report: Optional[Dict[str, Any]] = None,
    devig_method: str = "shin",
) -> Dict[str, Any]:
    """
    Runs the 5-Head Deep-Pitch v2 Ensemble on a single live fixture payload,
    applies the conservative Fresh-Injury/Lineup logit adjustment, and attaches
    de-vigged market reference probabilities.
    """
    bundle = load_v2_bundle()
    mlp_models = _V2_MLP_CACHE

    # 1. Generative Base (5 components blended via OOF weights & temperature)
    gen_comps = [np.array(p, dtype=np.float64) for p in v2_payload["gen_components"]]
    w_gen = bundle["w_gen"]
    T_gen = float(bundle["T_gen"])
    p_mix_gen = np.zeros(3, dtype=np.float64)
    for j in range(len(w_gen)):
        p_mix_gen += float(w_gen[j]) * gen_comps[j]
    p_base = _apply_temp_1d(p_mix_gen, T_gen)
    base_logits = np.log(np.clip(p_base, 1e-6, 1.0)).reshape(1, 3)

    # 2. Extract ordered MOT & RES vectors (impute non-finite with training medians)
    mot_dict = v2_payload["mot_features"]
    res_dict = v2_payload["res_features"]
    mot_cols = bundle["mot_cols"]
    res_cols = bundle["res_cols"]

    xm = np.zeros((1, len(mot_cols)), dtype=np.float64)
    for j, col in enumerate(mot_cols):
        val = float(mot_dict.get(col, bundle["mot_medians"][j]))
        xm[0, j] = val if np.isfinite(val) else float(bundle["mot_medians"][j])

    xr = np.zeros((1, len(res_cols)), dtype=np.float64)
    for j, col in enumerate(res_cols):
        val = float(res_dict.get(col, bundle["res_medians"][j]))
        xr[0, j] = val if np.isfinite(val) else float(bundle["res_medians"][j])

    xboth = np.hstack([xm, xr])

    # 3. Head 1: Grouped-Ridge Offset Linear
    lin_cfg = bundle["offset_linear"]
    xr_sc = np.clip(lin_cfg["scaler_res"].transform(xr), -5.0, 5.0)
    z_lin = float(lin_cfg["alpha"]) * base_logits[0] + lin_cfg["bias"] + (xm @ lin_cfg["Wm"])[0] + (xr_sc @ lin_cfg["Wr"])[0]
    p_lin = _softmax_1d(z_lin)

    # 4. Head 2: Residual-Offset LightGBM (5-seed soup)
    lgb_preds = []
    for m in bundle["lgb_models"]:
        raw_z = m.predict(xboth, raw_score=True)[0] + base_logits[0]
        lgb_preds.append(_softmax_1d(raw_z))
    p_lgb = np.mean(lgb_preds, axis=0)

    # 5. Head 3: Residual-Offset XGBoost (5-seed soup)
    xgb_preds = []
    for m in bundle["xgb_models"]:
        p_x = m.predict_proba(xboth, base_margin=base_logits)[0]
        xgb_preds.append(p_x)
    p_xgb = np.mean(xgb_preds, axis=0)

    # 6. Head 4: Residual-Offset MLP (10-seed Model Soup)
    xb_sc = np.clip(bundle["scaler_both"].transform(xboth), -5.0, 5.0)
    bt = torch.tensor(base_logits, dtype=torch.float32)
    xt = torch.tensor(xb_sc, dtype=torch.float32)
    mlp_preds = []
    with torch.no_grad():
        for m in mlp_models:
            mlp_preds.append(F.softmax(m(bt, xt), dim=1).numpy()[0])
    p_mlp = np.mean(mlp_preds, axis=0)

    # 7. Final OOF-Calibrated 5-Head Ensemble
    w_ens = bundle["w_ens"]
    T_ens = float(bundle["T_ens"])
    heads = [p_base, p_lin, p_lgb, p_xgb, p_mlp]
    p_mix = np.zeros(3, dtype=np.float64)
    for j in range(len(w_ens)):
        p_mix += float(w_ens[j]) * heads[j]
    p_v2_raw = _apply_temp_1d(p_mix, T_ens)

    # 8. Apply Live FotMob Fresh-Injury & Lineup Delta adjustment
    p_final = apply_injury_lineup_adjustment(p_v2_raw.tolist(), injury_report)

    res: Dict[str, Any] = {
        "Version": bundle.get("version", "deep-pitch-v2.0"),
        "V2_Raw_1X2": {
            "Probabilita_%": {
                "1": round(float(p_v2_raw[0]) * 100.0, 2),
                "X": round(float(p_v2_raw[1]) * 100.0, 2),
                "2": round(float(p_v2_raw[2]) * 100.0, 2),
            },
            "Fair_Odds": {
                "1": round(1.0 / max(float(p_v2_raw[0]), 1e-6), 2),
                "X": round(1.0 / max(float(p_v2_raw[1]), 1e-6), 2),
                "2": round(1.0 / max(float(p_v2_raw[2]), 1e-6), 2),
            },
        },
        "Final_1X2": {
            "Probabilita_%": {
                "1": round(float(p_final[0]) * 100.0, 2),
                "X": round(float(p_final[1]) * 100.0, 2),
                "2": round(float(p_final[2]) * 100.0, 2),
            },
            "Fair_Odds": {
                "1": round(1.0 / max(float(p_final[0]), 1e-6), 2),
                "X": round(1.0 / max(float(p_final[1]), 1e-6), 2),
                "2": round(1.0 / max(float(p_final[2]), 1e-6), 2),
            },
            "Probs_Float": [float(p_final[0]), float(p_final[1]), float(p_final[2])],
        },
        "Heads_1X2": {
            "Gen_Base": [round(float(x) * 100.0, 2) for x in p_base],
            "Offset_Linear": [round(float(x) * 100.0, 2) for x in p_lin],
            "Offset_LGBM": [round(float(x) * 100.0, 2) for x in p_lgb],
            "Offset_XGB": [round(float(x) * 100.0, 2) for x in p_xgb],
            "Offset_MLP": [round(float(x) * 100.0, 2) for x in p_mlp],
        },
    }

    if market_odds is not None and len(market_odds) == 3:
        if devig_method == "shin":
            p_book, shin_z = devig_shin(market_odds)
        else:
            invs = [1.0 / q for q in market_odds]
            s = sum(invs)
            p_book = [inv / s for inv in invs]
            shin_z = 0.0
        res["Market_Reference"] = {
            "Market_Odds": {"1": market_odds[0], "X": market_odds[1], "2": market_odds[2]},
            "De_Vigged_Market_%": {
                "1": round(float(p_book[0]) * 100.0, 2),
                "X": round(float(p_book[1]) * 100.0, 2),
                "2": round(float(p_book[2]) * 100.0, 2),
            },
            "Shin_Z": round(float(shin_z), 5),
        }

    return res
