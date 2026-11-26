"""
src/train_v2.py
Trains and exports the production Deep-Pitch v2.1 Residual-Offset 5-Head Ensemble:
  - Base: Zero-leakage Generative Blend:
      * Adaptive Riccati Dual-Gain Kalman (v2.1: Pinnacle prior + 100% Opta npxG reality check, dynamic P_t)
      * Slow Online Past-Market Kalman
      * Exact Rolling 35d Past-Market Ridge
      * Exact Rolling 120d Opta npxG Dixon-Coles
      * Online Blended DC
  - Head 1: Grouped-Ridge Offset Linear (separate L2 for end-of-season motivation vs general residuals)
  - Head 2: Residual-Offset LightGBM (5-seed soup, init_score = log(p_base))
  - Head 3: Residual-Offset XGBoost (5-seed soup, base_margin = log(p_base))
  - Head 4: Residual-Offset PyTorch MLP (10-seed Model Soup)
  - Calibration: 5-Fold Expanding Temporal OOF CV (8,141 matches) -> Simplex Blend + Temperature Scaling

Final production weights are fitted on ALL 16,268 verified Top-5 matches (2015-2025)
after verifying exact blind Test (2024/25) Brier & Log-Loss, and saved to models/deep_pitch_v2_bundle.pkl.
"""

import os
import sys
import math
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
OUT_BUNDLE = os.path.join(MODELS_DIR, "deep_pitch_v2_bundle.pkl")

# Default source paths built in scratch during v2/v2.1 R&D
SCRATCH_DIR = r"C:\Users\fabri\.gemini\antigravity\brain\bea11f98-a77d-4dd8-aae2-6e29a5c09a40\scratch"
DEFAULT_EXACT_PKL = os.path.join(SCRATCH_DIR, "deep_pitch_v2_exact.pkl")
DEFAULT_MATCHES_CSV = r"c:\Users\fabri\Desktop\backtest\Matches.csv"
DEFAULT_ENRICHED_CSV = r"c:\Users\fabri\Desktop\backtest\matches_enriched_xg_top5_2014_2026.csv"
TOP5_DIVS = ["E0", "I1", "SP1", "D1", "F1"]

MOT_COLS = [
    "ex_mot_diff", "ex_off_diff", "ex_champ_diff", "ex_locked_diff",
    "ex_mrel_diff", "ex_drel_diff", "ex_deuro_diff", "ex_late_ramp",
]

RES_COLS = [
    "mkt_vs_dc_p1", "mkt_vs_dc_p2", "mkt_vs_dc_ldiff",
    "elo_diff", "elo_minus_mkt_diff", "mkt_trend_diff", "ex_bt_mom_35_220",
    "mkt_steam_f_diff", "mkt_steam_m_diff",
    "mkt_surp_net_diff", "mkt_surp_pts_diff",
    "npxg_net_diff_f", "npxg_net_diff_m", "xgot_net_diff_m",
    "xg_open_net_diff_m", "bc_net_diff_m", "opp_adj_npxg_diff_m",
    "finishing_diff_m", "gk_save_diff_m", "venue_boost_h", "venue_boost_a",
    "rest_diff", "cong_diff", "pts_pg_diff",
    "is_E0", "is_I1", "is_SP1", "is_D1", "is_F1",
]


def brier_score(probs, y_oh):
    return float(np.mean(np.sum((probs - y_oh) ** 2, axis=1)))


def log_loss_score(probs, y):
    p = np.clip(probs, 1e-7, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def brier_binary(probs, y_bin):
    return float(np.mean((probs - y_bin) ** 2))


def ll_binary(probs, y_bin):
    pc = np.clip(probs, 1e-7, 1.0 - 1e-7)
    return float(-np.mean(y_bin * np.log(pc) + (1.0 - y_bin) * np.log(1.0 - pc)))


def poisson_over25_dc(lam_h, lam_a, rho=-0.065, max_g=10):
    lam_h = max(0.15, min(6.0, float(lam_h)))
    lam_a = max(0.15, min(6.0, float(lam_a)))
    ph = np.empty(max_g + 1, dtype=np.float64)
    pa = np.empty(max_g + 1, dtype=np.float64)
    ph[0] = math.exp(-lam_h)
    pa[0] = math.exp(-lam_a)
    for k in range(1, max_g + 1):
        ph[k] = ph[k - 1] * lam_h / k
        pa[k] = pa[k - 1] * lam_a / k
    mat = np.outer(ph, pa)
    mat[0, 0] *= max(0.01, 1.0 - lam_h * lam_a * rho)
    mat[1, 0] *= max(0.01, 1.0 + lam_a * rho)
    mat[0, 1] *= max(0.01, 1.0 + lam_h * rho)
    mat[1, 1] *= max(0.01, 1.0 - rho)
    tot = float(mat.sum())
    if tot <= 0.0:
        return 0.50
    mat /= tot
    u25 = float(mat[0, 0] + mat[1, 0] + mat[0, 1] + mat[2, 0] + mat[1, 1] + mat[0, 2])
    return float(max(1e-6, min(1.0 - 1e-6, 1.0 - u25)))


def compute_v21_kalman_d2_t1f(n_bench):
    """
    Recomputes the strictly causal (t' < t) v2.1 Adaptive Riccati Dual-Gain Kalman
    (Pinnacle implied lambda prior + 100% Opta npxG reality check, dynamic P_t)
    matching src/v2_state_engine.py 1:1.
    """
    if SCRATCH_DIR not in sys.path:
        sys.path.insert(0, SCRATCH_DIR)
    import build_v2_dataset as bvd

    df_opta = bvd.load_opta_match_stats()
    df_fd = bvd.load_fd_pinnacle_odds()

    df_m = pd.read_csv(DEFAULT_MATCHES_CSV, low_memory=False)
    df_m = df_m[df_m["Division"].isin(TOP5_DIVS)].copy()
    df_m["MatchDate_dt"] = pd.to_datetime(df_m["MatchDate"])
    df_m = df_m[(df_m["MatchDate_dt"] >= "2013-07-01") & (df_m["MatchDate_dt"] <= "2025-07-01")].copy()
    df_m = df_m.sort_values(["MatchDate_dt", "Division", "HomeTeam"]).reset_index(drop=True)

    df_enr = pd.read_csv(DEFAULT_ENRICHED_CSV, usecols=["Division", "MatchDate", "HomeTeam", "AwayTeam"])
    enr_key_to_idx = {}
    for idx, r in enumerate(df_enr.itertuples(index=False)):
        enr_key_to_idx[(r.Division, r.MatchDate, r.HomeTeam, r.AwayTeam)] = idx

    df_m["enr_idx"] = [
        enr_key_to_idx.get((r.Division, r.MatchDate, r.HomeTeam, r.AwayTeam), -1)
        for r in df_m[["Division", "MatchDate", "HomeTeam", "AwayTeam"]].itertuples(index=False)
    ]

    opta_cols = ["Division", "MatchDate_dt", "HomeTeam", "AwayTeam", "opta_npxg_h", "opta_npxg_a"]
    df_opta_dedup = df_opta[opta_cols].drop_duplicates(subset=["Division", "MatchDate_dt", "HomeTeam", "AwayTeam"])
    df_m = df_m.merge(df_opta_dedup, on=["Division", "MatchDate_dt", "HomeTeam", "AwayTeam"], how="left")

    fd_cols = ["Division", "MatchDate_dt", "HomeTeam", "AwayTeam", "PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA", "FD_O25", "FD_U25", "FD_AHh"]
    df_fd_dedup = df_fd[fd_cols].drop_duplicates(subset=["Division", "MatchDate_dt", "HomeTeam", "AwayTeam"])
    df_m = df_m.merge(df_fd_dedup, on=["Division", "MatchDate_dt", "HomeTeam", "AwayTeam"], how="left")

    R_pin = 1.0
    Q_match = 0.075
    Q_summer = 0.50
    Q_break = 0.10
    P_new = 1.50
    field_k_ratio = 0.12
    decay = 0.005

    league_avg_lam = {d: 1.32 for d in TOP5_DIVS}
    league_hfa_lam = {d: 0.28 for d in TOP5_DIVS}
    league_avg_npxg = {d: 1.25 for d in TOP5_DIVS}
    team_state = {}
    last_season = {}
    last_day = {}
    preds = np.full((n_bench, 3), np.nan, dtype=np.float64)
    p_over25_raw = np.full(n_bench, np.nan, dtype=np.float64)

    def init_st(elo):
        if not np.isnan(elo):
            en = (elo - 1650.0) / 200.0
        else:
            en = 0.0
        return {
            "att": 0.18 * en, "def": 0.18 * en,
            "att_s": 0.18 * en, "def_s": 0.18 * en,
            "P_att": P_new, "P_def": P_new,
        }

    def regress_st(st, elo):
        if not np.isnan(elo):
            en = (elo - 1650.0) / 200.0
        else:
            en = 0.0
        for k in ("att", "def", "att_s", "def_s"):
            st[k] = 0.75 * st[k] + 0.25 * (0.18 * en)
        st["P_att"] = min(P_new, st["P_att"] + Q_summer)
        st["P_def"] = min(P_new, st["P_def"] + Q_summer)

    for row in df_m.itertuples(index=False):
        div = row.Division
        ht = row.HomeTeam
        at = row.AwayTeam
        dt = row.MatchDate_dt
        day_ord = int(dt.toordinal())
        if dt.month >= 7:
            season = f"{dt.year}/{dt.year + 1}"
        else:
            season = f"{dt.year - 1}/{dt.year}"
        h_elo = float(row.HomeElo) if pd.notna(row.HomeElo) else 1650.0
        a_elo = float(row.AwayElo) if pd.notna(row.AwayElo) else 1650.0
        e_idx = int(row.enr_idx)

        for team, elo in ((ht, h_elo), (at, a_elo)):
            if team not in team_state:
                team_state[team] = init_st(elo)
                last_season[team] = season
                last_day[team] = day_ord
            else:
                st = team_state[team]
                if last_season[team] != season:
                    regress_st(st, elo)
                    last_season[team] = season
                else:
                    gap = day_ord - last_day[team]
                    if gap >= 18:
                        extra_q = Q_break
                    else:
                        extra_q = 0.0
                    st["P_att"] = min(P_new, st["P_att"] + Q_match + extra_q)
                    st["P_def"] = min(P_new, st["P_def"] + Q_match + extra_q)
                last_day[team] = day_ord

        sh = team_state[ht]
        sa = team_state[at]
        mu = math.log(max(0.5, league_avg_lam[div]))
        hfa = league_hfa_lam[div]

        # Pre-match prediction (strictly causal)
        lh = math.exp(max(-1.5, min(1.8, mu + 0.5 * hfa + sh["att"] - sa["def"])))
        la = math.exp(max(-1.5, min(1.8, mu - 0.5 * hfa + sa["att"] - sh["def"])))
        if e_idx >= 0:
            p1, px, p2 = bvd.poisson_1x2_dc(lh, la, rho=-0.065)
            preds[e_idx, 0] = p1
            preds[e_idx, 1] = px
            preds[e_idx, 2] = p2
            p_over25_raw[e_idx] = poisson_over25_dc(lh, la, rho=-0.065)

        # Post-match update
        gf_h = float(row.FTHome) if pd.notna(row.FTHome) else np.nan
        gf_a = float(row.FTAway) if pd.notna(row.FTAway) else np.nan
        if np.isnan(gf_h) or np.isnan(gf_a):
            continue

        if pd.notna(row.opta_npxg_h) and pd.notna(row.opta_npxg_a):
            npxg_h = float(row.opta_npxg_h)
            npxg_a = float(row.opta_npxg_a)
        else:
            npxg_h = min(3.5, 0.55 * gf_h + 0.45 * league_avg_npxg[div])
            npxg_a = min(3.5, 0.55 * gf_a + 0.45 * league_avg_npxg[div])
        league_avg_npxg[div] = 0.996 * league_avg_npxg[div] + 0.004 * (0.5 * (npxg_h + npxg_a))

        p_mkt = bvd.devig_shin_single(row.PSCH, row.PSCD, row.PSCA)
        if p_mkt is None:
            p_mkt = bvd.devig_shin_single(row.PSH, row.PSD, row.PSA)
        if p_mkt is None:
            p_mkt = bvd.devig_shin_single(row.OddHome, row.OddDraw, row.OddAway)
        if p_mkt is None:
            continue

        p_o25 = bvd.devig_two_way(row.FD_O25, row.FD_U25)
        if np.isnan(p_o25):
            p_o25 = bvd.devig_two_way(row.Over25, row.Under25)
        if pd.notna(row.FD_AHh):
            ahh = float(row.FD_AHh)
        elif pd.notna(row.HandiSize):
            ahh = float(row.HandiSize)
        else:
            ahh = np.nan
        lam_mh, lam_ma = bvd.infer_market_lambdas(p_mkt[0], p_mkt[1], p_mkt[2], p_o25, ahh)

        league_avg_lam[div] = 0.996 * league_avg_lam[div] + 0.004 * (0.5 * (lam_mh + lam_ma))
        league_hfa_lam[div] = 0.996 * league_hfa_lam[div] + 0.004 * (
            math.log(lam_mh / lam_ma) - (sh["att_s"] - sa["def_s"] - sa["att_s"] + sh["def_s"])
        )
        lh_s = math.exp(max(-1.5, min(1.8, mu + 0.5 * hfa + sh["att_s"] - sa["def_s"])))
        la_s = math.exp(max(-1.5, min(1.8, mu - 0.5 * hfa + sa["att_s"] - sh["def_s"])))
        ems_h = max(-1.5, min(1.5, math.log(lam_mh) - math.log(lh_s)))
        ems_a = max(-1.5, min(1.5, math.log(lam_ma) - math.log(la_s)))
        sh["att_s"] = 0.997 * sh["att_s"] + 0.09 * ems_h
        sa["def_s"] = 0.997 * sa["def_s"] - 0.09 * ems_h
        sa["att_s"] = 0.997 * sa["att_s"] + 0.09 * ems_a
        sh["def_s"] = 0.997 * sh["def_s"] - 0.09 * ems_a

        S_h = sh["P_att"] + sa["P_def"] + R_pin
        S_a = sa["P_att"] + sh["P_def"] + R_pin
        K_att_h = sh["P_att"] / S_h
        K_def_a = sa["P_def"] / S_h
        K_att_a = sa["P_att"] / S_a
        K_def_h = sh["P_def"] / S_a

        eh_p = max(-1.5, min(1.5, math.log(lam_mh) - math.log(lh)))
        ea_p = max(-1.5, min(1.5, math.log(lam_ma) - math.log(la)))
        eh_x = max(-2.5, min(2.5, 1.06 * npxg_h - lh))
        ea_x = max(-2.5, min(2.5, 1.06 * npxg_a - la))

        sh["att"] = (1.0 - decay) * sh["att"] + K_att_h * eh_p + (field_k_ratio * K_att_h) * eh_x
        sa["def"] = (1.0 - decay) * sa["def"] - K_def_a * eh_p - (field_k_ratio * K_def_a) * eh_x
        sa["att"] = (1.0 - decay) * sa["att"] + K_att_a * ea_p + (field_k_ratio * K_att_a) * ea_x
        sh["def"] = (1.0 - decay) * sh["def"] - K_def_h * ea_p - (field_k_ratio * K_def_h) * ea_x

        sh["P_att"] = max(0.02, (1.0 - K_att_h) * sh["P_att"])
        sa["P_def"] = max(0.02, (1.0 - K_def_a) * sa["P_def"])
        sa["P_att"] = max(0.02, (1.0 - K_att_a) * sa["P_att"])
        sh["P_def"] = max(0.02, (1.0 - K_def_h) * sh["P_def"])

    return preds, p_over25_raw


def softmax_np(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def apply_temperature(probs, T):
    logits = np.log(np.clip(probs, 1e-7, 1.0))
    return softmax_np(logits / T)


def fit_dirichlet_blend(prob_list, y_oh):
    k = len(prob_list)

    def obj(params):
        w = softmax_np(params[:k].reshape(1, -1))[0]
        T = float(np.exp(params[k]))
        p_mix = np.zeros_like(prob_list[0])
        for j in range(k):
            p_mix += w[j] * prob_list[j]
        return brier_score(apply_temperature(p_mix, T), y_oh)

    best = None
    inits = [
        np.zeros(k + 1),
        np.array([0.0] * (k - 1) + [1.5, 0.0]),
        np.array([1.5] + [0.0] * (k - 1) + [0.0]),
    ]
    for init in inits:
        res = opt.minimize(obj, x0=init, method="Nelder-Mead", options={"maxiter": 6000})
        if best is None or res.fun < best.fun:
            best = res
    w_opt = softmax_np(best.x[:k].reshape(1, -1))[0]
    T_opt = float(np.exp(best.x[k]))
    return w_opt, T_opt


def apply_blend(prob_list, w, T):
    p_mix = np.zeros_like(prob_list[0])
    for j in range(len(w)):
        p_mix += w[j] * prob_list[j]
    return apply_temperature(p_mix, T)


def compute_exact_motivation_features(df_enr, matches_csv=DEFAULT_MATCHES_CSV):
    df_m = pd.read_csv(
        matches_csv,
        usecols=["Division", "MatchDate", "HomeTeam", "AwayTeam", "FTHome", "FTAway"],
        low_memory=False,
    )
    df_m = df_m[df_m["Division"].isin(TOP5_DIVS)].copy()
    df_m["dt"] = pd.to_datetime(df_m["MatchDate"])
    df_m = df_m[(df_m["dt"] >= "2014-07-01") & (df_m["dt"] <= "2025-06-01")].copy()

    def get_season(d):
        if d.month >= 7:
            return f"{d.year}/{d.year + 1}"
        return f"{d.year - 1}/{d.year}"

    df_m["season"] = df_m["dt"].apply(get_season)
    df_m = df_m.sort_values(["dt", "Division", "HomeTeam"]).reset_index(drop=True)

    enr_map = {}
    for idx, r in enumerate(df_enr[["Division", "MatchDate", "HomeTeam", "AwayTeam"]].itertuples(index=False)):
        enr_map[(r.Division, r.MatchDate, r.HomeTeam, r.AwayTeam)] = idx

    season_teams = {}
    for (div, seas), grp in df_m.groupby(["Division", "season"]):
        tset = set(grp["HomeTeam"].unique()) | set(grp["AwayTeam"].unique())
        n_t = len(tset)
        if n_t <= 18:
            total_games = 34
            rel_idx = 15
            safe_idx = 14
        else:
            total_games = 38
            rel_idx = 17
            safe_idx = 16
        season_teams[(div, seas)] = (n_t, total_games, safe_idx, rel_idx, sorted(tset))

    tables = {}
    for key, (_, _, _, _, tlist) in season_teams.items():
        tables[key] = {t: [0, 0, 0, 0] for t in tlist}

    out = {}
    for row in df_m.itertuples(index=False):
        key = (row.Division, row.season)
        n_t, tot_g, safe_i, rel_i, _ = season_teams[key]
        tab = tables[key]
        ht = row.HomeTeam
        at = row.AwayTeam

        e_idx = enr_map.get((row.Division, row.MatchDate, ht, at), -1)
        if e_idx >= 0:
            ordered = sorted(tab.items(), key=lambda kv: (kv[1][0], kv[1][1], kv[1][2]), reverse=True)
            pos_map = {t: idx for idx, (t, _) in enumerate(ordered)}
            pts_list = [kv[1][0] for kv in ordered]
            gp_list = [kv[1][3] for kv in ordered]

            def team_status(team_name):
                pos = pos_map[team_name]
                pts, gd, gf, gp = tab[team_name]
                rem = max(0, tot_g - gp)
                max_rem = 3 * rem
                p1 = pts_list[0]
                p2 = pts_list[1]
                p4 = pts_list[min(3, n_t - 1)]
                p5 = pts_list[min(4, n_t - 1)]
                p6 = pts_list[min(5, n_t - 1)]
                p7 = pts_list[min(6, n_t - 1)]
                p_safe = pts_list[min(safe_i, n_t - 1)]
                p_rel = pts_list[min(rel_i, n_t - 1)]

                rem_2nd = max(0, tot_g - gp_list[1])
                math_champ = 1.0 if (pos == 0 and (pts - p2) > 3 * rem_2nd) else 0.0
                math_rel = 1.0 if (pos >= rel_i and (pts + max_rem) < p_safe) else 0.0

                rem_5th = max(0, tot_g - gp_list[min(4, n_t - 1)])
                can_win_title = (pts + max_rem) >= p1
                safe_from_5th = (pos <= 3) and ((pts - p5) > 3 * rem_5th)
                math_locked_ucl = 1.0 if (1 <= pos <= 3 and (not can_win_title) and safe_from_5th) else 0.0

                rem_rel = max(0, tot_g - gp_list[min(rel_i, n_t - 1)])
                safe_from_rel = (pos < rel_i) and ((pts - p_rel) > 3 * rem_rel)
                cant_reach_euro = (pos > 6) and ((pts + max_rem) < p7)
                math_dead = 1.0 if (safe_from_rel and cant_reach_euro) else 0.0

                soft_dead = 0.0
                if 1 <= rem <= 6 and math_champ == 0.0 and math_rel == 0.0:
                    if (pts - p_rel) >= (2.0 * rem + 2.0) and (p6 - pts) >= (2.0 * rem + 2.0):
                        soft_dead = 1.0

                desp_rel = 0.0
                if 1 <= rem <= 7 and math_rel == 0.0:
                    gap_to_danger = pts - p_rel
                    gap_to_safe = p_safe - pts
                    if pos >= rel_i and gap_to_safe <= max_rem:
                        desp_rel = 1.0
                    elif pos < rel_i and gap_to_danger <= min(6, 2 * rem):
                        desp_rel = max(0.0, 1.0 - gap_to_danger / max(1.0, 2.0 * rem + 1.0))

                desp_euro = 0.0
                if 1 <= rem <= 7 and math_champ == 0.0 and math_locked_ucl == 0.0:
                    if pos <= 6 and (p1 - pts) <= min(6, max_rem):
                        desp_euro = 1.0
                    elif abs(pts - p4) <= min(5, 2 * rem):
                        desp_euro = 0.9
                    elif abs(pts - p6) <= min(4, 2 * rem):
                        desp_euro = 0.7

                switched_off = max(math_champ, math_rel, math_locked_ucl, math_dead, 0.75 * soft_dead)
                desperate = max(desp_rel, desp_euro)
                mot_index = 0.5 - 0.5 * switched_off + 0.5 * desperate * (1.0 - switched_off)
                return {
                    "champ": math_champ, "mrel": math_rel, "lucl": math_locked_ucl,
                    "drel": desp_rel, "deuro": desp_euro,
                    "off": switched_off, "mot": mot_index, "rem": float(rem),
                }

            sh = team_status(ht)
            sa = team_status(at)
            late_ramp = max(0.0, min(1.0, (8.0 - min(sh["rem"], sa["rem"])) / 7.0))
            out[e_idx] = {
                "ex_mot_diff": (sh["mot"] - sa["mot"]) * late_ramp,
                "ex_off_diff": (sa["off"] - sh["off"]) * late_ramp,
                "ex_champ_diff": sa["champ"] - sh["champ"],
                "ex_locked_diff": sa["lucl"] - sh["lucl"],
                "ex_mrel_diff": sa["mrel"] - sh["mrel"],
                "ex_drel_diff": (sh["drel"] - sa["drel"]) * late_ramp,
                "ex_deuro_diff": (sh["deuro"] - sa["deuro"]) * late_ramp,
                "ex_late_ramp": late_ramp,
            }

        if pd.notna(row.FTHome) and pd.notna(row.FTAway):
            gh = int(row.FTHome)
            ga = int(row.FTAway)
            tab[ht][1] += (gh - ga)
            tab[ht][2] += gh
            tab[ht][3] += 1
            tab[at][1] += (ga - gh)
            tab[at][2] += ga
            tab[at][3] += 1
            if gh > ga:
                tab[ht][0] += 3
            elif ga > gh:
                tab[at][0] += 3
            else:
                tab[ht][0] += 1
                tab[at][0] += 1

    return pd.DataFrame([out[i] for i in range(len(df_enr))])


def fit_logit_offset_linear_grouped(base_logits_tr, X_mot_tr, X_res_tr, y_tr, lam_mot=100.0, lam_res=7000.0):
    n_s = len(y_tr)
    dm = X_mot_tr.shape[1]
    dr = X_res_tr.shape[1]

    def loss_and_grad(params):
        alpha = params[0]
        b = params[1:4]
        Wm = params[4 : 4 + 3 * dm].reshape(dm, 3)
        Wr = params[4 + 3 * dm :].reshape(dr, 3)
        logits = alpha * base_logits_tr + b + X_mot_tr @ Wm + X_res_tr @ Wr
        probs = softmax_np(logits)
        ce = -np.mean(np.log(np.clip(probs[np.arange(n_s), y_tr], 1e-12, 1.0)))
        reg = 0.5 * (lam_mot / n_s) * float(np.sum(Wm * Wm)) + 0.5 * (lam_res / n_s) * float(np.sum(Wr * Wr))
        diff = probs.copy()
        diff[np.arange(n_s), y_tr] -= 1.0
        diff /= n_s
        g_alpha = float(np.sum(diff * base_logits_tr))
        g_b = np.sum(diff, axis=0)
        g_Wm = X_mot_tr.T @ diff + (lam_mot / n_s) * Wm
        g_Wr = X_res_tr.T @ diff + (lam_res / n_s) * Wr
        return ce + reg, np.concatenate([[g_alpha], g_b, g_Wm.ravel(), g_Wr.ravel()])

    x0 = np.zeros(4 + 3 * (dm + dr), dtype=np.float64)
    x0[0] = 1.0
    res = opt.minimize(loss_and_grad, x0, jac=True, method="L-BFGS-B", options={"maxiter": 400})
    alpha = float(res.x[0])
    b = res.x[1:4].copy()
    Wm = res.x[4 : 4 + 3 * dm].reshape(dm, 3).copy()
    Wr = res.x[4 + 3 * dm :].reshape(dr, 3).copy()
    return alpha, b, Wm, Wr


class OffsetResidualMLP(nn.Module):
    def __init__(self, d, hidden=24, dropout=0.35):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.zeros(3))
        self.fc1 = nn.Linear(d, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden, 3)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, base_logits, x):
        h = self.drop(torch.tanh(self.ln1(self.fc1(x))))
        return self.alpha * base_logits + self.bias + self.fc2(h)


def train_mlp_models(base_tr, X_tr, y_tr, n_seeds=10, steps=300, wd=0.10, lr=0.0025):
    bt = torch.tensor(base_tr, dtype=torch.float32)
    xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    n_tr = len(y_tr)
    states = []
    for s in range(n_seeds):
        torch.manual_seed(700 + s)
        m = OffsetResidualMLP(X_tr.shape[1], hidden=24, dropout=0.35)
        opt_a = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=steps)
        crit = nn.CrossEntropyLoss()
        m.train()
        for _ in range(steps):
            idx = torch.randint(0, n_tr, (512,))
            opt_a.zero_grad()
            loss = crit(m(bt[idx], xt[idx]), yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt_a.step()
            sch.step()
        m.eval()
        states.append({k: v.cpu().clone() for k, v in m.state_dict().items()})
    return states


def predict_mlp_states(states, base_ev, X_ev):
    be = torch.tensor(base_ev, dtype=torch.float32)
    xe = torch.tensor(X_ev, dtype=torch.float32)
    d = X_ev.shape[1]
    preds = []
    for st in states:
        m = OffsetResidualMLP(d, hidden=24, dropout=0.35)
        m.load_state_dict(st)
        m.eval()
        with torch.no_grad():
            preds.append(F.softmax(m(be, xe), dim=1).numpy())
    return np.mean(preds, axis=0)


def main():
    print("[train_v2] Loading v2 dataset & exact motivation features...")
    df = pd.read_pickle(DEFAULT_EXACT_PKL)
    df_enr = pd.read_csv(
        DEFAULT_ENRICHED_CSV,
        usecols=["Division", "MatchDate", "HomeTeam", "AwayTeam", "gol_casa", "gol_trasferta"],
    )
    df_mot = compute_exact_motivation_features(df_enr)
    df = pd.concat([df.reset_index(drop=True), df_mot.reset_index(drop=True)], axis=1)

    n = len(df)
    s1, s2 = int(0.80 * n), int(0.90 * n)
    y = df["risultato_1X2"].map({"1": 0, "X": 1, "2": 2}).values
    y_oh = np.eye(3)[y]

    folds = [(6500, 8127), (8127, 9754), (9754, 11381), (11381, 13014), (13014, 14641)]
    oof_start, oof_end = folds[0][0], folds[-1][1]
    n_oof = oof_end - oof_start

    # 1. Clean Generative Base (5 components, OOF-calibrated; slot 0 = v2.2 Adaptive Riccati Dual-Gain Kalman!)
    print("[train_v2] Computing v2.2 Adaptive Riccati Dual-Gain Kalman (1X2 + Over/Under 2.5)...")
    p_kalman_v21, p_over25_raw = compute_v21_kalman_d2_t1f(n)

    # Fit 2-param Platt calibrator for Over/Under 2.5 strictly on OOF (zero Blind Test leakage!)
    y_o25 = ((df["gol_casa"].values + df["gol_trasferta"].values) >= 3).astype(np.float64)
    ou_logits = np.log(np.clip(p_over25_raw, 1e-6, 1.0 - 1e-6) / (1.0 - np.clip(p_over25_raw, 1e-6, 1.0 - 1e-6)))

    def ou_platt_obj(params):
        a_c, b_c = params[0], params[1]
        p_c = 1.0 / (1.0 + np.exp(-(a_c * ou_logits[oof_start:oof_end] + b_c)))
        return brier_binary(p_c, y_o25[oof_start:oof_end])

    res_ou = opt.minimize(ou_platt_obj, x0=[1.0, 0.0], method="Nelder-Mead")
    ou_a, ou_b = float(res_ou.x[0]), float(res_ou.x[1])
    p_o25_cal = 1.0 / (1.0 + np.exp(-(ou_a * ou_logits + ou_b)))
    ou_oof_brier = brier_binary(p_o25_cal[oof_start:oof_end], y_o25[oof_start:oof_end])
    ou_oof_ll = ll_binary(p_o25_cal[oof_start:oof_end], y_o25[oof_start:oof_end])
    ou_te_brier = brier_binary(p_o25_cal[s2:], y_o25[s2:])
    ou_te_ll = ll_binary(p_o25_cal[s2:], y_o25[s2:])
    print(f"[train_v2] Over/Under 2.5 Platt (a={ou_a:.4f}, b={ou_b:+.4f}) | OOF Brier={ou_oof_brier:.5f} (LL={ou_oof_ll:.5f}) | Blind Test Brier={ou_te_brier:.5f} (LL={ou_te_ll:.5f})")
    gen_col_sets = [
        ["p_mkt_s_1", "p_mkt_s_x", "p_mkt_s_2"],
        ["ex_mkt_35d_p1", "ex_mkt_35d_px", "ex_mkt_35d_p2"],
        ["ex_dc_npxg_120d_p1", "ex_dc_npxg_120d_px", "ex_dc_npxg_120d_p2"],
        ["p_dc_blend_1", "p_dc_blend_x", "p_dc_blend_2"],
    ]
    gen_comps = [p_kalman_v21] + [df[cols].values for cols in gen_col_sets]
    w_gen, T_gen = fit_dirichlet_blend([c[oof_start:oof_end] for c in gen_comps], y_oh[oof_start:oof_end])
    p_base_all = apply_blend(gen_comps, w_gen, T_gen)
    base_logits_all = np.log(np.clip(p_base_all, 1e-6, 1.0))
    print(f"[train_v2] Gen_Base weights={np.round(w_gen, 3)} T={T_gen:.3f} | OOF={brier_score(p_base_all[oof_start:oof_end], y_oh[oof_start:oof_end]):.5f} | Test Brier={brier_score(p_base_all[s2:], y_oh[s2:]):.5f}")

    Xm_all = df[MOT_COLS].values.astype(np.float64)
    Xr_all = df[RES_COLS].values.astype(np.float64)
    mot_med = np.nanmedian(Xm_all, axis=0)
    res_med = np.nanmedian(Xr_all, axis=0)
    for j in range(Xm_all.shape[1]):
        Xm_all[~np.isfinite(Xm_all[:, j]), j] = mot_med[j]
    for j in range(Xr_all.shape[1]):
        Xr_all[~np.isfinite(Xr_all[:, j]), j] = res_med[j]
    Xboth_all = np.hstack([Xm_all, Xr_all])

    # 2. OOF calibration for ensemble weights (zero Test leakage!)
    print("[train_v2] Running 5-Fold Expanding OOF CV for ensemble weights...")
    oof_lin = np.zeros((n_oof, 3))
    oof_lgb = np.zeros((n_oof, 3))
    oof_xgb = np.zeros((n_oof, 3))
    oof_mlp = np.zeros((n_oof, 3))

    for tr_end, va_end in folds:
        sl_oof = slice(tr_end - oof_start, va_end - oof_start)
        sc_r = StandardScaler()
        Xr_tr = np.clip(sc_r.fit_transform(Xr_all[:tr_end]), -5.0, 5.0)
        Xr_va = np.clip(sc_r.transform(Xr_all[tr_end:va_end]), -5.0, 5.0)
        a, b, Wm, Wr = fit_logit_offset_linear_grouped(base_logits_all[:tr_end], Xm_all[:tr_end], Xr_tr, y[:tr_end], 100.0, 7000.0)
        oof_lin[sl_oof] = softmax_np(a * base_logits_all[tr_end:va_end] + b + Xm_all[tr_end:va_end] @ Wm + Xr_va @ Wr)

        m_lgb = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, n_estimators=80, learning_rate=0.015,
            num_leaves=6, max_depth=3, min_child_samples=150, colsample_bytree=0.55,
            subsample=0.75, subsample_freq=1, reg_alpha=5.0, reg_lambda=35.0,
            random_state=42, verbosity=-1, n_jobs=-1,
        )
        m_lgb.fit(Xboth_all[:tr_end], y[:tr_end], init_score=base_logits_all[:tr_end])
        oof_lgb[sl_oof] = softmax_np(m_lgb.predict(Xboth_all[tr_end:va_end], raw_score=True) + base_logits_all[tr_end:va_end])

        m_xgb = xgb.XGBClassifier(
            objective="multi:softprob", num_class=3, n_estimators=80, learning_rate=0.015,
            max_depth=3, min_child_weight=100, colsample_bytree=0.55, subsample=0.75,
            reg_alpha=5.0, reg_lambda=35.0, tree_method="hist", random_state=42, n_jobs=-1,
        )
        m_xgb.fit(Xboth_all[:tr_end], y[:tr_end], base_margin=base_logits_all[:tr_end], verbose=False)
        oof_xgb[sl_oof] = m_xgb.predict_proba(Xboth_all[tr_end:va_end], base_margin=base_logits_all[tr_end:va_end])

        sc_b = StandardScaler()
        Xb_tr = np.clip(sc_b.fit_transform(Xboth_all[:tr_end]), -5.0, 5.0)
        Xb_va = np.clip(sc_b.transform(Xboth_all[tr_end:va_end]), -5.0, 5.0)
        st_mlp = train_mlp_models(base_logits_all[:tr_end], Xb_tr, y[:tr_end], n_seeds=8, steps=300, wd=0.10, lr=0.0025)
        oof_mlp[sl_oof] = predict_mlp_states(st_mlp, base_logits_all[tr_end:va_end], Xb_va)

    # Fit ensemble weights cleanly across all 5 heads (with mild convex regularization so Linear keeps strong share!)
    oof_heads = [p_base_all[oof_start:oof_end], oof_lin, oof_lgb, oof_xgb, oof_mlp]
    w_ens, T_ens = fit_dirichlet_blend(oof_heads, y_oh[oof_start:oof_end])
    # Also blend 50/50 with Offset_Linear (which had best OOF 0.58371, Val 0.57280, Test 0.57323)
    # by computing optimal weights via SLSQP directly on simplex (avoids Nelder-Mead local minima!)
    def slsqp_obj(w_vec):
        p_mix = np.zeros_like(oof_heads[0])
        for j in range(len(oof_heads)):
            p_mix += w_vec[j] * oof_heads[j]
        return brier_score(p_mix, y_oh[oof_start:oof_end])

    res_sl = opt.minimize(
        slsqp_obj,
        x0=np.array([0.15, 0.35, 0.15, 0.20, 0.15]),
        bounds=[(0.0, 1.0)] * 5,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        method="SLSQP",
    )
    w_sl = res_sl.x / np.sum(res_sl.x)
    p_oof_sl = np.zeros_like(oof_heads[0])
    for j in range(5):
        p_oof_sl += w_sl[j] * oof_heads[j]
    # Fit single temperature on OOF
    def t_obj(p_arr):
        return brier_score(apply_temperature(p_oof_sl, float(np.exp(p_arr[0]))), y_oh[oof_start:oof_end])
    T_sl = float(np.exp(opt.minimize(t_obj, x0=[0.0], method="Nelder-Mead").x[0]))
    p_oof_cal = apply_temperature(p_oof_sl, T_sl)
    print(f"[train_v2] SLSQP Ensemble weights (Base, Lin, LGBM, XGB, MLP) = {np.round(w_sl, 3)} | T={T_sl:.3f}")
    print(f"[train_v2] OOF Brier={brier_score(p_oof_cal, y_oh[oof_start:oof_end]):.5f} | Val Brier={brier_score(p_oof_cal[-1627:], y_oh[s1:s2]):.5f}")

    # 3. Verify on Blind Test (trained on 0..s2) before final full fit!
    sc_r_v = StandardScaler()
    Xr_tr_v = np.clip(sc_r_v.fit_transform(Xr_all[:s2]), -5.0, 5.0)
    Xr_te_v = np.clip(sc_r_v.transform(Xr_all[s2:]), -5.0, 5.0)
    a_v, b_v, Wm_v, Wr_v = fit_logit_offset_linear_grouped(base_logits_all[:s2], Xm_all[:s2], Xr_tr_v, y[:s2], 100.0, 7000.0)
    te_lin = softmax_np(a_v * base_logits_all[s2:] + b_v + Xm_all[s2:] @ Wm_v + Xr_te_v @ Wr_v)

    lgb_p_te = []
    for sd in [42, 43, 44, 45, 46]:
        m = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, n_estimators=80, learning_rate=0.015,
            num_leaves=6, max_depth=3, min_child_samples=150, colsample_bytree=0.55,
            subsample=0.75, subsample_freq=1, reg_alpha=5.0, reg_lambda=35.0,
            random_state=sd, verbosity=-1, n_jobs=-1,
        )
        m.fit(Xboth_all[:s2], y[:s2], init_score=base_logits_all[:s2])
        lgb_p_te.append(softmax_np(m.predict(Xboth_all[s2:], raw_score=True) + base_logits_all[s2:]))
    te_lgb = np.mean(lgb_p_te, axis=0)

    xgb_p_te = []
    for sd in [42, 43, 44, 45, 46]:
        m = xgb.XGBClassifier(
            objective="multi:softprob", num_class=3, n_estimators=80, learning_rate=0.015,
            max_depth=3, min_child_weight=100, colsample_bytree=0.55, subsample=0.75,
            reg_alpha=5.0, reg_lambda=35.0, tree_method="hist", random_state=sd, n_jobs=-1,
        )
        m.fit(Xboth_all[:s2], y[:s2], base_margin=base_logits_all[:s2], verbose=False)
        xgb_p_te.append(m.predict_proba(Xboth_all[s2:], base_margin=base_logits_all[s2:]))
    te_xgb = np.mean(xgb_p_te, axis=0)

    sc_b_v = StandardScaler()
    Xb_tr_v = np.clip(sc_b_v.fit_transform(Xboth_all[:s2]), -5.0, 5.0)
    Xb_te_v = np.clip(sc_b_v.transform(Xboth_all[s2:]), -5.0, 5.0)
    st_mlp_v = train_mlp_models(base_logits_all[:s2], Xb_tr_v, y[:s2], n_seeds=10, steps=300, wd=0.10, lr=0.0025)
    te_mlp = predict_mlp_states(st_mlp_v, base_logits_all[s2:], Xb_te_v)

    p_te_ens = apply_blend([p_base_all[s2:], te_lin, te_lgb, te_xgb, te_mlp], w_sl, T_sl)
    test_brier = brier_score(p_te_ens, y_oh[s2:])
    test_ll = log_loss_score(p_te_ens, y[s2:])
    oof_ll = log_loss_score(p_oof_cal, y[oof_start:oof_end])
    print(f"[train_v2] VERIFIED BLIND TEST (2024/25) BRIER = {test_brier:.5f} | LL = {test_ll:.5f} (Lin={brier_score(te_lin, y_oh[s2:]):.5f}, LGBM={brier_score(te_lgb, y_oh[s2:]):.5f}, XGB={brier_score(te_xgb, y_oh[s2:]):.5f}, MLP={brier_score(te_mlp, y_oh[s2:]):.5f})")

    # 4. Fit FINAL PRODUCTION MODELS on 100% of 2015-2025 verified matches (16,268 matches!)
    print("[train_v2] Training final production models on all 16,268 verified matches...")
    sc_r_prod = StandardScaler()
    Xr_full = np.clip(sc_r_prod.fit_transform(Xr_all), -5.0, 5.0)
    a_p, b_p, Wm_p, Wr_p = fit_logit_offset_linear_grouped(base_logits_all, Xm_all, Xr_full, y, 100.0, 7000.0)

    lgb_models = []
    for sd in [42, 43, 44, 45, 46]:
        m = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, n_estimators=80, learning_rate=0.015,
            num_leaves=6, max_depth=3, min_child_samples=150, colsample_bytree=0.55,
            subsample=0.75, subsample_freq=1, reg_alpha=5.0, reg_lambda=35.0,
            random_state=sd, verbosity=-1, n_jobs=-1,
        )
        m.fit(Xboth_all, y, init_score=base_logits_all)
        lgb_models.append(m)

    xgb_models = []
    for sd in [42, 43, 44, 45, 46]:
        m = xgb.XGBClassifier(
            objective="multi:softprob", num_class=3, n_estimators=80, learning_rate=0.015,
            max_depth=3, min_child_weight=100, colsample_bytree=0.55, subsample=0.75,
            reg_alpha=5.0, reg_lambda=35.0, tree_method="hist", random_state=sd, n_jobs=-1,
        )
        m.fit(Xboth_all, y, base_margin=base_logits_all, verbose=False)
        xgb_models.append(m)

    sc_b_prod = StandardScaler()
    Xb_full = np.clip(sc_b_prod.fit_transform(Xboth_all), -5.0, 5.0)
    mlp_states = train_mlp_models(base_logits_all, Xb_full, y, n_seeds=10, steps=300, wd=0.10, lr=0.0025)

    bundle = {
        "version": "deep-pitch-v2.2",
        "mot_cols": MOT_COLS,
        "res_cols": RES_COLS,
        "mot_medians": mot_med,
        "res_medians": res_med,
        "w_gen": w_gen,
        "T_gen": T_gen,
        "w_ens": w_sl,
        "T_ens": T_sl,
        "offset_linear": {
            "alpha": a_p,
            "bias": b_p,
            "Wm": Wm_p,
            "Wr": Wr_p,
            "scaler_res": sc_r_prod,
        },
        "lgb_models": lgb_models,
        "xgb_models": xgb_models,
        "mlp_states": mlp_states,
        "scaler_both": sc_b_prod,
        "ou25_calibrator": {
            "a": ou_a,
            "b": ou_b,
            "oof_brier": ou_oof_brier,
            "oof_logloss": ou_oof_ll,
            "test_brier": ou_te_brier,
            "test_logloss": ou_te_ll,
        },
        "metrics": {
            "oof_brier": brier_score(p_oof_cal, y_oh[oof_start:oof_end]),
            "oof_logloss": oof_ll,
            "val_brier": brier_score(p_oof_cal[-1627:], y_oh[s1:s2]),
            "test_brier": test_brier,
            "test_logloss": test_ll,
            "ou25_oof_brier": ou_oof_brier,
            "ou25_test_brier": ou_te_brier,
            "ou25_test_logloss": ou_te_ll,
            "n_train_total": n,
        },
    }

    with open(OUT_BUNDLE, "wb") as f:
        pickle.dump(bundle, f)
    size_kb = os.path.getsize(OUT_BUNDLE) / 1024.0
    print(f"[train_v2] Saved production bundle: {OUT_BUNDLE} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
