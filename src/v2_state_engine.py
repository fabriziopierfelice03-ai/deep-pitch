"""
src/v2_state_engine.py
Zero-leakage live state & feature builder for Deep-Pitch v2.

Automatically syncs & caches football-data.co.uk Top-5 CSVs (prior 2 seasons + current season)
and replays all completed matches chronologically up to today to produce the exact
live pre-match states used by Deep-Pitch v2:
  1. Online & Exact Rolling 35d Past-Market Dixon-Coles (Pinnacle Closing/Opening + O2.5 + AHh)
  2. Exact Rolling 120d Weighted Poisson-MLE Dixon-Coles on Opta/FotMob npxG + xGOT
  3. Multi-scale EWMAs (fast/med) on npxG, xGOT, OpenPlay xG, Big Chances, Opp-Adjusted npxG
  4. Exact Mathematical End-of-Season Motivation features from official league standings
  5. Schedule / Rest / ClubElo integration

100% free sources, zero manual steps, automatic 12-hour local cache in data/fd_cache/.
"""

import os
import re
import math
import time
import unicodedata
import urllib.request
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FD_CACHE_DIR = os.path.join(BASE_DIR, "data", "fd_cache")
os.makedirs(FD_CACHE_DIR, exist_ok=True)

TOP5_DIVS = ["E0", "I1", "SP1", "D1", "F1"]
CACHE_TTL_SECONDS = 12 * 3600

# Canonical team name normalization mapping FotMob / BetExplorer / football-data names together
TEAM_CANON_ALIASES = {
    # England
    "manchestercity": "mancity",
    "manchesterunited": "manunited",
    "newcastleunited": "newcastle",
    "nottinghamforest": "nottmforest",
    "nottmforest": "nottmforest",
    "tottenhamhotspur": "tottenham",
    "spurs": "tottenham",
    "wolverhamptonwanderers": "wolves",
    "wolverhampton": "wolves",
    "brightonandhovealbion": "brighton",
    "brightonhovealbion": "brighton",
    "westhamunited": "westham",
    "afcbournemouth": "bournemouth",
    "leedsunited": "leeds",
    "leicestercity": "leicester",
    "ipswichtown": "ipswich",
    "lutontown": "luton",
    "westbromwichalbion": "westbrom",
    "sheffieldunited": "sheffieldunited",
    # Italy
    "acmilan": "milan",
    "internazionale": "inter",
    "intermilan": "inter",
    "fcinternazionalemilano": "inter",
    "hellasverona": "verona",
    "asroma": "roma",
    "sslazio": "lazio",
    "sscnapoli": "napoli",
    "acffiorentina": "fiorentina",
    "parmacalcio1913": "parma",
    "como1907": "como",
    "calciocomo": "como",
    "acmonza": "monza",
    # Spain
    "athleticclub": "athbilbao",
    "athleticbilbao": "athbilbao",
    "atleticomadrid": "athmadrid",
    "atleticodemadrid": "athmadrid",
    "celtavigo": "celta",
    "celtadevigo": "celta",
    "deportivoalaves": "alaves",
    "realbetis": "betis",
    "realbetisbalompie": "betis",
    "realsociedad": "sociedad",
    "rayovallecano": "vallecano",
    "rcdespanyol": "espanol",
    "espanyol": "espanol",
    "realvalladolid": "valladolid",
    "realoviedo": "oviedo",
    "rcdmallorca": "mallorca",
    "fcbarcelona": "barcelona",
    "realmadridcf": "realmadrid",
    # Germany
    "bayernmunich": "bayernmunich",
    "bayernmunchen": "bayernmunich",
    "fcbayernmunchen": "bayernmunich",
    "bayerleverkusen": "leverkusen",
    "bayer04leverkusen": "leverkusen",
    "borussiadortmund": "dortmund",
    "borussiamonchengladbach": "mgladbach",
    "borussiamgladbach": "mgladbach",
    "mgladbach": "mgladbach",
    "eintrachtfrankfurt": "einfrankfurt",
    "einfrankfurt": "einfrankfurt",
    "fckoln": "fckoln",
    "1fckoln": "fckoln",
    "cologne": "fckoln",
    "mainz05": "mainz",
    "1fsvmainz05": "mainz",
    "vfbstuttgart": "stuttgart",
    "scfreiburg": "freiburg",
    "tsghoffenheim": "hoffenheim",
    "vflwolfsburg": "wolfsburg",
    "vflbochum": "bochum",
    "fcaugsburg": "augsburg",
    "1fcunionberlin": "unionberlin",
    "1fcheidenheim": "heidenheim",
    "fcheidenheim": "heidenheim",
    "fcstpauli": "stpauli",
    "holsteinkiel": "holsteinkiel",
    "rasenballsportleipzig": "rbleipzig",
    "svwerderbremen": "werderbremen",
    "hamburgersv": "hamburg",
    # France
    "parissaintgermain": "parissg",
    "psg": "parissg",
    "olympiquedemarseille": "marseille",
    "olympiquelyonnais": "lyon",
    "asmonaco": "monaco",
    "ogcnice": "nice",
    "lilleosc": "lille",
    "staderennais": "rennes",
    "stadebrestois": "brest",
    "stadebrestois29": "brest",
    "stadedereims": "reims",
    "rcstrasbourg": "strasbourg",
    "rcstrasbourgalsace": "strasbourg",
    "rclens": "lens",
    "assaintetienne": "stetienne",
    "saintetienne": "stetienne",
    "ajauxerre": "auxerre",
    "lehavreac": "lehavre",
    "angerssco": "angers",
    "fclorient": "lorient",
    "fcmetz": "metz",
    "fcnantes": "nantes",
    "toulousefc": "toulouse",
    "parisfc": "parisfc",
}


def canon_team(name: str) -> str:
    if not name:
        return ""
    s = "".join(c for c in unicodedata.normalize("NFD", str(name).lower()) if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9]", "", s)
    return TEAM_CANON_ALIASES.get(s, s)


def devig_shin_single(q1: float, qx: float, q2: float) -> Optional[np.ndarray]:
    if np.isnan(q1) or np.isnan(qx) or np.isnan(q2):
        return None
    if q1 <= 1.0 or qx <= 1.0 or q2 <= 1.0:
        return None
    pi = np.array([1.0 / q1, 1.0 / qx, 1.0 / q2], dtype=np.float64)
    b = float(pi.sum())
    if b <= 1.0:
        return pi / b
    lo = 1e-7
    hi = 1.0 - 1e-7
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        p_est = (np.sqrt(mid * mid + 4.0 * (1.0 - mid) * (pi * pi / b)) - mid) / (2.0 * (1.0 - mid))
        if float(p_est.sum()) > 1.0:
            lo = mid
        else:
            hi = mid
    return p_est / float(p_est.sum())


def devig_two_way(q_over: float, q_under: float) -> float:
    if np.isnan(q_over) or np.isnan(q_under) or q_over <= 1.0 or q_under <= 1.0:
        return np.nan
    po = 1.0 / q_over
    pu = 1.0 / q_under
    return po / (po + pu)


def poisson_1x2_dc(lam_h: float, lam_a: float, rho: float = -0.065, max_g: int = 9) -> Tuple[float, float, float]:
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
        return 0.45, 0.27, 0.28
    mat /= tot
    return float(np.tril(mat, k=-1).sum()), float(np.trace(mat)), float(np.triu(mat, k=1).sum())


def solve_total_goals_from_over25(p_o25: float) -> float:
    if np.isnan(p_o25) or p_o25 <= 0.02 or p_o25 >= 0.98:
        return 2.65
    lo, hi = 0.5, 6.5
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        est = 1.0 - math.exp(-mid) * (1.0 + mid + 0.5 * mid * mid)
        if est < p_o25:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def infer_market_lambdas(p1: float, px: float, p2: float, p_o25: float, ahh: float) -> Tuple[float, float]:
    L = solve_total_goals_from_over25(p_o25)
    target = p1 - p2
    lo = -(L - 0.30)
    hi = L - 0.30
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        ph, _, pa = poisson_1x2_dc(0.5 * (L + mid), 0.5 * (L - mid), rho=-0.06, max_g=7)
        if (ph - pa) < target:
            lo = mid
        else:
            hi = mid
    d_est = 0.5 * (lo + hi)
    if not np.isnan(ahh) and abs(ahh) <= 3.5:
        d_est = 0.70 * d_est + 0.30 * (-ahh)
        d_est = max(-(L - 0.30), min(L - 0.30, d_est))
    return max(0.20, 0.5 * (L + d_est)), max(0.20, 0.5 * (L - d_est))


def _solve_weighted_ridge_att_def(h_ids, a_ids, y_h, y_a, weights, n_teams, prior_att, prior_def, reg=2.2):
    dim = 2 + 2 * n_teams
    ATA = np.zeros((dim, dim), dtype=np.float64)
    ATy = np.zeros(dim, dtype=np.float64)
    ATA[0, 0] += 0.05
    ATy[0] += 0.05 * 0.25
    ATA[1, 1] += 0.5
    ATy[1] += 0.5 * 0.28
    for t in range(n_teams):
        ATA[2 + t, 2 + t] += reg
        ATy[2 + t] += reg * prior_att[t]
        ATA[2 + n_teams + t, 2 + n_teams + t] += reg
        ATy[2 + n_teams + t] += reg * prior_def[t]
    for k in range(len(h_ids)):
        w = weights[k]
        if w <= 1e-6:
            continue
        hi = h_ids[k]
        ai = a_ids[k]
        idx1 = (0, 1, 2 + hi, 2 + n_teams + ai)
        val1 = (1.0, 0.5, 1.0, -1.0)
        for p in range(4):
            ip = idx1[p]
            vp = val1[p] * w
            ATy[ip] += vp * y_h[k]
            for q in range(4):
                ATA[ip, idx1[q]] += vp * val1[q]
        idx2 = (0, 1, 2 + ai, 2 + n_teams + hi)
        val2 = (1.0, -0.5, 1.0, -1.0)
        for p in range(4):
            ip = idx2[p]
            vp = val2[p] * w
            ATy[ip] += vp * y_a[k]
            for q in range(4):
                ATA[ip, idx2[q]] += vp * val2[q]
    theta = np.linalg.solve(ATA, ATy)
    return float(theta[0]), float(theta[1]), theta[2 : 2 + n_teams], theta[2 + n_teams :]


def _solve_weighted_poisson_att_def(h_ids, a_ids, y_h, y_a, weights, n_teams, prior_att, prior_def, reg=5.0, n_iter=6):
    dim = 2 + 2 * n_teams
    theta = np.zeros(dim, dtype=np.float64)
    theta[0] = 0.24
    theta[1] = 0.28
    theta[2 : 2 + n_teams] = prior_att
    theta[2 + n_teams :] = prior_def
    for _ in range(n_iter):
        H = np.zeros((dim, dim), dtype=np.float64)
        g = np.zeros(dim, dtype=np.float64)
        H[0, 0] += 0.1
        g[0] -= 0.1 * (theta[0] - 0.24)
        H[1, 1] += 1.0
        g[1] -= 1.0 * (theta[1] - 0.28)
        for t in range(n_teams):
            ia = 2 + t
            id_ = 2 + n_teams + t
            H[ia, ia] += reg
            g[ia] -= reg * (theta[ia] - prior_att[t])
            H[id_, id_] += reg
            g[id_] -= reg * (theta[id_] - prior_def[t])
        mu = theta[0]
        hfa = theta[1]
        for k in range(len(h_ids)):
            w = weights[k]
            if w <= 1e-6:
                continue
            hi = h_ids[k]
            ai = a_ids[k]
            ia_h = 2 + hi
            id_a = 2 + n_teams + ai
            lam_h = math.exp(max(-1.8, min(2.0, mu + 0.5 * hfa + theta[ia_h] - theta[id_a])))
            res_h = w * (y_h[k] - lam_h)
            wl_h = w * lam_h
            idx1 = (0, 1, ia_h, id_a)
            val1 = (1.0, 0.5, 1.0, -1.0)
            for p in range(4):
                ip = idx1[p]
                g[ip] += val1[p] * res_h
                vp = val1[p] * wl_h
                for q in range(4):
                    H[ip, idx1[q]] += vp * val1[q]
            ia_a = 2 + ai
            id_h = 2 + n_teams + hi
            lam_a = math.exp(max(-1.8, min(2.0, mu - 0.5 * hfa + theta[ia_a] - theta[id_h])))
            res_a = w * (y_a[k] - lam_a)
            wl_a = w * lam_a
            idx2 = (0, 1, ia_a, id_h)
            val2 = (1.0, -0.5, 1.0, -1.0)
            for p in range(4):
                ip = idx2[p]
                g[ip] += val2[p] * res_a
                vp = val2[p] * wl_a
                for q in range(4):
                    H[ip, idx2[q]] += vp * val2[q]
        step = np.linalg.solve(H, g)
        theta += step
        if float(np.max(np.abs(step))) < 1e-4:
            break
    return float(theta[0]), float(theta[1]), theta[2 : 2 + n_teams], theta[2 + n_teams :]


def _solve_weighted_bt_strength(h_ids, a_ids, y_diff, weights, n_teams, prior_r, reg=2.0):
    dim = 1 + n_teams
    ATA = np.zeros((dim, dim), dtype=np.float64)
    ATy = np.zeros(dim, dtype=np.float64)
    ATA[0, 0] += 0.2
    ATy[0] += 0.2 * 0.35
    for t in range(n_teams):
        ATA[1 + t, 1 + t] += reg
        ATy[1 + t] += reg * prior_r[t]
    for k in range(len(h_ids)):
        w = weights[k]
        if w <= 1e-6:
            continue
        idx = (0, 1 + h_ids[k], 1 + a_ids[k])
        val = (1.0, 1.0, -1.0)
        for p in range(3):
            ip = idx[p]
            vp = val[p] * w
            ATy[ip] += vp * y_diff[k]
            for q in range(3):
                ATA[ip, idx[q]] += vp * val[q]
    theta = np.linalg.solve(ATA, ATy)
    return float(theta[0]), theta[1:]


def get_season_codes_to_sync(now_dt: Optional[datetime] = None) -> List[str]:
    """Returns [prev2, prev1, current] 4-digit football-data season codes (e.g. ['2425', '2526', '2627'])."""
    dt = now_dt if now_dt is not None else datetime.now()
    if dt.month >= 7:
        curr_start = dt.year
    else:
        curr_start = dt.year - 1
    codes = []
    for yr in (curr_start - 2, curr_start - 1, curr_start):
        codes.append(f"{yr % 100:02d}{(yr + 1) % 100:02d}")
    return codes


def sync_football_data_csvs(force_refresh: bool = False) -> List[str]:
    """Downloads & caches football-data.co.uk Top-5 CSVs (12h TTL for current season, permanent for past)."""
    codes = get_season_codes_to_sync()
    curr_code = codes[-1]
    paths = []
    for sc in codes:
        for div in TOP5_DIVS:
            dest = os.path.join(FD_CACHE_DIR, f"{sc}_{div}.csv")
            need_dl = force_refresh or (not os.path.exists(dest)) or (os.path.getsize(dest) < 300)
            if not need_dl and sc == curr_code:
                age = time.time() - os.path.getmtime(dest)
                if age > CACHE_TTL_SECONDS:
                    need_dl = True
            if need_dl:
                url = f"https://www.football-data.co.uk/mmz4281/{sc}/{div}.csv"
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=12) as resp:
                        raw = resp.read()
                    if len(raw) > 200:
                        with open(dest, "wb") as f:
                            f.write(raw)
                except Exception:
                    pass
            if os.path.exists(dest) and os.path.getsize(dest) > 200:
                paths.append(dest)
    return paths


def extract_fotmob_advanced_stats(match_json: Dict[str, Any], side_idx: int) -> Dict[str, float]:
    """
    Extracts npxG, xGOT, OpenPlay xG, SetPiece xG, Big Chances, Shots in Box, SOT
    from a completed FotMob matchDetails JSON (`side_idx = 0` Home, `1` Away).
    """
    opp_idx = 1 - side_idx
    out = {
        "npxg_for": np.nan, "npxg_ag": np.nan,
        "xgot_for": np.nan, "xgot_ag": np.nan,
        "xg_open_for": np.nan, "xg_open_ag": np.nan,
        "bc_for": np.nan, "bc_ag": np.nan,
        "sot_for": np.nan, "sot_ag": np.nan,
    }
    if not isinstance(match_json, dict):
        return out
    pp = match_json.get("pageProps", match_json)
    periods = pp.get("content", {}).get("stats", {}).get("Periods", {}).get("All", {}).get("stats", [])
    key_map = {
        "expected_goals_non_penalty": ("npxg_for", "npxg_ag"),
        "expected_goals_on_target": ("xgot_for", "xgot_ag"),
        "expected_goals_open_play": ("xg_open_for", "xg_open_ag"),
        "big_chance": ("bc_for", "bc_ag"),
        "ShotsOnTarget": ("sot_for", "sot_ag"),
    }
    xg_fallback = [np.nan, np.nan]
    for grp in periods:
        for st in grp.get("stats", []):
            k = st.get("key")
            vals = st.get("stats", [])
            if not isinstance(vals, list) or len(vals) < 2:
                continue
            if k == "expected_goals" and vals[0] is not None and vals[1] is not None:
                try:
                    xg_fallback = [float(vals[0]), float(vals[1])]
                except Exception:
                    pass
            if k in key_map and vals[side_idx] is not None and vals[opp_idx] is not None:
                try:
                    kf, ka = key_map[k]
                    out[kf] = float(vals[side_idx])
                    out[ka] = float(vals[opp_idx])
                except Exception:
                    pass
    if np.isnan(out["npxg_for"]) and not np.isnan(xg_fallback[0]):
        out["npxg_for"] = xg_fallback[side_idx]
        out["npxg_ag"] = xg_fallback[opp_idx]
    return out


def compute_live_motivation_features(
    standings_rows: List[Dict[str, Any]],
    home_team: str,
    away_team: str,
    total_games: int,
    rel_pos_1idx: int,
) -> Dict[str, float]:
    """
    Computes exact mathematical end-of-season motivation features from FotMob standings table.
    """
    zeros = {
        "ex_mot_diff": 0.0, "ex_off_diff": 0.0, "ex_champ_diff": 0.0,
        "ex_locked_diff": 0.0, "ex_mrel_diff": 0.0, "ex_drel_diff": 0.0,
        "ex_deuro_diff": 0.0, "ex_late_ramp": 0.0,
    }
    if not standings_rows or len(standings_rows) < 10:
        return zeros

    # Sort table cleanly by idx/position
    ordered = sorted(standings_rows, key=lambda r: int(r.get("idx", 99)))
    n_t = len(ordered)
    rel_i = min(n_t - 1, max(0, rel_pos_1idx - 1))
    safe_i = max(0, rel_i - 1)

    pts_list = [int(r.get("pts", 0)) for r in ordered]
    gp_list = [int(r.get("played", 0)) for r in ordered]

    def find_idx(tname: str) -> int:
        cn = canon_team(tname)
        for idx, r in enumerate(ordered):
            rn = canon_team(r.get("name", ""))
            if rn == cn:
                return idx
        for idx, r in enumerate(ordered):
            rn = canon_team(r.get("name", ""))
            if (cn and cn in rn) or (rn and rn in cn):
                return idx
        return -1

    hi = find_idx(home_team)
    ai = find_idx(away_team)
    if hi < 0 or ai < 0:
        return zeros

    def team_status(pos: int) -> Dict[str, float]:
        pts = pts_list[pos]
        gp = gp_list[pos]
        rem = max(0, total_games - gp)
        max_rem = 3 * rem
        p1 = pts_list[0]
        p2 = pts_list[1]
        p4 = pts_list[min(3, n_t - 1)]
        p5 = pts_list[min(4, n_t - 1)]
        p6 = pts_list[min(5, n_t - 1)]
        p7 = pts_list[min(6, n_t - 1)]
        p_safe = pts_list[safe_i]
        p_rel = pts_list[rel_i]

        rem_2nd = max(0, total_games - gp_list[1])
        math_champ = 1.0 if (pos == 0 and (pts - p2) > 3 * rem_2nd) else 0.0
        math_rel = 1.0 if (pos >= rel_i and (pts + max_rem) < p_safe) else 0.0

        rem_5th = max(0, total_games - gp_list[min(4, n_t - 1)])
        can_win_title = (pts + max_rem) >= p1
        safe_from_5th = (pos <= 3) and ((pts - p5) > 3 * rem_5th)
        math_locked_ucl = 1.0 if (1 <= pos <= 3 and (not can_win_title) and safe_from_5th) else 0.0

        rem_rel = max(0, total_games - gp_list[rel_i])
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
            "drel": desp_rel, "deuro": desp_euro, "off": switched_off,
            "mot": mot_index, "rem": float(rem),
        }

    sh = team_status(hi)
    sa = team_status(ai)
    late_ramp = max(0.0, min(1.0, (8.0 - min(sh["rem"], sa["rem"])) / 7.0))
    return {
        "ex_mot_diff": (sh["mot"] - sa["mot"]) * late_ramp,
        "ex_off_diff": (sa["off"] - sh["off"]) * late_ramp,
        "ex_champ_diff": sa["champ"] - sh["champ"],
        "ex_locked_diff": sa["lucl"] - sh["lucl"],
        "ex_mrel_diff": sa["mrel"] - sh["mrel"],
        "ex_drel_diff": (sh["drel"] - sa["drel"]) * late_ramp,
        "ex_deuro_diff": (sh["deuro"] - sa["deuro"]) * late_ramp,
        "ex_late_ramp": late_ramp,
    }


class LiveV2StateEngine:
    """
    Maintains live, up-to-today Deep-Pitch v2 team & league states across Top-5 leagues
    from football-data.co.uk historical/current CSVs + live FotMob match caches.
    """

    def __init__(self, force_refresh: bool = False):
        self.league_online: Dict[str, Dict[str, Any]] = {}
        self.league_solvers: Dict[str, Dict[str, Any]] = {}
        self.team_state: Dict[str, Dict[str, float]] = {}
        self._build(force_refresh=force_refresh)

    def _init_team(self, elo_val: float = 1650.0) -> Dict[str, float]:
        elo_n = (elo_val - 1650.0) / 200.0
        return {
            "mkt_elo_f": elo_val, "mkt_elo_s": elo_val,
            "mkt_att_f": 0.18 * elo_n, "mkt_def_f": 0.18 * elo_n,
            "mkt_att_s": 0.18 * elo_n, "mkt_def_s": 0.18 * elo_n,
            "dc_blend_att": 0.15 * elo_n, "dc_blend_def": 0.15 * elo_n,
            "mkt_surp_npxg_for": 0.0, "mkt_surp_npxg_ag": 0.0, "mkt_surp_pts": 0.0,
            "mkt_steam_f": 0.0, "mkt_steam_m": 0.0,
            "venue_att_h": 0.0, "venue_def_h": 0.0, "venue_att_a": 0.0, "venue_def_a": 0.0,
            "npxg_for_f": 1.25, "npxg_ag_f": 1.25,
            "npxg_for_m": 1.25, "npxg_ag_m": 1.25,
            "xgot_for_m": 1.20, "xgot_ag_m": 1.20,
            "xg_open_for_m": 0.90, "xg_open_ag_m": 0.90,
            "bc_for_m": 1.80, "bc_ag_m": 1.80,
            "opp_adj_npxg_for_m": 0.0, "opp_adj_npxg_ag_m": 0.0,
            "finishing_m": 0.0, "gk_save_m": 0.0,
        }

    def _season_regress(self, st: Dict[str, float]):
        for k in ("mkt_att_f", "mkt_def_f", "mkt_att_s", "mkt_def_s", "dc_blend_att", "dc_blend_def"):
            st[k] *= 0.74
        st["mkt_elo_f"] = 0.68 * st["mkt_elo_f"] + 0.32 * 1650.0
        st["mkt_elo_s"] = 0.78 * st["mkt_elo_s"] + 0.22 * 1650.0
        st["mkt_surp_npxg_for"] *= 0.4
        st["mkt_surp_npxg_ag"] *= 0.4
        st["mkt_surp_pts"] *= 0.4
        st["mkt_steam_f"] = 0.0
        st["mkt_steam_m"] = 0.0

    def _load_fd_dataframe(self, force_refresh: bool = False) -> pd.DataFrame:
        paths = sync_football_data_csvs(force_refresh=force_refresh)
        needed = [
            "Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HST", "AST", "HS", "AS",
            "PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA", "AvgH", "AvgD", "AvgA",
            "P>2.5", "P<2.5", "Avg>2.5", "Avg<2.5", "AHh", "AHCh",
        ]
        dfs = []
        for p in sorted(paths):
            try:
                d = pd.read_csv(p, encoding="utf-8-sig", on_bad_lines="skip")
            except Exception:
                d = pd.read_csv(p, encoding="latin1", on_bad_lines="skip")
            d.columns = [c.lstrip("\ufeff").strip() for c in d.columns]
            if "Div" not in d.columns or "Date" not in d.columns:
                continue
            keep = [c for c in needed if c in d.columns]
            d = d[keep].dropna(subset=["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
            dfs.append(d)
        if not dfs:
            return pd.DataFrame()
        fd = pd.concat(dfs, ignore_index=True).copy()

        def parse_dt(s):
            s = str(s).strip()
            for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
                try:
                    return pd.to_datetime(s, format=fmt)
                except Exception:
                    pass
            return pd.NaT

        fd["dt"] = fd["Date"].apply(parse_dt)
        fd = fd.dropna(subset=["dt"]).copy()
        fd["Division"] = fd["Div"].astype(str).str.strip()
        fd["H_canon"] = fd["HomeTeam"].apply(canon_team)
        fd["A_canon"] = fd["AwayTeam"].apply(canon_team)
        fd = fd.sort_values(["dt", "Division", "H_canon"]).reset_index(drop=True)

        for col in ("FTHG", "FTAG", "HST", "AST", "HS", "AS", "PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA",
                    "AvgH", "AvgD", "AvgA", "P>2.5", "P<2.5", "Avg>2.5", "Avg<2.5", "AHh", "AHCh"):
            if col in fd.columns:
                fd[col] = pd.to_numeric(fd[col], errors="coerce")
            else:
                fd[col] = np.nan

        fd["O25"] = fd["P>2.5"].combine_first(fd["Avg>2.5"])
        fd["U25"] = fd["P<2.5"].combine_first(fd["Avg<2.5"])
        fd["AH"] = fd["AHCh"].combine_first(fd["AHh"])
        return fd

    def _build(self, force_refresh: bool = False):
        fd = self._load_fd_dataframe(force_refresh=force_refresh)
        if fd.empty:
            return

        # Build FotMob cache index by (H_canon, A_canon, YYYY-MM-DD) for exact npxG/xGOT where cached!
        fotmob_stats_map: Dict[Tuple[str, str, str], Tuple[Dict[str, float], Dict[str, float]]] = {}
        cache_dir = os.path.join(BASE_DIR, "data", "scanner_cache")
        if os.path.isdir(cache_dir):
            import json
            for fn in os.listdir(cache_dir):
                if not fn.endswith(".json"):
                    continue
                fp = os.path.join(cache_dir, fn)
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        mj = json.load(f)
                    pp = mj.get("pageProps", mj)
                    gen = pp.get("general", {})
                    if not gen.get("finished"):
                        continue
                    ht_c = canon_team(gen.get("homeTeam", {}).get("name", ""))
                    at_c = canon_team(gen.get("awayTeam", {}).get("name", ""))
                    dt_str = str(gen.get("matchTimeUTCDate", ""))[:10]
                    if ht_c and at_c and dt_str:
                        sh_fm = extract_fotmob_advanced_stats(mj, 0)
                        sa_fm = extract_fotmob_advanced_stats(mj, 1)
                        if not np.isnan(sh_fm["npxg_for"]):
                            fotmob_stats_map[(ht_c, at_c, dt_str)] = (sh_fm, sa_fm)
                except Exception:
                    continue

        last_season: Dict[str, str] = {}
        for div in TOP5_DIVS:
            self.league_online[div] = {
                "avg_mkt_lam": 1.32, "hfa_mkt_lam": 0.28,
                "hfa_mkt_elo": 60.0, "avg_g": 1.32, "hfa_g": 0.28,
            }

        # Replay every completed match chronologically!
        per_div_records: Dict[str, List[Dict[str, Any]]] = {d: [] for d in TOP5_DIVS}
        ref_date = fd["dt"].max()

        for row in fd.itertuples(index=False):
            div = row.Division
            if div not in self.league_online:
                continue
            ht = row.H_canon
            at = row.A_canon
            dt = row.dt
            seas = f"{dt.year}/{dt.year+1}" if dt.month >= 7 else f"{dt.year-1}/{dt.year}"

            if ht not in self.team_state:
                self.team_state[ht] = self._init_team()
                last_season[ht] = seas
            elif last_season[ht] != seas:
                self._season_regress(self.team_state[ht])
                last_season[ht] = seas

            if at not in self.team_state:
                self.team_state[at] = self._init_team()
                last_season[at] = seas
            elif last_season[at] != seas:
                self._season_regress(self.team_state[at])
                last_season[at] = seas

            sh = self.team_state[ht]
            sa = self.team_state[at]
            lo_div = self.league_online[div]

            # Pre-match expectations
            mu_mkt = math.log(max(0.5, lo_div["avg_mkt_lam"]))
            hfa_mkt = lo_div["hfa_mkt_lam"]
            mkt_f_lh = math.exp(max(-1.5, min(1.8, mu_mkt + 0.5 * hfa_mkt + sh["mkt_att_f"] - sa["mkt_def_f"])))
            mkt_f_la = math.exp(max(-1.5, min(1.8, mu_mkt - 0.5 * hfa_mkt + sa["mkt_att_f"] - sh["mkt_def_f"])))
            mkt_s_lh = math.exp(max(-1.5, min(1.8, mu_mkt + 0.5 * hfa_mkt + sh["mkt_att_s"] - sa["mkt_def_s"])))
            mkt_s_la = math.exp(max(-1.5, min(1.8, mu_mkt - 0.5 * hfa_mkt + sa["mkt_att_s"] - sh["mkt_def_s"])))

            mu_g = math.log(max(0.5, lo_div["avg_g"]))
            hfa_g = lo_div["hfa_g"]
            dc_bl_lh = math.exp(max(-1.5, min(1.8, mu_g + 0.5 * hfa_g + sh["dc_blend_att"] - sa["dc_blend_def"])))
            dc_bl_la = math.exp(max(-1.5, min(1.8, mu_g - 0.5 * hfa_g + sa["dc_blend_att"] - sh["dc_blend_def"])))

            hfa_e = lo_div["hfa_mkt_elo"]
            mkt_exp_f = 1.0 / (1.0 + 10.0 ** (-(sh["mkt_elo_f"] - sa["mkt_elo_f"] + hfa_e) / 400.0))
            mkt_exp_s = 1.0 / (1.0 + 10.0 ** (-(sh["mkt_elo_s"] - sa["mkt_elo_s"] + hfa_e) / 400.0))

            # Match outcomes & xG proxy (uses exact FotMob npxG/xGOT if cached, else calibrated shots/SOT/goals proxy!)
            gf_h = float(row.FTHG)
            gf_a = float(row.FTAG)
            dt_iso = dt.strftime("%Y-%m-%d")
            fm_hit = fotmob_stats_map.get((ht, at, dt_iso))
            if fm_hit is not None:
                fm_h, fm_a = fm_hit
                npxg_h = fm_h["npxg_for"]
                npxg_a = fm_h["npxg_ag"]
                xgot_h = fm_h["xgot_for"] if not np.isnan(fm_h["xgot_for"]) else npxg_h
                xgot_a = fm_h["xgot_ag"] if not np.isnan(fm_h["xgot_ag"]) else npxg_a
                xgo_h = fm_h["xg_open_for"] if not np.isnan(fm_h["xg_open_for"]) else 0.72 * npxg_h
                xgo_a = fm_h["xg_open_ag"] if not np.isnan(fm_h["xg_open_ag"]) else 0.72 * npxg_a
                bc_h = fm_h["bc_for"] if not np.isnan(fm_h["bc_for"]) else 1.4 * npxg_h
                bc_a = fm_h["bc_ag"] if not np.isnan(fm_h["bc_ag"]) else 1.4 * npxg_a
            else:
                # Calibrated shot/SOT/goal npxG proxy for uncached warm-up matches
                sot_h = float(row.HST) if pd.notna(row.HST) else 4.2
                sot_a = float(row.AST) if pd.notna(row.AST) else 4.2
                sh_h = float(row.HS) if pd.notna(row.HS) else 12.0
                sh_a = float(row.AS) if pd.notna(row.AS) else 12.0
                npxg_h = max(0.15, min(3.8, 0.35 * gf_h + 0.14 * sot_h + 0.025 * sh_h))
                npxg_a = max(0.15, min(3.8, 0.35 * gf_a + 0.14 * sot_a + 0.025 * sh_a))
                xgot_h = npxg_h
                xgot_a = npxg_a
                xgo_h = 0.72 * npxg_h
                xgo_a = 0.72 * npxg_a
                bc_h = 1.4 * npxg_h
                bc_a = 1.4 * npxg_a

            if gf_h > gf_a:
                pts_h, pts_a = 3.0, 0.0
            elif gf_h < gf_a:
                pts_h, pts_a = 0.0, 3.0
            else:
                pts_h, pts_a = 1.0, 1.0

            # Online DC Blend update
            y_bl_h = 0.65 * npxg_h + 0.20 * xgot_h + 0.15 * gf_h
            y_bl_a = 0.65 * npxg_a + 0.20 * xgot_a + 0.15 * gf_a
            err_bl_h = max(-2.5, min(2.5, y_bl_h - dc_bl_lh))
            err_bl_a = max(-2.5, min(2.5, y_bl_a - dc_bl_la))
            sh["dc_blend_att"] = 0.998 * sh["dc_blend_att"] + 0.045 * err_bl_h
            sa["dc_blend_def"] = 0.998 * sa["dc_blend_def"] - 0.045 * err_bl_h
            sa["dc_blend_att"] = 0.998 * sa["dc_blend_att"] + 0.045 * err_bl_a
            sh["dc_blend_def"] = 0.998 * sh["dc_blend_def"] - 0.045 * err_bl_a

            sh["venue_att_h"] = 0.90 * sh["venue_att_h"] + 0.10 * err_bl_h
            sh["venue_def_h"] = 0.90 * sh["venue_def_h"] - 0.10 * err_bl_a
            sa["venue_att_a"] = 0.90 * sa["venue_att_a"] + 0.10 * err_bl_a
            sa["venue_def_a"] = 0.90 * sa["venue_def_a"] - 0.10 * err_bl_h

            # Past-Market Pinnacle update
            p_close = devig_shin_single(row.PSCH, row.PSCD, row.PSCA)
            p_open = devig_shin_single(row.PSH, row.PSD, row.PSA)
            p_mkt = p_close if p_close is not None else p_open
            if p_mkt is None:
                p_mkt = devig_shin_single(row.AvgH, row.AvgD, row.AvgA)

            log_mlh = np.nan
            log_mla = np.nan
            mkt_logit_s = np.nan
            if p_mkt is not None:
                s_mkt = max(0.02, min(0.98, float(p_mkt[0] + 0.5 * p_mkt[1])))
                mkt_logit_s = math.log(s_mkt / (1.0 - s_mkt))
                sh["mkt_elo_f"] += 95.0 * (s_mkt - mkt_exp_f)
                sa["mkt_elo_f"] -= 95.0 * (s_mkt - mkt_exp_f)
                sh["mkt_elo_s"] += 25.0 * (s_mkt - mkt_exp_s)
                sa["mkt_elo_s"] -= 25.0 * (s_mkt - mkt_exp_s)

                p_o25 = devig_two_way(row.O25, row.U25)
                ahh = float(row.AH) if pd.notna(row.AH) else np.nan
                lam_mh, lam_ma = infer_market_lambdas(p_mkt[0], p_mkt[1], p_mkt[2], p_o25, ahh)
                log_mlh = math.log(lam_mh)
                log_mla = math.log(lam_ma)

                lo_div["avg_mkt_lam"] = 0.996 * lo_div["avg_mkt_lam"] + 0.004 * (0.5 * (lam_mh + lam_ma))
                err_mf_h = max(-1.5, min(1.5, log_mlh - math.log(mkt_f_lh)))
                err_mf_a = max(-1.5, min(1.5, log_mla - math.log(mkt_f_la)))
                sh["mkt_att_f"] = 0.995 * sh["mkt_att_f"] + 0.22 * err_mf_h
                sa["mkt_def_f"] = 0.995 * sa["mkt_def_f"] - 0.22 * err_mf_h
                sa["mkt_att_f"] = 0.995 * sa["mkt_att_f"] + 0.22 * err_mf_a
                sh["mkt_def_f"] = 0.995 * sh["mkt_def_f"] - 0.22 * err_mf_a

                err_ms_h = max(-1.5, min(1.5, log_mlh - math.log(mkt_s_lh)))
                err_ms_a = max(-1.5, min(1.5, log_mla - math.log(mkt_s_la)))
                sh["mkt_att_s"] = 0.997 * sh["mkt_att_s"] + 0.09 * err_ms_h
                sa["mkt_def_s"] = 0.997 * sa["mkt_def_s"] - 0.09 * err_ms_h
                sa["mkt_att_s"] = 0.997 * sa["mkt_att_s"] + 0.09 * err_ms_a
                sh["mkt_def_s"] = 0.997 * sh["mkt_def_s"] - 0.09 * err_ms_a

                sh["mkt_surp_npxg_for"] = 0.85 * sh["mkt_surp_npxg_for"] + 0.15 * (npxg_h - lam_mh)
                sh["mkt_surp_npxg_ag"] = 0.85 * sh["mkt_surp_npxg_ag"] + 0.15 * (npxg_a - lam_ma)
                sa["mkt_surp_npxg_for"] = 0.85 * sa["mkt_surp_npxg_for"] + 0.15 * (npxg_a - lam_ma)
                sa["mkt_surp_npxg_ag"] = 0.85 * sa["mkt_surp_npxg_ag"] + 0.15 * (npxg_h - lam_mh)

                mkt_xpts_h = 3.0 * p_mkt[0] + p_mkt[1]
                mkt_xpts_a = 3.0 * p_mkt[2] + p_mkt[1]
                sh["mkt_surp_pts"] = 0.85 * sh["mkt_surp_pts"] + 0.15 * (pts_h - mkt_xpts_h)
                sa["mkt_surp_pts"] = 0.85 * sa["mkt_surp_pts"] + 0.15 * (pts_a - mkt_xpts_a)

                if p_close is not None and p_open is not None:
                    steam = (p_close[0] + 0.5 * p_close[1]) - (p_open[0] + 0.5 * p_open[1])
                    sh["mkt_steam_f"] = 0.72 * sh["mkt_steam_f"] + 0.28 * steam
                    sa["mkt_steam_f"] = 0.72 * sa["mkt_steam_f"] - 0.28 * steam
                    sh["mkt_steam_m"] = 0.88 * sh["mkt_steam_m"] + 0.12 * steam
                    sa["mkt_steam_m"] = 0.88 * sa["mkt_steam_m"] - 0.12 * steam

            # Multi-scale EWMA updates
            adj_for_h = npxg_h - sa["npxg_ag_m"]
            adj_ag_h = npxg_a - sa["npxg_for_m"]
            adj_for_a = npxg_a - sh["npxg_ag_m"]
            adj_ag_a = npxg_h - sh["npxg_for_m"]

            sh["npxg_for_f"] = 0.72 * sh["npxg_for_f"] + 0.28 * npxg_h
            sh["npxg_ag_f"] = 0.72 * sh["npxg_ag_f"] + 0.28 * npxg_a
            sa["npxg_for_f"] = 0.72 * sa["npxg_for_f"] + 0.28 * npxg_a
            sa["npxg_ag_f"] = 0.72 * sa["npxg_ag_f"] + 0.28 * npxg_h

            for st_t, nf, na, xf, xa, of_, oa, bf, ba, af, aa, gf_t, ga_t in (
                (sh, npxg_h, npxg_a, xgot_h, xgot_a, xgo_h, xgo_a, bc_h, bc_a, adj_for_h, adj_ag_h, gf_h, gf_a),
                (sa, npxg_a, npxg_h, xgot_a, xgot_h, xgo_a, xgo_h, bc_a, bc_h, adj_for_a, adj_ag_a, gf_a, gf_h),
            ):
                st_t["npxg_for_m"] = 0.87 * st_t["npxg_for_m"] + 0.13 * nf
                st_t["npxg_ag_m"] = 0.87 * st_t["npxg_ag_m"] + 0.13 * na
                st_t["xgot_for_m"] = 0.87 * st_t["xgot_for_m"] + 0.13 * xf
                st_t["xgot_ag_m"] = 0.87 * st_t["xgot_ag_m"] + 0.13 * xa
                st_t["xg_open_for_m"] = 0.87 * st_t["xg_open_for_m"] + 0.13 * of_
                st_t["xg_open_ag_m"] = 0.87 * st_t["xg_open_ag_m"] + 0.13 * oa
                st_t["bc_for_m"] = 0.87 * st_t["bc_for_m"] + 0.13 * bf
                st_t["bc_ag_m"] = 0.87 * st_t["bc_ag_m"] + 0.13 * ba
                st_t["opp_adj_npxg_for_m"] = 0.87 * st_t["opp_adj_npxg_for_m"] + 0.13 * af
                st_t["opp_adj_npxg_ag_m"] = 0.87 * st_t["opp_adj_npxg_ag_m"] + 0.13 * aa
                st_t["finishing_m"] = 0.87 * st_t["finishing_m"] + 0.13 * (xf - nf)
                st_t["gk_save_m"] = 0.87 * st_t["gk_save_m"] + 0.13 * (xa - ga_t)

            age_days = max(0.0, float((ref_date - dt).days))
            if age_days <= 730.0:
                per_div_records[div].append({
                    "ht": ht, "at": at, "age": age_days,
                    "log_mlh": log_mlh, "log_mla": log_mla, "mls": mkt_logit_s,
                    "npxg_h": npxg_h, "npxg_a": npxg_a,
                })

        # Fit exact rolling snapshot solvers as of TODAY per league!
        for div in TOP5_DIVS:
            recs = per_div_records[div]
            if len(recs) < 40:
                continue
            teams = sorted({r["ht"] for r in recs} | {r["at"] for r in recs})
            t2i = {t: idx for idx, t in enumerate(teams)}
            nt = len(teams)
            prior_att = np.zeros(nt, dtype=np.float64)
            prior_def = np.zeros(nt, dtype=np.float64)
            prior_bt = np.zeros(nt, dtype=np.float64)

            hh_m, aa_m, lmh, lma, mls, wm35, wm220 = [], [], [], [], [], [], []
            hh_x, aa_x, ynh, yna, wx120 = [], [], [], [], []
            for r in recs:
                hi = t2i[r["ht"]]
                ai = t2i[r["at"]]
                age = r["age"]
                if np.isfinite(r["log_mlh"]) and np.isfinite(r["log_mla"]):
                    hh_m.append(hi); aa_m.append(ai)
                    lmh.append(r["log_mlh"]); lma.append(r["log_mla"]); mls.append(r["mls"])
                    wm35.append(math.exp(-math.log(2.0) * age / 35.0))
                    wm220.append(math.exp(-math.log(2.0) * age / 220.0))
                hh_x.append(hi); aa_x.append(ai)
                ynh.append(r["npxg_h"]); yna.append(r["npxg_a"])
                wx120.append(math.exp(-math.log(2.0) * age / 120.0))

            mkt35 = _solve_weighted_ridge_att_def(
                np.array(hh_m), np.array(aa_m), np.array(lmh), np.array(lma), np.array(wm35), nt, prior_att, prior_def, reg=2.2
            )
            bt35 = _solve_weighted_bt_strength(np.array(hh_m), np.array(aa_m), np.array(mls), np.array(wm35), nt, prior_bt, reg=1.8)
            bt220 = _solve_weighted_bt_strength(np.array(hh_m), np.array(aa_m), np.array(mls), np.array(wm220), nt, prior_bt, reg=3.2)
            dc120 = _solve_weighted_poisson_att_def(
                np.array(hh_x), np.array(aa_x), np.array(ynh), np.array(yna), np.array(wx120), nt, prior_att, prior_def, reg=5.0, n_iter=6
            )
            self.league_solvers[div] = {
                "t2i": t2i, "mkt35": mkt35, "bt35": bt35, "bt220": bt220, "dc120": dc120
            }

    def build_match_payload(
        self,
        div: str,
        home_team: str,
        away_team: str,
        home_elo: float,
        away_elo: float,
        rest_days_h: float,
        rest_days_a: float,
        matches_14d_h: float,
        matches_14d_a: float,
        home_pts: float,
        away_pts: float,
        matchday: float,
        standings_rows: List[Dict[str, Any]],
        total_games: int = 38,
        rel_pos_1idx: int = 18,
    ) -> Dict[str, Any]:
        """
        Builds exact Deep-Pitch v2 generative base components + MOT_COLS + RES_COLS for a live fixture.
        """
        ht = canon_team(home_team)
        at = canon_team(away_team)
        sh = self.team_state.get(ht, self._init_team(home_elo))
        sa = self.team_state.get(at, self._init_team(away_elo))
        lo_div = self.league_online.get(div, {"avg_mkt_lam": 1.32, "hfa_mkt_lam": 0.28, "avg_g": 1.32, "hfa_g": 0.28})

        # Online Past-Market DC (fast & slow)
        mu_m = math.log(max(0.5, lo_div["avg_mkt_lam"]))
        hfa_m = lo_div["hfa_mkt_lam"]
        mf_lh = math.exp(max(-1.5, min(1.8, mu_m + 0.5 * hfa_m + sh["mkt_att_f"] - sa["mkt_def_f"])))
        mf_la = math.exp(max(-1.5, min(1.8, mu_m - 0.5 * hfa_m + sa["mkt_att_f"] - sh["mkt_def_f"])))
        p_mkt_f = list(poisson_1x2_dc(mf_lh, mf_la, rho=-0.065))

        ms_lh = math.exp(max(-1.5, min(1.8, mu_m + 0.5 * hfa_m + sh["mkt_att_s"] - sa["mkt_def_s"])))
        ms_la = math.exp(max(-1.5, min(1.8, mu_m - 0.5 * hfa_m + sa["mkt_att_s"] - sh["mkt_def_s"])))
        p_mkt_s = list(poisson_1x2_dc(ms_lh, ms_la, rho=-0.065))

        # Online DC Blend
        mu_g = math.log(max(0.5, lo_div["avg_g"]))
        hfa_g = lo_div["hfa_g"]
        bl_lh = math.exp(max(-1.5, min(1.8, mu_g + 0.5 * hfa_g + sh["dc_blend_att"] - sa["dc_blend_def"])))
        bl_la = math.exp(max(-1.5, min(1.8, mu_g - 0.5 * hfa_g + sa["dc_blend_att"] - sh["dc_blend_def"])))
        p_dc_bl = list(poisson_1x2_dc(bl_lh, bl_la, rho=-0.065))

        # Exact Rolling 35d Past-Market & 120d Opta npxG solvers
        slv = self.league_solvers.get(div)
        if slv is not None and ht in slv["t2i"] and at in slv["t2i"]:
            hi = slv["t2i"][ht]
            ai = slv["t2i"][at]
            mu35, hfa35, att35, def35 = slv["mkt35"]
            ex_m35_lh = math.exp(max(-1.5, min(1.8, mu35 + 0.5 * hfa35 + att35[hi] - def35[ai])))
            ex_m35_la = math.exp(max(-1.5, min(1.8, mu35 - 0.5 * hfa35 + att35[ai] - def35[hi])))
            p_ex_mkt35 = list(poisson_1x2_dc(ex_m35_lh, ex_m35_la, rho=-0.065))

            mu120, hfa120, att120, def120 = slv["dc120"]
            ex_dc_lh = math.exp(max(-1.5, min(1.8, mu120 + 0.5 * hfa120 + att120[hi] - def120[ai])))
            ex_dc_la = math.exp(max(-1.5, min(1.8, mu120 - 0.5 * hfa120 + att120[ai] - def120[hi])))
            p_ex_dc120 = list(poisson_1x2_dc(ex_dc_lh, ex_dc_la, rho=-0.065))

            hfa_bt35, r35 = slv["bt35"]
            hfa_bt220, r220 = slv["bt220"]
            ex_bt_mom = (hfa_bt35 + r35[hi] - r35[ai]) - (hfa_bt220 + r220[hi] - r220[ai])
        else:
            ex_m35_lh, ex_m35_la = mf_lh, mf_la
            p_ex_mkt35 = p_mkt_f
            ex_dc_lh, ex_dc_la = bl_lh, bl_la
            p_ex_dc120 = p_dc_bl
            ex_bt_mom = 0.0

        # Exact mathematical end-of-season motivation
        mot_dict = compute_live_motivation_features(
            standings_rows, home_team, away_team, total_games=total_games, rel_pos_1idx=rel_pos_1idx
        )

        # Residual features dictionary (1:1 match with train_v2.RES_COLS!)
        elo_diff = float(home_elo - away_elo)
        mkt_elo_diff_f = float(sh["mkt_elo_f"] - sa["mkt_elo_f"])
        res_dict = {
            "mkt_vs_dc_p1": p_ex_mkt35[0] - p_ex_dc120[0],
            "mkt_vs_dc_p2": p_ex_mkt35[2] - p_ex_dc120[2],
            "mkt_vs_dc_ldiff": (ex_m35_lh - ex_m35_la) - (ex_dc_lh - ex_dc_la),
            "elo_diff": elo_diff,
            "elo_minus_mkt_diff": elo_diff - mkt_elo_diff_f,
            "mkt_trend_diff": (sh["mkt_elo_f"] - sh["mkt_elo_s"]) - (sa["mkt_elo_f"] - sa["mkt_elo_s"]),
            "ex_bt_mom_35_220": float(ex_bt_mom),
            "mkt_steam_f_diff": sh["mkt_steam_f"] - sa["mkt_steam_f"],
            "mkt_steam_m_diff": sh["mkt_steam_m"] - sa["mkt_steam_m"],
            "mkt_surp_net_diff": (sh["mkt_surp_npxg_for"] - sh["mkt_surp_npxg_ag"]) - (sa["mkt_surp_npxg_for"] - sa["mkt_surp_npxg_ag"]),
            "mkt_surp_pts_diff": sh["mkt_surp_pts"] - sa["mkt_surp_pts"],
            "npxg_net_diff_f": (sh["npxg_for_f"] - sh["npxg_ag_f"]) - (sa["npxg_for_f"] - sa["npxg_ag_f"]),
            "npxg_net_diff_m": (sh["npxg_for_m"] - sh["npxg_ag_m"]) - (sa["npxg_for_m"] - sa["npxg_ag_m"]),
            "xgot_net_diff_m": (sh["xgot_for_m"] - sh["xgot_ag_m"]) - (sa["xgot_for_m"] - sa["xgot_ag_m"]),
            "xg_open_net_diff_m": (sh["xg_open_for_m"] - sh["xg_open_ag_m"]) - (sa["xg_open_for_m"] - sa["xg_open_ag_m"]),
            "bc_net_diff_m": (sh["bc_for_m"] - sh["bc_ag_m"]) - (sa["bc_for_m"] - sa["bc_ag_m"]),
            "opp_adj_npxg_diff_m": (sh["opp_adj_npxg_for_m"] - sh["opp_adj_npxg_ag_m"]) - (sa["opp_adj_npxg_for_m"] - sa["opp_adj_npxg_ag_m"]),
            "finishing_diff_m": sh["finishing_m"] - sa["finishing_m"],
            "gk_save_diff_m": sh["gk_save_m"] - sa["gk_save_m"],
            "venue_boost_h": sh["venue_att_h"] + sh["venue_def_h"],
            "venue_boost_a": sa["venue_att_a"] + sa["venue_def_a"],
            "rest_diff": float(min(rest_days_h, 14.0) - min(rest_days_a, 14.0)),
            "cong_diff": float(matches_14d_h - matches_14d_a),
            "pts_pg_diff": float(home_pts - away_pts) / max(1.0, float(matchday)),
            "is_E0": 1.0 if div == "E0" else 0.0,
            "is_I1": 1.0 if div == "I1" else 0.0,
            "is_SP1": 1.0 if div == "SP1" else 0.0,
            "is_D1": 1.0 if div == "D1" else 0.0,
            "is_F1": 1.0 if div == "F1" else 0.0,
        }

        return {
            "gen_components": [p_mkt_f, p_mkt_s, p_ex_mkt35, p_ex_dc120, p_dc_bl],
            "mot_features": mot_dict,
            "res_features": res_dict,
            "known_home": ht in self.team_state,
            "known_away": at in self.team_state,
        }
