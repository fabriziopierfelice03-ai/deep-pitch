"""
src/injury_lineup.py
Live pre-match Injury, Suspension & Predicted/Confirmed Lineup Delta analyzer
powered by FotMob matchDetails JSONs (zero extra APIs, 100% free).

Core quant principle (avoiding double-counting):
  - Long-term injuries (out for weeks) are ALREADY priced into Deep-Pitch v2's
    Past-Market Pinnacle ratings.
  - Only FRESH shocks (regular starters from the team's last 5 matches who are
    now injured, suspended, doubtful, or rotated out of today's predicted/confirmed XI)
    represent missing F_t \\ F_{t-1} information.
"""

import re
import unicodedata
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


def _norm_name(text: str) -> str:
    if not text:
        return ""
    clean = "".join(c for c in unicodedata.normalize("NFD", str(text).lower()) if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]", "", clean)


def _extract_team_block_from_completed(match_json: Dict[str, Any], team_name: str, team_id: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    pp = match_json.get("pageProps", match_json)
    lineup = pp.get("content", {}).get("lineup", {})
    if not isinstance(lineup, dict):
        return None
    target_norm = _norm_name(team_name)
    for side in ("homeTeam", "awayTeam"):
        blk = lineup.get(side)
        if not isinstance(blk, dict):
            continue
        if team_id is not None and blk.get("id") == team_id:
            return blk
        blk_norm = _norm_name(blk.get("name", ""))
        if blk_norm and (blk_norm == target_norm or target_norm in blk_norm or blk_norm in target_norm):
            return blk
    return None


def build_recent_starter_baseline(
    recent_match_jsons: List[Dict[str, Any]],
    team_name: str,
    team_id: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Scans up to the last 5 completed FotMob match JSONs for a team and extracts:
      - regular starters (player_id / normalized name, start count, avg rating)
      - team baseline average XI FotMob rating across those matches
    """
    by_id: Dict[int, Dict[str, Any]] = {}
    by_name: Dict[str, Dict[str, Any]] = {}
    match_xi_ratings: List[float] = []
    n_parsed = 0

    for mj in recent_match_jsons[:5]:
        if not isinstance(mj, dict):
            continue
        blk = _extract_team_block_from_completed(mj, team_name, team_id=team_id)
        if not blk:
            continue
        starters = blk.get("starters", [])
        if not starters:
            continue
        n_parsed += 1
        ratings_this_match = []
        for pl in starters:
            if not isinstance(pl, dict):
                continue
            pid = pl.get("id")
            pname = str(pl.get("name", "")).strip()
            pnorm = _norm_name(pname)
            perf = pl.get("performance", {})
            rating_val = None
            if isinstance(perf, dict) and perf.get("rating") is not None:
                try:
                    rating_val = float(perf["rating"])
                    ratings_this_match.append(rating_val)
                except Exception:
                    rating_val = None

            if isinstance(pid, int) and pid > 0:
                if pid not in by_id:
                    by_id[pid] = {"id": pid, "name": pname, "starts": 0, "ratings": []}
                by_id[pid]["starts"] += 1
                if rating_val is not None:
                    by_id[pid]["ratings"].append(rating_val)
            if pnorm:
                if pnorm not in by_name:
                    by_name[pnorm] = {"name": pname, "starts": 0, "ratings": []}
                by_name[pnorm]["starts"] += 1
                if rating_val is not None:
                    by_name[pnorm]["ratings"].append(rating_val)

        if ratings_this_match:
            match_xi_ratings.append(float(np.mean(ratings_this_match)))

    baseline_xi_rating = float(np.mean(match_xi_ratings)) if match_xi_ratings else 6.80
    return {
        "n_matches": n_parsed,
        "by_id": by_id,
        "by_name": by_name,
        "baseline_xi_rating": baseline_xi_rating,
    }


def _lookup_recent_starter(baseline: Dict[str, Any], pid: Any, pname: str) -> Optional[Dict[str, Any]]:
    if isinstance(pid, int) and pid > 0 and pid in baseline["by_id"]:
        return baseline["by_id"][pid]
    pnorm = _norm_name(pname)
    if pnorm and pnorm in baseline["by_name"]:
        return baseline["by_name"][pnorm]
    return None


def analyze_team_availability(
    upcoming_team_block: Optional[Dict[str, Any]],
    lineup_type: str,
    baseline: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Analyzes a single team's upcoming FotMob lineup block against its last-5 baseline.
    """
    empty_res = {
        "lineup_type": lineup_type or "unknown",
        "starter_market_value": 0.0,
        "fresh_missing_count": 0.0,
        "fresh_missing_value_eur": 0.0,
        "fresh_missing_value_ratio": 0.0,
        "all_unavailable_count": 0,
        "all_unavailable_value_ratio": 0.0,
        "xi_rating_delta": 0.0,
        "fresh_missing_details": [],
        "all_unavailable_names": [],
    }
    if not isinstance(upcoming_team_block, dict):
        return empty_res

    starters = upcoming_team_block.get("starters", []) or []
    unavailable = upcoming_team_block.get("unavailable", []) or []

    total_mv = float(upcoming_team_block.get("totalStarterMarketValue") or 0.0)
    if total_mv <= 0.0 and starters:
        total_mv = sum(float(p.get("marketValue") or 0.0) for p in starters if isinstance(p, dict))
    denom_mv = max(total_mv, 60_000_000.0)

    n_base = baseline.get("n_matches", 0)
    min_starts_regular = 2 if n_base >= 3 else 1

    fresh_count = 0.0
    fresh_mv = 0.0
    fresh_details: List[str] = []
    all_unavail_mv = 0.0
    all_unavail_names: List[str] = []
    seen_missing_keys = set()

    # 1. Inspect officially listed unavailable players (injured / suspended / doubtful)
    for pl in unavailable:
        if not isinstance(pl, dict):
            continue
        pid = pl.get("id")
        pname = str(pl.get("name", "Unknown")).strip()
        mv = float(pl.get("marketValue") or 0.0)
        unav_info = pl.get("unavailability", {}) or {}
        utype = str(unav_info.get("type", "injury")).lower()
        exp_ret = str(unav_info.get("expectedReturn", "")).strip()

        is_doubtful = "doubt" in exp_ret.lower()
        w_out = 0.50 if is_doubtful else 1.00

        all_unavail_mv += w_out * mv
        tag_ret = f"{utype}/doubtful" if is_doubtful else utype
        all_unavail_names.append(f"{pname} ({tag_ret})")

        rec = _lookup_recent_starter(baseline, pid, pname)
        if rec is not None and rec["starts"] >= min_starts_regular:
            # Fresh shock: this player was starting in the last 5 matches and is now out!
            fresh_count += w_out
            # If marketValue missing on FotMob, assign modest 1/11th starter share
            eff_mv = mv if mv > 0.0 else (denom_mv / 11.0)
            fresh_mv += w_out * eff_mv
            mv_m = eff_mv / 1_000_000.0
            fresh_details.append(f"{pname} ({tag_ret}, {rec['starts']}/{n_base} tit., €{mv_m:.0f}M)")
            seen_missing_keys.add(_norm_name(pname))

    # 2. If lineupType is 'predicted' or 'confirmed' (not just lastStarting11), detect rotation!
    ltype_low = str(lineup_type or "").lower()
    xi_rating_delta = 0.0
    if ltype_low not in ("laststarting11", "unknown", "") and starters and n_base >= 3:
        starter_ids = {p.get("id") for p in starters if isinstance(p, dict) and isinstance(p.get("id"), int)}
        starter_norms = {_norm_name(p.get("name", "")) for p in starters if isinstance(p, dict)}

        # Check core starters (>= 3 starts in last 5) who were dropped / rested today
        for pnorm, rec in baseline["by_name"].items():
            if rec["starts"] >= 3 and pnorm not in starter_norms and pnorm not in seen_missing_keys:
                fresh_count += 0.75
                fresh_mv += 0.75 * (denom_mv / 11.0)
                fresh_details.append(f"{rec['name']} (benched/rotated, {rec['starts']}/{n_base} tit.)")
                seen_missing_keys.add(pnorm)

        # Compare average known rating of today's XI vs baseline XI
        today_ratings = []
        for pl in starters:
            if not isinstance(pl, dict):
                continue
            rec = _lookup_recent_starter(baseline, pl.get("id"), pl.get("name", ""))
            if rec is not None and rec["ratings"]:
                today_ratings.append(float(np.mean(rec["ratings"])))
        if len(today_ratings) >= 7:
            xi_rating_delta = float(np.mean(today_ratings) - baseline["baseline_xi_rating"])
            xi_rating_delta = max(-0.60, min(0.60, xi_rating_delta))

    fresh_ratio = min(0.50, fresh_mv / denom_mv)
    all_ratio = min(0.60, all_unavail_mv / denom_mv)

    return {
        "lineup_type": lineup_type or "unknown",
        "starter_market_value": total_mv,
        "fresh_missing_count": round(fresh_count, 2),
        "fresh_missing_value_eur": round(fresh_mv, 0),
        "fresh_missing_value_ratio": round(fresh_ratio, 4),
        "all_unavailable_count": len(unavailable),
        "all_unavailable_value_ratio": round(all_ratio, 4),
        "xi_rating_delta": round(xi_rating_delta, 4),
        "fresh_missing_details": fresh_details,
        "all_unavailable_names": all_unavail_names,
    }


def build_match_injury_report(
    upcoming_match_json: Optional[Dict[str, Any]],
    home_team_name: str,
    away_team_name: str,
    home_recent_jsons: List[Dict[str, Any]],
    away_recent_jsons: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Builds a complete pre-match injury & lineup availability report for both sides.
    """
    pp = upcoming_match_json.get("pageProps", upcoming_match_json) if isinstance(upcoming_match_json, dict) else {}
    lineup = pp.get("content", {}).get("lineup", {}) if isinstance(pp, dict) else {}
    if not isinstance(lineup, dict):
        lineup = {}

    lineup_type = str(lineup.get("lineupType", "unknown"))
    ht_blk = lineup.get("homeTeam")
    at_blk = lineup.get("awayTeam")
    ht_id = ht_blk.get("id") if isinstance(ht_blk, dict) else None
    at_id = at_blk.get("id") if isinstance(at_blk, dict) else None

    h_base = build_recent_starter_baseline(home_recent_jsons, home_team_name, team_id=ht_id)
    a_base = build_recent_starter_baseline(away_recent_jsons, away_team_name, team_id=at_id)

    h_rep = analyze_team_availability(ht_blk, lineup_type, h_base)
    a_rep = analyze_team_availability(at_blk, lineup_type, a_base)

    # Positive net_home_adv means AWAY is more hurt by fresh absences than HOME.
    # Calibrated conservatively: missing ~20% of XI value (~2 key starters) shifts win prob ~4-5% (~0.11 logit).
    net_val_ratio = a_rep["fresh_missing_value_ratio"] - h_rep["fresh_missing_value_ratio"]
    net_count = a_rep["fresh_missing_count"] - h_rep["fresh_missing_count"]
    net_xi_delta = h_rep["xi_rating_delta"] - a_rep["xi_rating_delta"]

    raw_logit_shift = 0.42 * net_val_ratio + 0.020 * net_count + 0.08 * net_xi_delta
    logit_shift = max(-0.16, min(0.16, float(raw_logit_shift)))

    return {
        "lineup_type": lineup_type,
        "home": h_rep,
        "away": a_rep,
        "net_logit_shift_home": round(logit_shift, 4),
    }


def apply_injury_lineup_adjustment(probs: List[float], injury_report: Optional[Dict[str, Any]]) -> List[float]:
    """
    Applies a conservative, bounded logit adjustment for FRESH starter absences/rotation.
    If no fresh absences exist, returns exact original probabilities unchanged.
    """
    if not injury_report:
        return list(probs)
    shift = float(injury_report.get("net_logit_shift_home", 0.0))
    if abs(shift) < 1e-5:
        return list(probs)

    p = np.clip(np.array(probs, dtype=np.float64), 1e-6, 1.0)
    logits = np.log(p)
    logits[0] += shift
    logits[2] -= shift
    logits -= np.max(logits)
    exp_l = np.exp(logits)
    p_adj = exp_l / np.sum(exp_l)
    return [float(p_adj[0]), float(p_adj[1]), float(p_adj[2])]


def check_injury_trap_guard(selection: str, injury_report: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Adverse-selection trap guard:
    Blocks betting ON a team that is suffering a major FRESH injury/suspension/rotation shock
    compared to its opponent (i.e. missing >= 2 regular starters or >= 18% of XI value freshly).
    """
    if not injury_report:
        return False, ""
    h = injury_report.get("home", {})
    a = injury_report.get("away", {})

    h_cnt = float(h.get("fresh_missing_count", 0.0))
    a_cnt = float(a.get("fresh_missing_count", 0.0))
    h_rat = float(h.get("fresh_missing_value_ratio", 0.0))
    a_rat = float(a.get("fresh_missing_value_ratio", 0.0))

    if selection == "1":
        if (h_cnt >= 2.0 and (h_cnt - a_cnt) >= 1.5) or (h_rat >= 0.18 and (h_rat - a_rat) >= 0.10):
            names = ", ".join(h.get("fresh_missing_details", [])[:4])
            return True, f"FRESH_INJURY_TRAP_HOME ({h_cnt:.1f} tit. out, {h_rat*100:.1f}% XI val: {names})"

    if selection == "2":
        if (a_cnt >= 2.0 and (a_cnt - h_cnt) >= 1.5) or (a_rat >= 0.18 and (a_rat - h_rat) >= 0.10):
            names = ", ".join(a.get("fresh_missing_details", [])[:4])
            return True, f"FRESH_INJURY_TRAP_AWAY ({a_cnt:.1f} tit. out, {a_rat*100:.1f}% XI val: {names})"

    return False, ""
