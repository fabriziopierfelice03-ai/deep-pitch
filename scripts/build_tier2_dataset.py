"""
scripts/build_tier2_dataset.py
Costruisce il dataset storico completo per le leghe europee Tier-2 con esattamente
le 66 colonne (56 feature + target + xG + Elo) richieste da Deep-Pitch (`src/train_and_export.py`
e `src/feature_builder.py`), utilizzando esclusivamente dati Opta/FotMob verificati.

Leghe incluse (tutte con Opta xG completo dal 2021/22 al 2026/27):
- E1: English Championship (id=48)
- N1: Dutch Eredivisie (id=57)
- P1: Liga Portugal (id=61)
- D2: German 2. Bundesliga (id=146)
- B1: Belgian Pro League (id=40)
"""

import os
import sys
import json
import time
import random
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATS_JSONL = os.path.join(DATA_DIR, "tier2_fotmob_stats.jsonl")
OUT_CSV = os.path.join(DATA_DIR, "matches_enriched_xg_tier2_2021_2026.csv")

SEASONS = ["2021/2022", "2022/2023", "2023/2024", "2024/2025", "2025/2026", "2026/2027"]

TIER2_CONFIG = {
    "E1": {"name": "Championship", "fotmob_id": 48, "init_elo": 1560.0, "ucl": 2, "uel": 6, "rel": 22},
    "N1": {"name": "Eredivisie", "fotmob_id": 57, "init_elo": 1545.0, "ucl": 2, "uel": 4, "rel": 16},
    "P1": {"name": "Liga Portugal", "fotmob_id": 61, "init_elo": 1550.0, "ucl": 2, "uel": 4, "rel": 16},
    "D2": {"name": "2. Bundesliga", "fotmob_id": 146, "init_elo": 1540.0, "ucl": 2, "uel": 3, "rel": 16},
    "B1": {"name": "Belgian Pro League", "fotmob_id": 40, "init_elo": 1540.0, "ucl": 2, "uel": 6, "rel": 15},
}

CSV_HEADER = [
    "Division", "MatchDate", "HomeTeam", "AwayTeam", "HomeElo", "AwayElo",
    "gol_casa", "gol_trasferta", "risultato_1X2", "xg_casa", "xg_trasferta",
    "pos_classifica_casa", "pos_classifica_trasferta", "punti_totali_casa", "punti_totali_trasferta",
    "punti_m5_pond_gen_casa", "punti_m5_pond_gen_trasferta", "punti_m5_pond_split_casa", "punti_m5_pond_split_trasferta",
    "xg_m5_pond_gen_casa", "xg_m5_pond_gen_trasferta", "xg_m5_pond_split_casa", "xg_m5_pond_split_trasferta",
    "xga_m5_pond_gen_casa", "xga_m5_pond_gen_trasferta", "xga_m5_pond_split_casa", "xga_m5_pond_split_trasferta",
    "gf_m5_pond_gen_casa", "gf_m5_pond_gen_trasferta", "gf_m5_pond_split_casa", "gf_m5_pond_split_trasferta",
    "gs_m5_pond_gen_casa", "gs_m5_pond_gen_trasferta", "gs_m5_pond_split_casa", "gs_m5_pond_split_trasferta",
    "diff_xg_gol_m5_gen_casa", "diff_xg_gol_m5_gen_trasferta",
    "h2h_punti_m5_casa", "h2h_punti_m5_trasferta", "h2h_gf_m5_casa", "h2h_gs_m5_casa",
    "h2h_xg_m5_casa", "h2h_xg_m5_trasferta", "h2h_split_punti_m5_casa", "h2h_split_gf_m5_casa",
    "giorni_riposo_casa", "giorni_riposo_trasferta", "partite_14gg_casa", "partite_14gg_trasferta",
    "is_derby",
    "dist_champions_casa", "dist_champions_trasferta", "dist_europa_casa", "dist_europa_trasferta",
    "dist_salvezza_casa", "dist_salvezza_trasferta", "giornata_stagione", "meta_stagione",
    "tiri_m5_gen_casa", "tiri_m5_gen_trasferta", "tiri_porta_m5_gen_casa", "tiri_porta_m5_gen_trasferta",
    "corner_m5_gen_casa", "corner_m5_gen_trasferta", "cartellini_m5_gen_casa", "cartellini_m5_gen_trasferta"
]


def fetch_json_retry(url: str, retries: int = 4, timeout: int = 15) -> Optional[Dict[str, Any]]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep((1.5 ** attempt) + random.uniform(0.2, 0.8))
                continue
            return None
        except Exception:
            time.sleep((1.3 ** attempt) + random.uniform(0.1, 0.5))
    return None


def discover_regular_fixtures() -> List[Dict[str, Any]]:
    """Scarica i calendari completi delle 5 leghe su tutte le stagioni filtrando la sola regular season."""
    fixtures = []
    seen_ids = set()

    for lcode, conf in TIER2_CONFIG.items():
        lid = conf["fotmob_id"]
        for season in SEASONS:
            u = f"https://www.fotmob.com/api/data/leagues?id={lid}&season={urllib.parse.quote(season)}"
            data = fetch_json_retry(u)
            if not data or "fixtures" not in data:
                print(f"[WARN] Impossibile leggere calendario {lcode} {season}")
                continue

            all_m = data["fixtures"].get("allMatches", [])
            digit_finished = []
            teams_in_season = set()

            for m in all_m:
                st = m.get("status", {})
                if not st.get("finished", False) or st.get("cancelled", False):
                    continue
                r_raw = str(m.get("roundName", m.get("round", "")))
                if not r_raw.isdigit():
                    continue
                hid = str(m.get("home", {}).get("id", ""))
                aid = str(m.get("away", {}).get("id", ""))
                if not hid or not aid:
                    continue
                digit_finished.append(m)
                teams_in_season.add(hid)
                teams_in_season.add(aid)

            num_teams = len(teams_in_season)
            max_reg_rounds = (num_teams - 1) * 2 if num_teams >= 10 else 46
            max_reg_matches = num_teams * (num_teams - 1)

            # Ordina cronologicamente ed esclude i play-off post-stagione (es. Belgio round 1..10 ad aprile)
            digit_finished.sort(key=lambda m: ((m.get("status") or {}).get("utcTime") or "", int(m.get("id", 0))))

            season_added = 0
            for m in digit_finished[:max_reg_matches]:
                r_num = int(str(m.get("roundName", m.get("round", ""))))
                if r_num > max_reg_rounds:
                    continue
                mid = str(m.get("id"))
                if mid in seen_ids:
                    continue
                st = m.get("status", {})
                utc_time = st.get("utcTime") or m.get("time", {}).get("utcTime", "")
                score_str = str(st.get("scoreStr", "")).replace("\u200e", "").strip()
                parts = score_str.split(" - ")
                if len(parts) != 2 or not utc_time:
                    continue
                try:
                    hg = int(parts[0].strip())
                    ag = int(parts[1].strip())
                except Exception:
                    continue

                seen_ids.add(mid)
                fixtures.append({
                    "mid": mid,
                    "league": lcode,
                    "season": season,
                    "utc_time": utc_time,
                    "date": utc_time[:10],
                    "round": r_num,
                    "max_rounds": max_reg_rounds,
                    "num_teams": num_teams,
                    "home_id": str(m["home"]["id"]),
                    "home_name": m["home"]["name"].replace(",", "").strip(),
                    "away_id": str(m["away"]["id"]),
                    "away_name": m["away"]["name"].replace(",", "").strip(),
                    "hg": hg,
                    "ag": ag,
                })
                season_added += 1

            print(f"  [{lcode}] {season}: {season_added} match regular season (squadre={num_teams}, max_round={max_reg_rounds})")
            time.sleep(0.15)

    return fixtures


def parse_single_match_stats(mid: str) -> Dict[str, Any]:
    """Scarica ed estrae xG, tiri, tiri in porta, corner e gialli da FotMob matchDetails."""
    time.sleep(random.uniform(0.05, 0.14))
    u = f"https://www.fotmob.com/api/data/matchDetails?matchId={mid}"
    md = fetch_json_retry(u)
    if not md:
        return {"mid": mid, "ok": False}

    stats_list = (((md.get("content") or {}).get("stats") or {}).get("Periods") or {}).get("All", {}).get("stats") or []
    res = {
        "mid": mid,
        "ok": True,
        "has_xg": False,
        "xg_h": None, "xg_a": None,
        "shots_h": None, "shots_a": None,
        "sot_h": None, "sot_a": None,
        "corners_h": None, "corners_a": None,
        "cards_h": 0, "cards_a": 0,
    }

    for grp in stats_list:
        for item in grp.get("stats", []):
            title = item.get("title", "")
            vals = item.get("stats", [])
            if len(vals) != 2 or vals[0] is None or vals[1] is None:
                continue
            try:
                if "Expected goals" in title:
                    res["xg_h"] = float(vals[0])
                    res["xg_a"] = float(vals[1])
                    res["has_xg"] = True
                elif title == "Total shots":
                    res["shots_h"] = int(vals[0])
                    res["shots_a"] = int(vals[1])
                elif title == "Shots on target":
                    res["sot_h"] = int(vals[0])
                    res["sot_a"] = int(vals[1])
                elif title == "Corners":
                    res["corners_h"] = int(vals[0])
                    res["corners_a"] = int(vals[1])
                elif title == "Yellow cards":
                    res["cards_h"] = int(vals[0])
                    res["cards_a"] = int(vals[1])
            except Exception:
                pass

    return res


def load_cached_stats() -> Dict[str, Dict[str, Any]]:
    cache = {}
    if os.path.exists(STATS_JSONL):
        with open(STATS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("ok"):
                        cache[obj["mid"]] = obj
                except Exception:
                    pass
    return cache


def download_all_stats(fixtures: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = load_cached_stats()
    todo = [f["mid"] for f in fixtures if f["mid"] not in cache]
    print(f"\n[*] Tabellini in cache: {len(cache)} | Da scaricare: {len(todo)}")

    if not todo:
        return cache

    completed = 0
    xg_ok = sum(1 for v in cache.values() if v.get("has_xg"))
    t_start = time.time()

    with open(STATS_JSONL, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(parse_single_match_stats, mid): mid for mid in todo}
            for fut in as_completed(futures):
                rec = fut.result()
                completed += 1
                if rec.get("ok"):
                    cache[rec["mid"]] = rec
                    f_out.write(json.dumps(rec) + "\n")
                    if rec.get("has_xg"):
                        xg_ok += 1
                if completed % 250 == 0 or completed == len(todo):
                    f_out.flush()
                    elapsed = time.time() - t_start
                    rate = completed / max(elapsed, 0.1)
                    eta_min = (len(todo) - completed) / max(rate, 0.1) / 60.0
                    print(f"  [PROGRESS] {completed}/{len(todo)} scaricati | xG presenti: {xg_ok} | Velocita: {rate:.1f} match/s | ETA: {eta_min:.1f} min", flush=True)

    return cache


def weighted_avg(vals: List[float]) -> float:
    if not vals:
        return 0.0
    w = [5.0, 4.0, 3.0, 2.0, 1.0][:len(vals)]
    return sum(v * wt for v, wt in zip(vals, w)) / sum(w)


def compute_metrics(records: List[Dict[str, Any]], venue: Optional[str] = None, fallback: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    if venue:
        records = [r for r in records if r["venue"] == venue]
    if not records:
        return dict(fallback) if fallback else {k: 0.0 for k in ("pts", "xg", "xga", "gf", "ga", "shots", "sot", "corners", "cards")}
    s = records[:5]  # most recent first
    return {
        "pts": weighted_avg([r["pts"] for r in s]),
        "xg": weighted_avg([r["xg_for"] for r in s]),
        "xga": weighted_avg([r["xg_against"] for r in s]),
        "gf": weighted_avg([r["gf"] for r in s]),
        "ga": weighted_avg([r["ga"] for r in s]),
        "shots": weighted_avg([r["shots"] for r in s]),
        "sot": weighted_avg([r["sot"] for r in s]),
        "corners": weighted_avg([r["corners"] for r in s]),
        "cards": weighted_avg([r["cards"] for r in s]),
    }


def compute_h2h(h2h_list: List[Dict[str, Any]], home_id: str) -> Tuple[float, float, float, float, float, float, float, float]:
    if not h2h_list:
        return 1.3, 1.3, 1.4, 1.2, 1.35, 1.25, 1.5, 1.5
    recent = h2h_list[:5]
    h_pts, a_pts, h_gf, a_gf, h_xg, a_xg = [], [], [], [], [], []
    v_pts, v_gf = [], []
    for m in recent:
        if m["home_id"] == home_id:
            hp = 3 if m["hg"] > m["ag"] else (1 if m["hg"] == m["ag"] else 0)
            ap = 3 if m["ag"] > m["hg"] else (1 if m["hg"] == m["ag"] else 0)
            h_pts.append(hp); a_pts.append(ap); h_gf.append(m["hg"]); a_gf.append(m["ag"])
            h_xg.append(m["xg_h"]); a_xg.append(m["xg_a"])
            v_pts.append(hp); v_gf.append(m["hg"])
        else:
            hp = 3 if m["ag"] > m["hg"] else (1 if m["hg"] == m["ag"] else 0)
            ap = 3 if m["hg"] > m["ag"] else (1 if m["hg"] == m["ag"] else 0)
            h_pts.append(hp); a_pts.append(ap); h_gf.append(m["ag"]); a_gf.append(m["hg"])
            h_xg.append(m["xg_a"]); a_xg.append(m["xg_h"])
    return (
        round(sum(h_pts) / len(h_pts), 4),
        round(sum(a_pts) / len(a_pts), 4),
        round(sum(h_gf) / len(h_gf), 4),
        round(sum(a_gf) / len(a_gf), 4),
        round(sum(h_xg) / len(h_xg), 4),
        round(sum(a_xg) / len(a_xg), 4),
        round(sum(v_pts) / len(v_pts), 4) if v_pts else 1.5,
        round(sum(v_gf) / len(v_gf), 4) if v_gf else 1.5,
    )


def build_chronological_csv(fixtures: List[Dict[str, Any]], stats_cache: Dict[str, Dict[str, Any]]) -> int:
    fixtures_sorted = sorted(fixtures, key=lambda x: (x["utc_time"], x["round"], int(x["mid"])))

    elo_map: Dict[str, float] = {}
    team_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  # most recent at index 0
    h2h_history: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    standings: Dict[Tuple[str, str], Dict[str, Dict[str, int]]] = defaultdict(dict)  # (league, season) -> tid -> {pts, gd, gf}

    rows_written = 0
    skipped_no_xg = 0
    skipped_warmup = 0

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f_csv:
        f_csv.write(",".join(CSV_HEADER) + "\n")

        for fix in fixtures_sorted:
            mid = fix["mid"]
            st = stats_cache.get(mid)
            lcode = fix["league"]
            season = fix["season"]
            conf = TIER2_CONFIG[lcode]
            hid, aid = fix["home_id"], fix["away_id"]

            # Initialize Elo & standings before match
            if hid not in elo_map:
                elo_map[hid] = conf["init_elo"]
            if aid not in elo_map:
                elo_map[aid] = conf["init_elo"]

            tbl = standings[(lcode, season)]
            if hid not in tbl:
                tbl[hid] = {"pts": 0, "gd": 0, "gf": 0, "played": 0}
            if aid not in tbl:
                tbl[aid] = {"pts": 0, "gd": 0, "gf": 0, "played": 0}

            h_elo_pre = round(elo_map[hid], 2)
            a_elo_pre = round(elo_map[aid], 2)

            has_valid_stats = (
                st is not None
                and st.get("has_xg")
                and st.get("xg_h") is not None
                and st.get("xg_a") is not None
                and st.get("shots_h") is not None
                and st.get("sot_h") is not None
                and st.get("corners_h") is not None
            )

            # Compute standings ranking snapshot BEFORE this match
            sorted_tbl = sorted(tbl.items(), key=lambda kv: (-kv[1]["pts"], -kv[1]["gd"], -kv[1]["gf"], kv[0]))
            pos_map = {tid: idx + 1 for idx, (tid, _) in enumerate(sorted_tbl)}
            pts_list = [v["pts"] for _, v in sorted_tbl]
            ucl_pts = pts_list[min(conf["ucl"] - 1, len(pts_list) - 1)] if pts_list else 0
            uel_pts = pts_list[min(conf["uel"] - 1, len(pts_list) - 1)] if pts_list else 0
            rel_idx = min(conf["rel"] - 1, len(pts_list) - 1) if pts_list else 0
            rel_pts = pts_list[rel_idx] if pts_list else 0

            h_hist = team_history[hid]
            a_hist = team_history[aid]

            if not has_valid_stats:
                skipped_no_xg += 1
            elif len(h_hist) < 3 or len(a_hist) < 3:
                skipped_warmup += 1
            else:
                cur_dt = datetime.strptime(fix["date"], "%Y-%m-%d").date()

                h_gen = compute_metrics(h_hist)
                a_gen = compute_metrics(a_hist)
                h_split = compute_metrics(h_hist, venue="H", fallback=h_gen)
                a_split = compute_metrics(a_hist, venue="A", fallback=a_gen)

                pair_key = tuple(sorted((hid, aid)))
                h2h_vals = compute_h2h(h2h_history[pair_key], hid)

                h_rest = min(14.0, max(1.0, float((cur_dt - h_hist[0]["date_obj"]).days)))
                a_rest = min(14.0, max(1.0, float((cur_dt - a_hist[0]["date_obj"]).days)))
                h_14d = float(sum(1 for r in h_hist if 0 < (cur_dt - r["date_obj"]).days <= 14))
                a_14d = float(sum(1 for r in a_hist if 0 < (cur_dt - r["date_obj"]).days <= 14))

                h_pos = pos_map[hid]
                a_pos = pos_map[aid]
                h_pts_tot = tbl[hid]["pts"]
                a_pts_tot = tbl[aid]["pts"]

                res_1x2 = "1" if fix["hg"] > fix["ag"] else ("X" if fix["hg"] == fix["ag"] else "2")
                giornata = fix["round"]
                meta_stag = 1 if giornata <= (fix["max_rounds"] / 2.0) else 2

                row = [
                    lcode, fix["date"], fix["home_name"], fix["away_name"],
                    f"{h_elo_pre:.2f}", f"{a_elo_pre:.2f}",
                    str(fix["hg"]), str(fix["ag"]), res_1x2,
                    f"{st['xg_h']:.2f}", f"{st['xg_a']:.2f}",
                    str(h_pos), str(a_pos), str(h_pts_tot), str(a_pts_tot),
                    repr(h_gen["pts"]), repr(a_gen["pts"]), repr(h_split["pts"]), repr(a_split["pts"]),
                    repr(h_gen["xg"]), repr(a_gen["xg"]), repr(h_split["xg"]), repr(a_split["xg"]),
                    repr(h_gen["xga"]), repr(a_gen["xga"]), repr(h_split["xga"]), repr(a_split["xga"]),
                    repr(h_gen["gf"]), repr(a_gen["gf"]), repr(h_split["gf"]), repr(a_split["gf"]),
                    repr(h_gen["ga"]), repr(a_gen["ga"]), repr(h_split["ga"]), repr(a_split["ga"]),
                    repr(h_gen["xg"] - h_gen["gf"]), repr(a_gen["xg"] - a_gen["gf"]),
                    *[repr(x) for x in h2h_vals],
                    repr(h_rest), repr(a_rest), str(int(h_14d)), str(int(a_14d)),
                    "0",  # is_derby (col 49, saltata da train_and_export.py)
                    str(h_pts_tot - ucl_pts), str(a_pts_tot - ucl_pts),
                    str(h_pts_tot - uel_pts), str(a_pts_tot - uel_pts),
                    str(h_pts_tot - rel_pts), str(a_pts_tot - rel_pts),
                    str(giornata), str(meta_stag),
                    repr(h_gen["shots"]), repr(a_gen["shots"]),
                    repr(h_gen["sot"]), repr(a_gen["sot"]),
                    repr(h_gen["corners"]), repr(a_gen["corners"]),
                    repr(h_gen["cards"]), repr(a_gen["cards"]),
                ]
                assert len(row) == 66, f"Attese 66 colonne, trovate {len(row)}"
                f_csv.write(",".join(row) + "\n")
                rows_written += 1

            # --- Post-match chronological updates (ZERO data leakage) ---
            hg, ag = fix["hg"], fix["ag"]
            hp = 3 if hg > ag else (1 if hg == ag else 0)
            ap = 3 if ag > hg else (1 if hg == ag else 0)

            tbl[hid]["pts"] += hp; tbl[hid]["gd"] += (hg - ag); tbl[hid]["gf"] += hg; tbl[hid]["played"] += 1
            tbl[aid]["pts"] += ap; tbl[aid]["gd"] += (ag - hg); tbl[aid]["gf"] += ag; tbl[aid]["played"] += 1

            dr = elo_map[hid] + 75.0 - elo_map[aid]
            e_home = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
            s_home = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            k_dyn = 20.0 * (max(1, abs(hg - ag)) ** 0.5)
            delta_elo = k_dyn * (s_home - e_home)
            elo_map[hid] += delta_elo
            elo_map[aid] -= delta_elo

            if has_valid_stats:
                cur_dt = datetime.strptime(fix["date"], "%Y-%m-%d").date()
                team_history[hid].insert(0, {
                    "date_obj": cur_dt, "venue": "H", "gf": hg, "ga": ag, "pts": hp,
                    "xg_for": st["xg_h"], "xg_against": st["xg_a"],
                    "shots": st["shots_h"], "sot": st["sot_h"], "corners": st["corners_h"], "cards": st["cards_h"]
                })
                team_history[aid].insert(0, {
                    "date_obj": cur_dt, "venue": "A", "gf": ag, "ga": hg, "pts": ap,
                    "xg_for": st["xg_a"], "xg_against": st["xg_h"],
                    "shots": st["shots_a"], "sot": st["sot_a"], "corners": st["corners_a"], "cards": st["cards_a"]
                })
                pair_key = tuple(sorted((hid, aid)))
                h2h_history[pair_key].insert(0, {
                    "home_id": hid, "away_id": aid, "hg": hg, "ag": ag,
                    "xg_h": st["xg_h"], "xg_a": st["xg_a"]
                })

    print(f"\n=== DATASET TIER-2 COMPLETATO ===")
    print(f"File salvato: {OUT_CSV}")
    print(f"Righe valide scritte (100% Opta xG + 56 feature): {rows_written}")
    print(f"Scartate per assenza xG: {skipped_no_xg} | Scartate per warmup (<3 match): {skipped_warmup}")
    return rows_written


def main():
    print("=================================================================")
    print("   DEEP-PITCH TIER-2 DATASET BUILDER (66 COLUMNS / 56 FEATURES)")
    print("=================================================================")
    fixtures = discover_regular_fixtures()
    print(f"\n[*] Totale match regular season scoperti: {len(fixtures)}")
    cache = download_all_stats(fixtures)
    build_chronological_csv(fixtures, cache)


if __name__ == "__main__":
    main()
