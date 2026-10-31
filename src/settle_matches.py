#!/usr/bin/env python3
"""
src/settle_matches.py
Automated Natural Match Settlement & Bankroll Compounding Engine for Deep-Pitch.

Settles matches that concluded at full-time (90'+ FT) without a Green Book exit.
Evaluates final 1X2 result, determines WIN / LOSS against pre-match bet, calculates
exact P&L, re-chains cumulative running bankroll sequentially, and syncs to GitHub.
"""

import os
import sys
import csv
import json
import argparse
import subprocess
from typing import List, Dict, Any, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKING_CSV = os.path.join(BASE_DIR, "tracking_2026_2027.csv")
FEED_JSON = os.path.abspath(os.path.join(BASE_DIR, "..", "live-sentinel", "live_feed.json"))


def parse_profit(val: str) -> float:
    """Safely parses profit string with optional +, -, or quotes."""
    if not val:
        return 0.0
    s = val.strip().replace('"', '').replace('+', '').replace(' ', '')
    try:
        return float(s)
    except ValueError:
        return 0.0


def evaluate_bet_outcome(bet: str, home_score: int, away_score: int) -> Tuple[str, bool]:
    """
    Returns (actual_result_str, is_bet_won).
    actual_result_str: '1', 'X', or '2'
    """
    if home_score > away_score:
        actual = "1"
    elif home_score == away_score:
        actual = "X"
    else:
        actual = "2"

    bet_clean = bet.strip()
    won = False
    if bet_clean == "1":
        won = (home_score > away_score)
    elif bet_clean == "X":
        won = (home_score == away_score)
    elif bet_clean == "2":
        won = (home_score < away_score)
    elif bet_clean == "1X":
        won = (home_score >= away_score)
    elif bet_clean == "X2":
        won = (home_score <= away_score)
    elif bet_clean == "12":
        won = (home_score != away_score)
    elif bet_clean == "Over 2.5":
        won = ((home_score + away_score) >= 3)
    elif bet_clean == "Under 2.5":
        won = ((home_score + away_score) < 3)
    else:
        won = (actual == bet_clean)

    return actual, won


def rechain_bankroll(rows: List[List[str]]) -> None:
    """
    Recomputes running bankroll sequentially from baseline.
    Preserves historical frozen baseline, then adds profit sequentially.
    """
    last_frozen_idx = 0
    running_bankroll = 1000.0

    # Step 1: Find frozen baseline row before live managed rows
    for i in range(1, len(rows)):
        r = rows[i]
        result = r[9].strip() if len(r) > 9 else ""
        bankroll_str = r[11].strip() if len(r) > 11 else ""
        if not result or not bankroll_str or result in ("GREENBOOK", "REDBOOK"):
            break
        try:
            running_bankroll = float(bankroll_str)
            last_frozen_idx = i
        except ValueError:
            pass

    # Step 2: Chain sequentially from last_frozen_idx + 1
    for i in range(last_frozen_idx + 1, len(rows)):
        r = rows[i]
        while len(r) < 12:
            r.append("")

        prof_str = r[10].strip() if len(r) > 10 else ""
        if prof_str:
            profit = parse_profit(prof_str)
            running_bankroll = round(running_bankroll + profit, 2)
            r[11] = f"{running_bankroll:.2f}"
        else:
            r[11] = ""


def load_live_feed() -> List[Dict[str, Any]]:
    """Loads match ticks from live-sentinel live_feed.json if available."""
    if not os.path.exists(FEED_JSON):
        return []
    try:
        with open(FEED_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def settle_open_matches(
    csv_path: str = TRACKING_CSV,
    manual_match: Optional[str] = None,
    manual_score: Optional[str] = None,
    dry_run: bool = False
) -> int:
    """
    Scans tracking CSV for open matches and settles them.
    Can use live_feed.json or manual CLI input.
    """
    if not os.path.exists(csv_path):
        print(f"[ERROR] File non trovato: {csv_path}")
        return 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [list(r) for r in reader]

    if len(rows) <= 1:
        print("[INFO] Nessun record trovato nel file CSV.")
        return 0

    feed_ticks = load_live_feed()
    settled_count = 0
    settled_details = []

    for idx in range(1, len(rows)):
        r = rows[idx]
        while len(r) < 12:
            r.append("")

        result_col = r[9].strip()
        if result_col:
            continue  # Already settled or hedged

        date_col = r[0].strip()
        league_col = r[1].strip()
        match_col = r[2].strip()
        bet_col = r[3].strip()
        odds_val = float(r[4].strip()) if r[4].strip() else 2.0
        stake_val = float(r[8].strip()) if r[8].strip() else 15.0

        # Check manual score input
        home_score = None
        away_score = None

        if manual_match and manual_score:
            m_norm = manual_match.lower().replace(" ", "").replace("-", "")
            row_norm = match_col.lower().replace(" ", "").replace("-", "")
            if m_norm in row_norm or row_norm in m_norm:
                parts = manual_score.strip().split("-")
                if len(parts) == 2:
                    try:
                        home_score = int(parts[0].strip())
                        away_score = int(parts[1].strip())
                    except ValueError:
                        pass

        # Check feed ticks
        if home_score is None and feed_ticks:
            teams = match_col.split("-")
            if len(teams) == 2:
                h_name = teams[0].strip().lower()
                a_name = teams[1].strip().lower()
                for tick in feed_ticks:
                    th = str(tick.get("home", "")).strip().lower()
                    ta = str(tick.get("away", "")).strip().lower()
                    if (h_name in th or th in h_name) and (a_name in ta or ta in a_name):
                        if tick.get("finished", False):
                            home_score = int(tick.get("home_score", 0))
                            away_score = int(tick.get("away_score", 0))
                            break

        if home_score is not None and away_score is not None:
            actual_res, won = evaluate_bet_outcome(bet_col, home_score, away_score)
            if won:
                profit = round(stake_val * (odds_val - 1.0), 2)
                prof_str = f"+{profit:.2f}"
            else:
                profit = round(-stake_val, 2)
                prof_str = f"{profit:.2f}"

            r[9] = actual_res
            r[10] = prof_str
            settled_count += 1
            verdict = "VINTA" if won else "PERSA"
            settled_details.append(
                f"  • {match_col} ({home_score}-{away_score} FT): Esito {actual_res} | Pre {bet_col} @ {odds_val} | {verdict} ({prof_str} EUR)"
            )

    if settled_count == 0:
        print("[INFO] Nessun match da liquidare (tutti i match sono già chiusi o non ancora finiti nel feed).")
        return 0

    # Recompute bankroll
    rechain_bankroll(rows)

    print(f"\n[SETTLE] Liquidati con successo {settled_count} match:")
    for d in settled_details:
        print(d)

    # Print updated rows preview
    print("\n[PREVIEW] Righe aggiornate:")
    for idx in range(1, len(rows)):
        r = rows[idx]
        if r[9].strip():
            print(f"  Row {idx+1}: {r[0]} | {r[2]} | Bet: {r[3]} | Result: {r[9]} | Profit: {r[10]} | Bankroll: {r[11]}")

    if not dry_run:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"\n[OK] File aggiornato salvato con successo: {csv_path}")

        # Git commit & push
        try:
            subprocess.run(["git", "-C", BASE_DIR, "add", "-A"], check=True)
            msg = f"settle: natural settlement for {settled_count} matches"
            subprocess.run(["git", "-C", BASE_DIR, "commit", "-m", msg], check=True)
            subprocess.run(["git", "-C", BASE_DIR, "push", "origin", "main"], check=True)
            print("[OK] Sincronizzato con successo su GitHub origin main.")
        except Exception as e:
            print(f"[WARN] Impossibile eseguire git push automatico: {e}")
    else:
        print("\n[DRY RUN] Nessuna modifica scritta su disco.")

    return settled_count


def main():
    parser = argparse.ArgumentParser(description="Deep-Pitch Natural Match Settlement Engine")
    parser.add_argument("--match", type=str, help="Nome match (es. 'Liverpool-Arsenal' o 'Liverpool')")
    parser.add_argument("--score", type=str, help="Punteggio finale (es. '2-1')")
    parser.add_argument("--rechain", action="store_true", help="Ricalcola la catena del bankroll sull'intero file CSV")
    parser.add_argument("--dry-run", action="store_true", help="Mostra l'anteprima senza scrivere su file")
    args = parser.parse_args()

    if args.rechain:
        with open(TRACKING_CSV, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = [list(r) for r in reader]
        rechain_bankroll(rows)
        if not args.dry_run:
            with open(TRACKING_CSV, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            print(f"[OK] Bankroll ricalcolato e salvato in {TRACKING_CSV}")
        else:
            print("[DRY RUN] Re-chain bankroll completato.")
        return

    settle_open_matches(
        csv_path=TRACKING_CSV,
        manual_match=args.match,
        manual_score=args.score,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
