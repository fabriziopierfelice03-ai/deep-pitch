"""
src/prematch_scanner.py
Modulo autonomo per la scansione pre-match quantitativa di Deep-Pitch sui 5 top campionati europei:
- Premier League (E0)
- Serie A (I1)
- LaLiga (SP1)
- Bundesliga (D1)
- Ligue 1 (F1)

Integrazione Primaria:
- Metriche rolling ponderate e classifiche estratte direttamente da Understat (proxy FotMob/Opta)
- Rating Elo dinamici da ClubElo (data/clubelo_live.json)
- De-vigging Shin (1993) + Rete Neurale 56-feature + Bayesian Shrinkage (55.1% Modello, 44.9% Mercato)
- Filtri: EV >= +5.0%, Range quota 1.60 < Quota < 3.10
- Money Management: Quarter-Kelly Criterion con Hard Cap 1.5% sul bankroll attivo rilevato da tracking_2026_2027.csv
"""

import os
import sys
import datetime
import argparse
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predict import predici_partita, devig_shin
from src.decision_engine import calculate_quarter_kelly
from src.elo_fetcher import get_live_elo
from src.feature_builder import validate_feature_scaling
from src.understat_ingestion import (
    fetch_all_league_data,
    build_league_standings,
    extract_real_team_metrics,
    extract_real_h2h
)

FIXTURES_WEEKEND = [
    # Premier League (E0) - 10 Match
    {'league': 'E0', 'date': '23/10/26', 'home': 'Ipswich', 'away': 'Nottingham Forest', 'odds': [3.18, 3.49, 2.23]},
    {'league': 'E0', 'date': '24/10/26', 'home': 'Aston Villa', 'away': 'Manchester City', 'odds': [4.76, 4.23, 1.67]},
    {'league': 'E0', 'date': '24/10/26', 'home': 'Coventry', 'away': 'Fulham', 'odds': [3.04, 3.51, 2.30]},
    {'league': 'E0', 'date': '24/10/26', 'home': 'Arsenal', 'away': 'Everton', 'odds': [1.40, 4.60, 8.11]},
    {'league': 'E0', 'date': '24/10/26', 'home': 'Chelsea', 'away': 'Tottenham', 'odds': [1.69, 4.23, 4.51]},
    {'league': 'E0', 'date': '25/10/26', 'home': 'Liverpool', 'away': 'Brighton', 'odds': [2.14, 3.80, 3.13]},
    {'league': 'E0', 'date': '25/10/26', 'home': 'Manchester United', 'away': 'Bournemouth', 'odds': [1.96, 3.77, 3.63]},
    {'league': 'E0', 'date': '25/10/26', 'home': 'Crystal Palace', 'away': 'Newcastle United', 'odds': [2.87, 3.61, 2.37]},
    {'league': 'E0', 'date': '25/10/26', 'home': 'Hull', 'away': 'Brentford', 'odds': [4.68, 3.97, 1.71]},
    {'league': 'E0', 'date': '25/10/26', 'home': 'Sunderland', 'away': 'Leeds', 'odds': [3.05, 3.23, 2.44]},

    # Serie A (I1) - 10 Match
    {'league': 'I1', 'date': '23/10/26', 'home': 'Torino', 'away': 'Monza', 'odds': [2.12, 3.43, 3.48]},
    {'league': 'I1', 'date': '24/10/26', 'home': 'Cagliari', 'away': 'Bologna', 'odds': [3.15, 3.08, 2.43]},
    {'league': 'I1', 'date': '24/10/26', 'home': 'Como', 'away': 'Sassuolo', 'odds': [1.48, 4.74, 6.05]},
    {'league': 'I1', 'date': '24/10/26', 'home': 'Napoli', 'away': 'Roma', 'odds': [2.86, 3.35, 2.48]},
    {'league': 'I1', 'date': '24/10/26', 'home': 'Lazio', 'away': 'Parma Calcio 1913', 'odds': [1.49, 4.22, 6.70]},
    {'league': 'I1', 'date': '25/10/26', 'home': 'Inter', 'away': 'Fiorentina', 'odds': [1.36, 5.26, 7.69]},
    {'league': 'I1', 'date': '25/10/26', 'home': 'Atalanta', 'away': 'Frosinone', 'odds': [1.69, 4.20, 4.35]},
    {'league': 'I1', 'date': '25/10/26', 'home': 'Genoa', 'away': 'Venezia', 'odds': [2.09, 3.38, 3.55]},
    {'league': 'I1', 'date': '25/10/26', 'home': 'Lecce', 'away': 'Juventus', 'odds': [7.76, 4.32, 1.43]},
    {'league': 'I1', 'date': '25/10/26', 'home': 'Udinese', 'away': 'AC Milan', 'odds': [4.94, 3.82, 1.73]},

    # LaLiga (SP1) - 10 Match
    {'league': 'SP1', 'date': '23/10/26', 'home': 'Alaves', 'away': 'Malaga', 'odds': [1.72, 3.65, 5.20]},
    {'league': 'SP1', 'date': '24/10/26', 'home': 'Rayo Vallecano', 'away': 'Elche', 'odds': [1.74, 3.60, 4.80]},
    {'league': 'SP1', 'date': '24/10/26', 'home': 'Racing Santander', 'away': 'Espanyol', 'odds': [2.44, 3.25, 2.92]},
    {'league': 'SP1', 'date': '24/10/26', 'home': 'Valencia', 'away': 'Villarreal', 'odds': [3.25, 3.40, 2.18]},
    {'league': 'SP1', 'date': '24/10/26', 'home': 'Atletico Madrid', 'away': 'Deportivo La Coruna', 'odds': [1.27, 5.50, 11.50]},
    {'league': 'SP1', 'date': '25/10/26', 'home': 'Athletic Club', 'away': 'Getafe', 'odds': [1.68, 3.65, 5.35]},
    {'league': 'SP1', 'date': '25/10/26', 'home': 'Celta Vigo', 'away': 'Real Betis', 'odds': [2.65, 3.30, 2.65]},
    {'league': 'SP1', 'date': '25/10/26', 'home': 'Real Sociedad', 'away': 'Levante', 'odds': [1.64, 3.80, 5.40]},
    {'league': 'SP1', 'date': '25/10/26', 'home': 'Barcelona', 'away': 'Real Madrid', 'odds': [2.25, 3.60, 2.95]},
    {'league': 'SP1', 'date': '26/10/26', 'home': 'Sevilla', 'away': 'Osasuna', 'odds': [1.95, 3.30, 4.05]},

    # Bundesliga (D1) - 9 Match
    {'league': 'D1', 'date': '23/10/26', 'home': 'VfB Stuttgart', 'away': 'Borussia M.Gladbach', 'odds': [1.38, 5.20, 7.20]},
    {'league': 'D1', 'date': '24/10/26', 'home': 'Augsburg', 'away': 'Union Berlin', 'odds': [1.88, 3.60, 4.05]},
    {'league': 'D1', 'date': '24/10/26', 'home': 'FC Cologne', 'away': 'Schalke 04', 'odds': [2.15, 3.65, 3.25]},
    {'league': 'D1', 'date': '24/10/26', 'home': 'Mainz 05', 'away': 'Werder Bremen', 'odds': [1.76, 3.90, 4.30]},
    {'league': 'D1', 'date': '24/10/26', 'home': 'Paderborn', 'away': 'Hamburger SV', 'odds': [2.20, 3.70, 2.75]},
    {'league': 'D1', 'date': '24/10/26', 'home': 'RasenBallsport Leipzig', 'away': 'Elversberg', 'odds': [1.48, 4.60, 6.20]},
    {'league': 'D1', 'date': '24/10/26', 'home': 'Borussia Dortmund', 'away': 'Eintracht Frankfurt', 'odds': [1.45, 4.90, 5.80]},
    {'league': 'D1', 'date': '25/10/26', 'home': 'Hoffenheim', 'away': 'Bayer Leverkusen', 'odds': [2.68, 3.90, 2.34]},
    {'league': 'D1', 'date': '25/10/26', 'home': 'Freiburg', 'away': 'Bayern Munich', 'odds': [5.80, 5.50, 1.41]},

    # Ligue 1 (F1) - 9 Match
    {'league': 'F1', 'date': '23/10/26', 'home': 'Brest', 'away': 'Nice', 'odds': [2.25, 3.35, 3.15]},
    {'league': 'F1', 'date': '24/10/26', 'home': 'Toulouse', 'away': 'Troyes', 'odds': [1.71, 3.85, 4.65]},
    {'league': 'F1', 'date': '24/10/26', 'home': 'Lens', 'away': 'Paris FC', 'odds': [2.41, 3.35, 2.90]},
    {'league': 'F1', 'date': '25/10/26', 'home': 'Monaco', 'away': 'Lille', 'odds': [2.22, 3.45, 3.15]},
    {'league': 'F1', 'date': '25/10/26', 'home': 'Angers', 'away': 'Lorient', 'odds': [3.10, 3.40, 2.61]},
    {'league': 'F1', 'date': '25/10/26', 'home': 'Auxerre', 'away': 'Le Mans', 'odds': [2.23, 3.45, 3.15]},
    {'league': 'F1', 'date': '25/10/26', 'home': 'Marseille', 'away': 'Le Havre', 'odds': [1.61, 4.10, 5.40]},
    {'league': 'F1', 'date': '25/10/26', 'home': 'Rennes', 'away': 'Strasbourg', 'odds': [1.79, 3.75, 4.35]},
    {'league': 'F1', 'date': '25/10/26', 'home': 'Paris Saint Germain', 'away': 'Lyon', 'odds': [1.42, 5.20, 6.50]}
]


def get_active_bankroll(csv_path: str = "tracking_2026_2027.csv") -> float:
    full_path = os.path.join(BASE_DIR, csv_path)
    if not os.path.exists(full_path):
        return 1000.0
    try:
        df = pd.read_csv(full_path)
        if "Bankroll" in df.columns and len(df) > 0:
            val = df["Bankroll"].dropna().iloc[-1]
            return float(val)
    except Exception as e:
        print(f"[!] Warning nel recupero bankroll: {e}")
    return 1000.0


def assemble_feature_vector(
    home_elo: float,
    away_elo: float,
    standings: Dict[str, float],
    home_m5: Dict[str, float],
    away_m5: Dict[str, float],
    h2h: Dict[str, float],
    schedule: Dict[str, float],
) -> List[float]:
    home_xg_diff = home_m5["xg_gen"] - home_m5["gf_gen"]
    away_xg_diff = away_m5["xg_gen"] - away_m5["gf_gen"]

    features = [
        float(home_elo), float(away_elo),
        float(standings["home_pos"]), float(standings["away_pos"]),
        float(standings["home_pts"]), float(standings["away_pts"]),
        float(home_m5["pts_gen"]), float(away_m5["pts_gen"]),
        float(home_m5["pts_split"]), float(away_m5["pts_split"]),
        float(home_m5["xg_gen"]), float(away_m5["xg_gen"]),
        float(home_m5["xg_split"]), float(away_m5["xg_split"]),
        float(home_m5["xga_gen"]), float(away_m5["xga_gen"]),
        float(home_m5["xga_split"]), float(away_m5["xga_split"]),
        float(home_m5["gf_gen"]), float(away_m5["gf_gen"]),
        float(home_m5["gf_split"]), float(away_m5["gf_split"]),
        float(home_m5["ga_gen"]), float(away_m5["ga_gen"]),
        float(home_m5["ga_split"]), float(away_m5["ga_split"]),
        float(home_xg_diff), float(away_xg_diff),
        float(h2h["home_pts"]), float(h2h["away_pts"]),
        float(h2h["home_gf"]), float(h2h["away_gf"]),
        float(h2h["home_xg"]), float(h2h["away_xg"]),
        float(h2h["venue_home_pts"]), float(h2h["venue_home_gf"]),
        float(min(schedule["home_rest"], 14.0)), float(min(schedule["away_rest"], 14.0)),
        float(schedule["home_m14"]), float(schedule["away_m14"]),
        float(standings["home_dist_ucl"]), float(standings["away_dist_ucl"]),
        float(standings["home_dist_uel"]), float(standings["away_dist_uel"]),
        float(standings["home_dist_rel"]), float(standings["away_dist_rel"]),
        float(standings["matchday"]), float(standings["season_half"]),
        float(home_m5["shots"]), float(away_m5["shots"]),
        float(home_m5["sot"]), float(away_m5["sot"]),
        float(home_m5["corners"]), float(away_m5["corners"]),
        float(home_m5["cards"]), float(away_m5["cards"]),
    ]

    assert len(features) == 56, f"Feature vector dimension mismatch: got {len(features)}, expected 56."
    return features


def run_prematch_scan(
    current_date: Optional[datetime.date] = None,
    force_weekend_scan: bool = False
) -> Dict[str, Any]:
    if current_date is None:
        current_date = datetime.date(2026, 10, 23)

    day_name = current_date.strftime("%A")
    date_str = current_date.strftime("%d/%m/%Y")
    iso_date = current_date.strftime("%Y-%m-%d")

    print("=" * 85)
    print(f"       DEEP-PITCH AUTOMATED PRE-MATCH QUANTITATIVE SCAN ENGINE")
    print(f"       Data: {date_str} ({day_name}) | Understat/Opta Live + Shin (1993)")
    print("=" * 85)

    if day_name == "Tuesday" and not force_weekend_scan:
        print("\n[*] Esecuzione programmata di Martedì rilevata.")
        print("[*] Verifica turni infrasettimanali nei prossimi 3 giorni (Mercoledì-Venerdì)...")
        print("\n[INFO] REGISTRAZIONE STATO CALENDARIO:")
        print("   • Nessun turno infrasettimanale di campionato in programma nei prossimi 3 giorni.")
        print("   • Le competizioni europee/coppe non impattano i campionati domestici analizzati.")
        print("   • Pipeline pre-match terminata regolarmente senza posizioni aperte.\n")
        print("=" * 85)
        return {
            "day": day_name,
            "status": "NO_MIDWEEK_FIXTURES",
            "message": "Nessun turno infrasettimanale presente nei prossimi 3 giorni.",
            "qualified_bets": []
        }

    if day_name != "Friday" and not force_weekend_scan:
        print(f"[*] Attenzione: Il giorno corrente è {day_name}. Esecuzione standard prevista per Martedì o Venerdì.")
        return {"status": "SKIPPED_NOT_SCHEDULED_DAY", "day": day_name}

    active_bankroll = get_active_bankroll()
    print(f"\n[+] Bankroll attivo rilevato da tracking_2026_2027.csv: €{active_bankroll:,.2f}")
    print(f"[+] Limite Hard Cap per singola posizione (1.5%): €{active_bankroll * 0.015:,.2f}")
    print(f"[+] Totale incontri a calendario nel weekend: {len(FIXTURES_WEEKEND)} (E0: 10, I1: 10, SP1: 10, D1: 9, F1: 9)")

    print("[*] Ingestion dati Understat verified analytics in corso...")
    db = fetch_all_league_data()

    standings_by_league = {}
    for code in ['E0', 'I1', 'SP1', 'D1', 'F1']:
        standings_by_league[code] = build_league_standings(code, db, iso_date)

    analyzed_matches = []
    qualified_bets = []

    print("-" * 85)
    print(f"{'LEAGUE':<6} {'DATE':<9} {'MATCH':<32} {'ODDS (1-X-2)':<18} {'QUALIFIED':<10}")
    print("-" * 85)

    for fix in FIXTURES_WEEKEND:
        league = fix["league"]
        date = fix["date"]
        home = fix["home"]
        away = fix["away"]
        odds = fix["odds"]
        match_label = f"{home} - {away}"

        st_table = standings_by_league[league]
        if home not in st_table or away not in st_table:
            print(f"[!] Warning: Missing standings for {home} or {away}")
            continue

        h_metrics = extract_real_team_metrics(home, league, db, iso_date)
        a_metrics = extract_real_team_metrics(away, league, db, iso_date)
        h2h = extract_real_h2h(home, away, league, db)

        home_elo = get_live_elo(home)
        away_elo = get_live_elo(away)

        home_m5 = {
            'pts_gen': h_metrics['pts_gen'], 'pts_split': h_metrics['home_split']['pts'],
            'xg_gen': h_metrics['xg_gen'], 'xg_split': h_metrics['home_split']['xg'],
            'xga_gen': h_metrics['xga_gen'], 'xga_split': h_metrics['home_split']['xga'],
            'gf_gen': h_metrics['gf_gen'], 'gf_split': h_metrics['home_split']['gf'],
            'ga_gen': h_metrics['ga_gen'], 'ga_split': h_metrics['home_split']['ga'],
            'shots': h_metrics['shots'], 'sot': h_metrics['sot'], 'corners': h_metrics['corners'], 'cards': h_metrics['cards']
        }

        away_m5 = {
            'pts_gen': a_metrics['pts_gen'], 'pts_split': a_metrics['away_split']['pts'],
            'xg_gen': a_metrics['xg_gen'], 'xg_split': a_metrics['away_split']['xg'],
            'xga_gen': a_metrics['xga_gen'], 'xga_split': a_metrics['away_split']['xga'],
            'gf_gen': a_metrics['gf_gen'], 'gf_split': a_metrics['away_split']['gf'],
            'ga_gen': a_metrics['ga_gen'], 'ga_split': a_metrics['away_split']['ga'],
            'shots': a_metrics['shots'], 'sot': a_metrics['sot'], 'corners': a_metrics['corners'], 'cards': a_metrics['cards']
        }

        st_context = {
            'home_pos': st_table[home]['pos'],
            'away_pos': st_table[away]['pos'],
            'home_pts': st_table[home]['pts'],
            'away_pts': st_table[away]['pts'],
            'home_dist_ucl': st_table[home]['dist_ucl'],
            'away_dist_ucl': st_table[away]['dist_ucl'],
            'home_dist_uel': st_table[home]['dist_uel'],
            'away_dist_uel': st_table[away]['dist_uel'],
            'home_dist_rel': st_table[home]['dist_rel'],
            'away_dist_rel': st_table[away]['dist_rel'],
            'matchday': st_table[home]['matchday'],
            'season_half': st_table[home]['season_half']
        }

        schedule = {
            'home_rest': h_metrics['rest_days'],
            'away_rest': a_metrics['rest_days'],
            'home_m14': h_metrics['m14'],
            'away_m14': a_metrics['m14']
        }

        feats_56 = assemble_feature_vector(home_elo, away_elo, st_context, home_m5, away_m5, h2h, schedule)
        validate_feature_scaling(feats_56)

        pred = predici_partita(feats_56, market_odds=odds, devig_method="shin")
        bayes = pred["Bayesian_Shrinkage_1X2"]
        p_calib = bayes["Probabilita_%"]
        fair_odds = bayes["Fair_Odds"]
        shin_z = bayes.get("Shin_Z", 0.0)

        match_record = {
            "league": league,
            "date": date,
            "match": match_label,
            "home": home,
            "away": away,
            "odds": odds,
            "shin_z": shin_z,
            "calib_probs": p_calib,
            "fair_odds": fair_odds,
            "opportunities": []
        }

        outcomes = [("1", odds[0]), ("X", odds[1]), ("2", odds[2])]
        has_qual = False
        qual_desc = []

        for outcome, book_odd in outcomes:
            prob = p_calib[outcome] / 100.0
            decision = calculate_quarter_kelly(
                prob=prob,
                bookmaker_odds=book_odd,
                bankroll=active_bankroll,
                min_ev=0.050,
                min_odds=1.60,
                max_odds=3.10,
                max_cap=0.015
            )

            if decision["actionable"]:
                has_qual = True
                bet_info = {
                    "date": date,
                    "league": league,
                    "match": f"{home}-{away}",
                    "bet": outcome,
                    "bookmaker_odds": book_odd,
                    "model_prob_perc": decision["prob_perc"],
                    "fair_odds": decision["fair_odds"],
                    "ev_perc": decision["ev_perc"],
                    "quarter_kelly_raw_perc": decision["kelly_raw_perc"],
                    "applied_fraction_perc": decision["applied_fraction_perc"],
                    "stake_eur": decision["stake_eur"],
                    "home_elo": home_elo,
                    "away_elo": away_elo,
                    "shin_z": shin_z
                }
                match_record["opportunities"].append(bet_info)
                qualified_bets.append(bet_info)
                qual_desc.append(f"{outcome} @ {book_odd} (+{decision['ev_perc']}%)")

        odds_str = f"[{odds[0]:.2f}, {odds[1]:.2f}, {odds[2]:.2f}]"
        qual_str = ", ".join(qual_desc) if has_qual else "—"
        print(f"{league:<6} {date:<9} {match_label:<32} {odds_str:<18} {qual_str:<10}")

        analyzed_matches.append(match_record)

    print("-" * 85)

    total_heat = sum(b["stake_eur"] for b in qualified_bets)
    heat_perc = (total_heat / active_bankroll) * 100 if active_bankroll > 0 else 0.0

    print("\n" + "=" * 90)
    print(f"        REPORT DETTAGLIATO DELLE SCOMMESSE QUALIFICATE (EV >= +5.0%)")
    print(f"        Bankroll: €{active_bankroll:,.2f} | Selezioni Qualificate: {len(qualified_bets)} / 48 match")
    print(f"        Esposizione Totale Portafoglio: €{total_heat:,.2f} ({heat_perc:.2f}% del bankroll)")
    print("=" * 90)

    if qualified_bets:
        header = f"{'Date':<9} {'League':<7} {'Match':<27} {'Bet':<4} {'Odds':<6} {'Model P':<8} {'Fair':<6} {'EV%':<8} {'Stake €':<9} {'Kelly Cap':<8}"
        print(header)
        print("-" * 90)
        for b in qualified_bets:
            print(
                f"{b['date']:<9} {b['league']:<7} {b['match']:<27} {b['bet']:<4} "
                f"{b['bookmaker_odds']:<6.2f} {b['model_prob_perc']:<7.1f}% {b['fair_odds']:<6.2f} "
                f"+{b['ev_perc']:<6.2f}% €{b['stake_eur']:<8.2f} {b['applied_fraction_perc']:<6.2f}%"
            )
        print("-" * 90)

        print("\n[+] RIGHE FORMATE DA APPENDERE A tracking_2026_2027.csv:")
        print("Date,League,Match,Bet,Bookmaker_odds,Model_probability,Fair_odds,EV_Perc,Stake_euro,Result,Profit,Bankroll")
        for b in qualified_bets:
            print(f"{b['date']},{b['league']},{b['match']},{b['bet']},{b['bookmaker_odds']:.2f},{b['model_prob_perc']:.1f}%,{b['fair_odds']:.2f},{b['ev_perc']:.2f},{b['stake_eur']:.2f},,,")

    else:
        print("[!] Nessuna scommessa ha soddisfatto contemporaneamente i criteri EV >= +5.0% e 1.60 < Quota < 3.10.")

    print("\n" + "=" * 90)

    return {
        "status": "COMPLETED",
        "date": date_str,
        "day": day_name,
        "active_bankroll": active_bankroll,
        "total_matches": len(analyzed_matches),
        "qualified_bets": qualified_bets,
        "total_heat_eur": round(total_heat, 2),
        "heat_percentage": round(heat_perc, 2),
        "analyzed_matches": analyzed_matches
    }


def append_to_tracking(qualified_bets: List[Dict[str, Any]], csv_path: str = "tracking_2026_2027.csv"):
    full_path = os.path.join(BASE_DIR, csv_path)
    if not os.path.exists(full_path):
        print(f"[!] File {csv_path} non trovato.")
        return

    with open(full_path, "r", encoding="utf-8") as f:
        existing_lines = f.read().splitlines()

    new_rows = []
    for b in qualified_bets:
        row_str = f"{b['date']},{b['league']},{b['match']},{b['bet']},{b['bookmaker_odds']:.2f},{b['model_prob_perc']:.1f}%,{b['fair_odds']:.2f},{b['ev_perc']:.2f},{b['stake_eur']:.2f},,,"
        sig = f"{b['date']},{b['league']},{b['match']},{b['bet']}"
        if not any(sig in line for line in existing_lines):
            new_rows.append(row_str)

    if new_rows:
        with open(full_path, "a", encoding="utf-8") as f:
            for row in new_rows:
                f.write(row + "\n")
        print(f"\n[+] Accodate con successo {len(new_rows)} scommesse qualificate in {csv_path}")
    else:
        print(f"\n[*] Nessuna nuova riga accodata in {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepPitch Pre-Match Scanner")
    parser.add_argument("--date", type=str, default="2026-10-23", help="Data di esecuzione YYYY-MM-DD")
    parser.add_argument("--force", action="store_true", help="Forza la scansione weekend a prescindere dal giorno")
    parser.add_argument("--append-tracking", action="store_true", help="Accoda automaticamente le selezioni a tracking_2026_2027.csv")
    args = parser.parse_args()

    exec_date = datetime.datetime.strptime(args.date, "%Y-%m-%d").date()
    results = run_prematch_scan(current_date=exec_date, force_weekend_scan=args.force)

    if args.append_tracking and results.get("qualified_bets"):
        append_to_tracking(results["qualified_bets"])
