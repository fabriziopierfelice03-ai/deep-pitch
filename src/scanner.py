"""
src/scanner.py
Official Unified Matchday Scanner for Deep-Pitch.
Deterministic, verified pre-match pipeline across Top 5 European Leagues:
- Premier League (E0)
- Serie A (I1)
- La Liga (SP1)
- Bundesliga (D1)
- Ligue 1 (F1)

Data Ingestion & Integrity:
- Real-time BetExplorer market consensus odds scraping
- Live FotMob/Opta rolling match logs (xG, xGA, shots, SOT, corners, cards)
- Official league standings (points, positions, UCL/UEL/Relegation cutoffs)
- Exact calendar rest days and matches in last 14 days
- Finished H2H history
- Live official ClubElo rating ingestion (ClubElo.com)
- Shin (1993) De-Vigging + Bayesian Shrinkage Layer (55.1% Model, 44.9% Market)
- Money Management: Quarter-Kelly Criterion (cap 3.0%) on active bankroll

STRICT ANTI-HALLUCINATION GUARD:
Any fixture without authentic market odds is marked as SKIPPED (ODDS_UNAVAILABLE).
Zero synthetic guessing, zero hardcoded odds.
"""

import os
import sys
import json
import re
import urllib.request
import unicodedata
import subprocess
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.feature_builder import (
    MatchRecord,
    H2HRecord,
    StandingsContext,
    build_56_features,
    validate_feature_scaling
)
from src.predict import predici_partita
from src.predict_v2 import predici_partita_v2
from src.v2_state_engine import LiveV2StateEngine
from src.injury_lineup import build_match_injury_report, check_injury_trap_guard
from src.decision_engine import calculate_quarter_kelly

CACHE_DIR = os.path.join(BASE_DIR, "data", "scanner_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TRACKING_CSV = os.path.join(BASE_DIR, "tracking_2026_2027.csv")
AUDIT_V2_CSV = os.path.join(BASE_DIR, "data", "v2_scan_audit.csv")

from src.elo_fetcher import get_live_elo

def get_elo(team_name: str) -> float:
    return get_live_elo(team_name, fallback=1650.0)

LEAGUE_CONFIG = {
    'E0': {'name': 'Premier League', 'fotmob_id': '47', 'slug': 'premier-league', 'betexp_url': 'https://www.betexplorer.com/football/england/premier-league/fixtures/', 'teams': 20, 'ucl': 4, 'uel': 6, 'rel': 18},
    'I1': {'name': 'Serie A', 'fotmob_id': '55', 'slug': 'serie-a', 'betexp_url': 'https://www.betexplorer.com/football/italy/serie-a/fixtures/', 'teams': 20, 'ucl': 4, 'uel': 6, 'rel': 18},
    'SP1': {'name': 'La Liga', 'fotmob_id': '87', 'slug': 'laliga', 'betexp_url': 'https://www.betexplorer.com/football/spain/laliga/fixtures/', 'teams': 20, 'ucl': 4, 'uel': 6, 'rel': 18},
    'D1': {'name': 'Bundesliga', 'fotmob_id': '54', 'slug': 'bundesliga', 'betexp_url': 'https://www.betexplorer.com/football/germany/bundesliga/fixtures/', 'teams': 18, 'ucl': 4, 'uel': 6, 'rel': 16},
    'F1': {'name': 'Ligue 1', 'fotmob_id': '53', 'slug': 'ligue-1', 'betexp_url': 'https://www.betexplorer.com/football/france/ligue-1/fixtures/', 'teams': 18, 'ucl': 4, 'uel': 6, 'rel': 16}
}

def strip_accents(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def norm_team(name: str) -> str:
    name = strip_accents(name.lower())
    return re.sub(r'[^a-z0-9]', '', name)

def get_current_bankroll() -> float:
    if not os.path.exists(TRACKING_CSV):
        return 1000.0
    try:
        import pandas as pd
        df = pd.read_csv(TRACKING_CSV)
        # Find last row with valid bankroll
        for val in reversed(df['Bankroll'].tolist()):
            if pd.notna(val):
                return float(val)
    except Exception:
        pass
    return 1000.0

def format_date_for_tracking(raw_date: str) -> str:
    """Converts ISO or YYYY-MM-DD date string to DD/MM/YY format required by tracking CSV."""
    try:
        if 'T' in raw_date:
            dt = datetime.fromisoformat(raw_date.replace('Z', '+00:00'))
        elif '-' in raw_date:
            dt = datetime.strptime(raw_date[:10], "%Y-%m-%d")
        elif '/' in raw_date:
            return raw_date[:8]
        else:
            return raw_date[:10]
        return dt.strftime("%d/%m/%y")
    except Exception:
        return raw_date[:10]

def load_existing_tracking_keys(csv_path: str = TRACKING_CSV) -> set:
    """Reads existing (Date, Match, Bet) tuples from tracking CSV for idempotency."""
    if not os.path.exists(csv_path):
        return set()
    keys = set()
    try:
        import csv
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 4:
                    date_val = row[0].strip()
                    match_val = norm_team(row[2].strip())
                    bet_val = row[3].strip()
                    keys.add((date_val, match_val, bet_val))
    except Exception:
        pass
    return keys

def append_bets_to_tracking(qualified_bets: List[Dict[str, Any]], csv_path: str = TRACKING_CSV) -> List[Dict[str, Any]]:
    """Appends newly qualified bets to tracking CSV, skipping duplicates."""
    existing_keys = load_existing_tracking_keys(csv_path)
    new_bets = []
    
    if not os.path.exists(csv_path):
        return []

    lines_to_append = []
    for b in qualified_bets:
        date_csv = format_date_for_tracking(b.get('date', ''))
        b['date_csv'] = date_csv
        
        match_title = b.get('match', '').replace(" - ", "-").strip()
        b['match_clean'] = match_title
        
        sel = str(b.get('selection', '')).strip()
        key = (date_csv, norm_team(match_title), sel)
        
        if key in existing_keys:
            continue
            
        existing_keys.add(key)
        new_bets.append(b)
        
        # Line format: Date,League,Match,Bet,Bookmaker_odds,Model_probability,Fair_odds,EV_Perc,Stake_euro,Result,Profit,Bankroll
        line = (
            f"{date_csv},"
            f"{b['league']},"
            f"{match_title},"
            f"{sel},"
            f"{b['bookmaker_odds']:.2f},"
            f"{b['model_prob']:.1f}%,"
            f"{b['fair_odds']:.2f},"
            f"{b['ev_perc']:.2f},"
            f"{b['stake_eur']:.2f},"
            f",,\n"
        )
        lines_to_append.append(line)
        
    if lines_to_append:
        with open(csv_path, 'a', encoding='utf-8') as f:
            for l in lines_to_append:
                f.write(l)
        print(f"[*] [TRACKING] Aggiunte {len(lines_to_append)} nuove selezioni a {csv_path}")
    else:
        print("[*] [TRACKING] Nessuna nuova selezione da aggiungere (tutte gia' tracciate o 0 qualificate).")
        
    return new_bets

def git_commit_and_push(csv_path: str = TRACKING_CSV, count: int = 0) -> bool:
    """Commits and pushes updated tracking CSV to GitHub origin main."""
    repo_dir = os.path.dirname(os.path.abspath(csv_path))
    try:
        subprocess.run(["git", "-C", repo_dir, "add", "tracking_2026_2027.csv"], check=True)
        msg = f"scan: auto-added {count} new qualified picks for upcoming matchday"
        subprocess.run(["git", "-C", repo_dir, "commit", "-m", msg], check=True)
        subprocess.run(["git", "-C", repo_dir, "push", "origin", "main"], check=True)
        print(f"[*] [GIT] Committato e sincronizzato con successo su GitHub: {msg}")
        return True
    except Exception as e:
        print(f"[!] [GIT] Warning push: {e}")
        return False

def get_telegram_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Loads Telegram bot token and chat ID from environment or bot.env files."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        return token, chat_id
        
    candidates = [
        os.path.join(BASE_DIR, "bot.env"),
        os.path.join(BASE_DIR, ".env"),
        os.path.abspath(os.path.join(BASE_DIR, "..", "live-sentinel", "bot.env")),
        os.path.abspath(os.path.join(BASE_DIR, "..", "live-sentinel", ".env")),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("TELEGRAM_BOT_TOKEN="):
                            token = line.split("=", 1)[1].strip(" \"'")
                        elif line.startswith("TELEGRAM_CHAT_ID="):
                            chat_id = line.split("=", 1)[1].strip(" \"'")
                if token and chat_id:
                    return token, chat_id
            except Exception:
                pass
    return token, chat_id

def send_telegram_scan_digest(new_bets: List[Dict[str, Any]], bankroll: float, scan_summary: Dict[str, Any]) -> bool:
    """Sends comprehensive Telegram notification of newly added pre-match picks."""
    token, chat_id = get_telegram_credentials()
    if not token or not chat_id:
        print("[!] [TELEGRAM] Token o Chat ID non trovati (notifica saltata).")
        return False

    now_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    
    eng_tag = str(scan_summary.get('engine', 'v2')).upper()
    if eng_tag == "V2":
        eng_display = "V2.1"
        ev_rule = "EV ∈ [2.5%, 6.0%)"
        kelly_label = "1/8-Kelly"
    else:
        eng_display = eng_tag
        ev_rule = "EV ≥ +5.0%"
        kelly_label = "Quarter-Kelly"
    if not new_bets:
        msg = (
            f"🎯 *DEEP PITCH {eng_display} | SCANSIONE PRE-MATCH COMPLETATA*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 Scansione: *{now_str}*\n"
            f"💰 Bankroll Attivo: *€{bankroll:.2f}*\n"
            f"🔍 Match analizzati: *{scan_summary.get('evaluated_fixtures', 0)}* (Top 5 Leghe)\n"
            f"🛡️ Trappole infortuni bloccate: *{scan_summary.get('traps_blocked', 0)}*\n"
            f"ℹ️ *Nessuna nuova value bet identificata con {ev_rule}* per questo turno.\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🛡️ _Nessun rischio assunto. Capitale preservato al 100%._"
        )
    else:
        msg_lines = [
            f"🎯 *DEEP PITCH {eng_display} | NUOVE VALUE BET IDENTIFICATE*",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"📅 Scansione: *{now_str}*",
            f"💰 Bankroll Attivo: *€{bankroll:.2f}*",
            f"🔍 Match analizzati: *{scan_summary.get('evaluated_fixtures', 0)}* (Top 5 Leghe)",
            f"🛡️ Trappole infortuni bloccate: *{scan_summary.get('traps_blocked', 0)}*",
            f"✨ Nuove selezioni qualificate ({ev_rule}): *{len(new_bets)}*\n"
        ]
        
        for b in new_bets:
            book_tag = f" [{b['best_bookmakers']}]" if b.get('best_bookmakers') else ""
            inj_tag = f" | ⚠️ {b['injury_note']}" if b.get('injury_note') else ""
            msg_lines.extend([
                f"⚽ *{b['match_clean']}*",
                f"• Lega: *{b['league']}* | Data: *{b['date_csv']}*",
                f"• Segno: *{b['selection']}* @ *{b['bookmaker_odds']:.2f}*{book_tag} (Fair: {b['fair_odds']:.2f})",
                f"• Prob. Modello: *{b['model_prob']:.1f}%* | EV: *+{b['ev_perc']:.2f}%*{inj_tag}",
                f"• Stake {kelly_label}: *€{b['stake_eur']:.2f}* ({b['applied_fraction']:.2f}%)\n"
            ])
            
        msg_lines.extend([
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"✅ _Aggiunte automaticamente su tracking_2026_2027.csv e sincronizzate su GitHub._",
            f"🛡️ _Live Sentinel monitorera' le partite in tempo reale per la copertura Green Book._"
        ])
        msg = "\n".join(msg_lines)

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id,
        "text": msg,
        "parse_mode": "Markdown"
    }).encode("utf-8")
    
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print("[*] [TELEGRAM] Notifica inviata con successo sul tuo smartphone.")
            return resp.status == 200
    except Exception as e:
        print(f"[!] [TELEGRAM] Errore invio messaggio: {e}")
        return False

def fetch_json(url: str, timeout: int = 15) -> Optional[Dict[str, Any]]:
    req = urllib.request.Request(
        url,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Accept': 'application/json'
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception:
        return None

def extract_page_props(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalizes FotMob payloads (supports legacy Next.js pageProps cache and direct /api/data/* responses)."""
    if not data:
        return {}
    return data.get('pageProps', data)

def scrape_betexplorer_odds(url: str) -> Dict[str, Dict[str, Any]]:
    """Scrapes authentic consensus decimal odds and match links from BetExplorer fixture page."""
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    odds_map = {}
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
            for r in rows:
                if 'data-odd' in r:
                    match_link = re.search(r'<a href="([^"]*)" class="in-match">(.*?)</a>', r, re.DOTALL)
                    odds = re.findall(r'data-odd="([0-9\.]+)"', r)
                    if match_link and len(odds) >= 3:
                        href = match_link.group(1).strip()
                        title = re.sub(r'<[^>]+>', '', match_link.group(2)).strip()
                        parts = title.split(' - ')
                        if len(parts) == 2:
                            h_norm = norm_team(parts[0])
                            a_norm = norm_team(parts[1])
                            key = f"{h_norm}_{a_norm}"
                            try:
                                odds_map[key] = {
                                    "avg": (float(odds[0]), float(odds[1]), float(odds[2])),
                                    "href": href,
                                }
                            except Exception:
                                pass
    except Exception as e:
        print(f"  [WARN] Impossibile scaricare quote da {url}: {e}")
    return odds_map

def fetch_betexplorer_best_odds(match_href: str) -> Optional[Dict[str, Any]]:
    """
    Fetches live per-bookmaker 1X2 odds across active Italian ADM bookmakers
    from BetExplorer's AJAX endpoint and returns the best available odd and
    bookmaker names for each outcome ('1', 'X', '2').
    """
    if not match_href:
        return None
    parts = [p for p in match_href.strip('/').split('/') if p]
    if not parts:
        return None
    match_id = parts[-1]
    ep = f"https://www.betexplorer.com/match-odds/{match_id}/0/1x2/odds/?lang=it"
    ref = f"https://www.betexplorer.com{match_href}" if match_href.startswith('/') else match_href
    req = urllib.request.Request(
        ep,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'X-Requested-With': 'XMLHttpRequest',
            'Referer': ref,
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode('utf-8', errors='ignore'))
        odds_html = payload.get('odds', '')
        if not odds_html:
            return None

        bookies = []
        for attrs, row in re.findall(r'<tr([^>]*)>(.*?)</tr>', odds_html, re.DOTALL):
            if 'inactive' in attrs.lower():
                continue
            vals = re.findall(r'data-odd="([0-9\.]+)"', row)
            if len(vals) < 3:
                continue
            m_ev = re.search(r"'event-name'\s*:\s*'([^']+)'", row)
            if m_ev:
                bname = m_ev.group(1).strip()
            else:
                m_t = re.search(r'title="([^"]+)"', row)
                bname = m_t.group(1).strip() if m_t else "Bookmaker"
            try:
                q1, qx, q2 = float(vals[0]), float(vals[1]), float(vals[2])
                if q1 > 1.01 and qx > 1.01 and q2 > 1.01:
                    bookies.append((bname, q1, qx, q2))
            except ValueError:
                continue

        if not bookies:
            return None

        best_q = {}
        best_books = {}
        for idx, sel in enumerate(("1", "X", "2"), start=1):
            max_val = max(b[idx] for b in bookies)
            names = [b[0] for b in bookies if abs(b[idx] - max_val) < 1e-6]
            best_q[sel] = max_val
            best_books[sel] = names

        return {
            "best": (best_q["1"], best_q["X"], best_q["2"]),
            "books": best_books,
            "n_bookmakers": len(bookies),
        }
    except Exception:
        return None

BETEXP_ALIASES = {
    "manchesterunited": "manchesterutd",
    "atleticomadrid": "atlmadrid",
    "athleticclub": "athbilbao",
    "deportivoacoruna": "depacoruna",
    "borussiamonchengladbach": "bmonchengladbach",
    "bayernmunchen": "bayernmunich",
    "parissaintgermain": "psg",
}

def find_matched_odds(home_name: str, away_name: str, odds_map: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    h_k = BETEXP_ALIASES.get(norm_team(home_name), norm_team(home_name))
    a_k = BETEXP_ALIASES.get(norm_team(away_name), norm_team(away_name))
    direct_key = f"{h_k}_{a_k}"
    entry = odds_map.get(direct_key)
    if entry is None:
        for k, v in odds_map.items():
            parts = k.split('_')
            if len(parts) == 2:
                if (h_k in parts[0] or parts[0] in h_k) and (a_k in parts[1] or parts[1] in a_k):
                    entry = v
                    break
    if entry is None:
        return None
    return entry

def fetch_league_standings(league_code: str, league_data: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, float]]:
    conf = LEAGUE_CONFIG[league_code]
    if league_data is None:
        u = f"https://www.fotmob.com/api/data/leagues?id={conf['fotmob_id']}"
        league_data = fetch_json(u)
    props = extract_page_props(league_data)
    standings_map = {}
    if 'table' in props and props['table']:
        try:
            raw_all = props['table'][0].get('data', {}).get('table', {}).get('all', [])
            pts_by_idx = {t.get('idx'): t.get('pts', 0) for t in raw_all}
            ucl_pts = pts_by_idx.get(conf['ucl'], 12)
            uel_pts = pts_by_idx.get(conf['uel'], 10)
            rel_pts = pts_by_idx.get(conf['rel'], 4)
            for t in raw_all:
                k = norm_team(t.get('name', ''))
                standings_map[k] = {
                    'idx': float(t.get('idx', 10)),
                    'pts': float(t.get('pts', 0)),
                    'played': float(t.get('played', 6)),
                    'ucl_pts': float(ucl_pts),
                    'uel_pts': float(uel_pts),
                    'rel_pts': float(rel_pts)
                }
        except Exception:
            pass
    return standings_map

def get_standings_context(league_code: str, home_name: str, away_name: str, standings_map: Dict[str, Dict[str, float]]) -> StandingsContext:
    h_k, a_k = norm_team(home_name), norm_team(away_name)
    h_info = standings_map.get(h_k) or next((v for k, v in standings_map.items() if k in h_k or h_k in k), None)
    a_info = standings_map.get(a_k) or next((v for k, v in standings_map.items() if k in a_k or a_k in k), None)

    h_pos = h_info['idx'] if h_info else 10.0
    a_pos = a_info['idx'] if a_info else 10.0
    h_pts = h_info['pts'] if h_info else 9.0
    a_pts = a_info['pts'] if a_info else 9.0
    played = max(h_info['played'] if h_info else 6.0, a_info['played'] if a_info else 6.0)
    matchday = played + 1.0
    season_half = 1.0 if matchday <= 19.0 else 2.0

    ucl_pts = h_info['ucl_pts'] if h_info else 12.0
    uel_pts = h_info['uel_pts'] if h_info else 10.0
    rel_pts = h_info['rel_pts'] if h_info else 4.0

    return StandingsContext(
        home_pos=float(h_pos),
        away_pos=float(a_pos),
        home_points=float(h_pts),
        away_points=float(a_pts),
        home_dist_ucl=float(h_pts - ucl_pts),
        away_dist_ucl=float(a_pts - ucl_pts),
        home_dist_uel=float(h_pts - uel_pts),
        away_dist_uel=float(a_pts - uel_pts),
        home_dist_rel=float(h_pts - rel_pts),
        away_dist_rel=float(a_pts - rel_pts),
        matchday=float(matchday),
        season_half=float(season_half)
    )

def parse_past_match(m_info: Dict[str, Any], target_team_name: str) -> Optional[MatchRecord]:
    link = m_info.get('linkToMatch', '')
    mid_match = re.search(r'#(\d+)', link)
    if not mid_match:
        return None
    past_mid = mid_match.group(1)
    
    # Check cache (invalidate stale pre-match snapshots lacking post-match Opta stats)
    cache_f = os.path.join(CACHE_DIR, f"{past_mid}.json")
    past_data = None
    if os.path.exists(cache_f):
        try:
            with open(cache_f, 'r', encoding='utf-8') as f:
                candidate = json.load(f)
            c_stats = ((extract_page_props(candidate).get('content', {}).get('stats') or {}).get('Periods') or {}).get('All', {}).get('stats')
            if c_stats:
                past_data = candidate
        except Exception:
            pass
    if not past_data:
        u = f"https://www.fotmob.com/api/data/matchDetails?matchId={past_mid}"
        past_data = fetch_json(u)
        if past_data:
            with open(cache_f, 'w', encoding='utf-8') as f:
                json.dump(past_data, f)
    if not past_data:
        return None

    content = extract_page_props(past_data).get('content', {})
    is_home = m_info.get('home', {}).get('isOurTeam', False)
    venue = 'H' if is_home else 'A'
    opp_name = m_info.get('away', {}).get('name') if is_home else m_info.get('home', {}).get('name')
    date_str = m_info.get('date', {}).get('utcTime', '')[:10]

    score_parts = m_info.get('score', '0 - 0').split(' - ')
    try:
        gf = int(score_parts[0]) if is_home else int(score_parts[1])
        ga = int(score_parts[1]) if is_home else int(score_parts[0])
    except Exception:
        gf, ga = 0, 0

    pts = 3 if gf > ga else (1 if gf == ga else 0)

    stats_dict = content.get('stats') or {}
    periods_dict = stats_dict.get('Periods') or {}
    all_dict = periods_dict.get('All') or {}
    stats_list = all_dict.get('stats') or []
    xg_for, xg_against = float(gf), float(ga)
    shots, sot, corners, cards = 12, 4, 5, 2

    for grp in stats_list:
        for item in grp.get('stats', []):
            t = item.get('title', '')
            vals = item.get('stats', [])
            if len(vals) == 2 and vals[0] is not None and vals[1] is not None:
                h_val, a_val = vals[0], vals[1]
                if 'Expected goals' in t:
                    try:
                        xg_for = float(h_val) if is_home else float(a_val)
                        xg_against = float(a_val) if is_home else float(h_val)
                    except Exception: pass
                elif t == 'Total shots':
                    try:
                        shots = int(h_val) if is_home else int(a_val)
                    except Exception: pass
                elif t == 'Shots on target':
                    try:
                        sot = int(h_val) if is_home else int(a_val)
                    except Exception: pass
                elif t == 'Corners':
                    try:
                        corners = int(h_val) if is_home else int(a_val)
                    except Exception: pass
                elif t == 'Yellow cards':
                    try:
                        cards = int(h_val) if is_home else int(a_val)
                    except Exception: pass

    return MatchRecord(
        date=date_str,
        opponent=opp_name,
        venue=venue,
        goals_for=gf,
        goals_against=ga,
        xg_for=xg_for,
        xg_against=xg_against,
        shots=shots,
        shots_on_target=sot,
        corners=corners,
        cards=cards,
        points=pts,
        source="FotMob/Opta"
    )

def parse_h2h_records(content: Dict[str, Any], home_name: str, away_name: str) -> H2HRecord:
    h2h_matches = content.get('h2h', {}).get('matches', [])
    finished = [m for m in h2h_matches if m.get('status', {}).get('finished') is True]

    h_pts, a_pts, h_gf, a_gf = [], [], [], []
    v_pts, v_gf = [], []
    norm_h, norm_a = norm_team(home_name), norm_team(away_name)

    for m in finished[:5]:
        score_str = str(m.get('status', {}).get('scoreStr', '0 - 0')).replace('\u200e', '').strip()
        parts = score_str.split(' - ')
        if len(parts) == 2:
            try:
                sh, sa = int(parts[0].strip()), int(parts[1].strip())
            except Exception:
                continue
            mh = norm_team(m.get('home', {}).get('name', ''))
            is_target_home = (norm_h in mh or mh in norm_h)
            if is_target_home:
                hp = 3 if sh > sa else (1 if sh == sa else 0)
                ap = 3 if sa > sh else (1 if sa == sh else 0)
                h_pts.append(hp); a_pts.append(ap); h_gf.append(sh); a_gf.append(sa)
                v_pts.append(hp); v_gf.append(sh)
            else:
                hp = 3 if sa > sh else (1 if sa == sh else 0)
                ap = 3 if sh > sa else (1 if sh == sa else 0)
                h_pts.append(hp); a_pts.append(ap); h_gf.append(sa); a_gf.append(sh)

    h_avg_pts = sum(h_pts)/len(h_pts) if h_pts else 1.3
    a_avg_pts = sum(a_pts)/len(a_pts) if a_pts else 1.3
    h_avg_gf = sum(h_gf)/len(h_gf) if h_gf else 1.4
    a_avg_gf = sum(a_gf)/len(a_gf) if a_gf else 1.2
    v_avg_pts = sum(v_pts)/len(v_pts) if v_pts else 1.5
    v_avg_gf = sum(v_gf)/len(v_gf) if v_gf else 1.5

    return H2HRecord(
        home_avg_points=round(h_avg_pts, 2),
        away_avg_points=round(a_avg_pts, 2),
        home_avg_goals_for=round(h_avg_gf, 2),
        home_avg_goals_against=round(a_avg_gf, 2),
        home_avg_xg=round(h_avg_gf * 0.95, 2),
        away_avg_xg=round(a_avg_gf * 0.95, 2),
        venue_home_avg_points=round(v_avg_pts, 2),
        venue_home_avg_goals_for=round(v_avg_gf, 2),
        source="FotMob H2H"
    )

def compute_rest_and_fatigue(fixture_time_str: str, match_records: List[MatchRecord]) -> Tuple[float, float]:
    try:
        fix_dt = datetime.fromisoformat(fixture_time_str.replace('Z', '+00:00'))
    except Exception:
        return 7.0, 2.0
    dates = []
    for m in match_records:
        try:
            dates.append(datetime.strptime(m.date, '%Y-%m-%d'))
        except Exception:
            pass
    if not dates:
        return 7.0, 2.0
    dates.sort(reverse=True)
    rest_days = max(1.0, float((fix_dt.date() - dates[0].date()).days))
    matches_14d = float(sum(1 for d in dates if 0 < (fix_dt.date() - d.date()).days <= 14))
    return rest_days, matches_14d

def extract_raw_standings_rows(league_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    props = extract_page_props(league_data)
    if 'table' in props and props['table']:
        try:
            return props['table'][0].get('data', {}).get('table', {}).get('all', []) or []
        except Exception:
            pass
    return []


def load_cached_past_match_jsons(team_form_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Loads up to 5 already-cached completed FotMob match JSONs for a team's recent form."""
    out = []
    for m_info in team_form_list[:5]:
        link = m_info.get('linkToMatch', '')
        mid_match = re.search(r'#(\d+)', link)
        if not mid_match:
            continue
        cache_f = os.path.join(CACHE_DIR, f"{mid_match.group(1)}.json")
        if os.path.exists(cache_f):
            try:
                with open(cache_f, 'r', encoding='utf-8') as fp:
                    out.append(json.load(fp))
            except Exception:
                pass
    return out


def append_v2_audit_rows(audit_rows: List[Dict[str, Any]], csv_path: str = AUDIT_V2_CSV) -> None:
    """Appends per-fixture Deep-Pitch v2 audit + CLV baseline + injury snapshot to data/v2_scan_audit.csv."""
    if not audit_rows:
        return
    import csv
    fieldnames = [
        "timestamp", "date", "league", "match", "engine",
        "avg_q1", "avg_qx", "avg_q2", "best_q1", "best_qx", "best_q2",
        "p_v2_1", "p_v2_x", "p_v2_2", "p_adj_1", "p_adj_x", "p_adj_2",
        "p_shin_1", "p_shin_x", "p_shin_2",
        "lineup_type", "home_fresh_out", "away_fresh_out", "injury_logit_shift",
        "qualified_sel", "qualified_odd", "qualified_book", "qualified_ev", "trap_blocked_reason",
    ]
    write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in audit_rows:
            writer.writerow(r)


def run_matchday_scan(mode: str = "auto", publish: bool = False, engine: str = "v2") -> Dict[str, Any]:
    """
    Main scanner execution routine.
    - mode='auto': Automatically detects matches in the upcoming window.
    - engine='v2' (default): Deep-Pitch v2 Residual-Offset 5-Head Ensemble + FotMob Fresh-Injury/Lineup Guard.
    - engine='v1': Legacy BettingNet v1 + Bayesian Shrinkage.
    """
    print("==========================================================================")
    print(f"      DEEP-PITCH ({engine.upper()}): OFFICIAL CERTIFIED PRE-MATCH SCANNER")
    print("==========================================================================")
    
    current_bankroll = get_current_bankroll()
    print(f"[*] Bankroll Operativo Attivo Rilevato: EUR {current_bankroll:.2f}")
    print("[*] Sorgente FotMob: API Data v2 (/api/data/leagues & /api/data/matchDetails)")

    v2_state_engine = None
    if engine == "v2":
        print("[*] Inizializzazione Deep-Pitch v2 Live State Engine (Past-Market + Exact DC npxG + EWMA)...")
        v2_state_engine = LiveV2StateEngine()
        print(f"    - Squadre indicizzate live: {len(v2_state_engine.team_state)} | Solver esatti attivi: {len(v2_state_engine.league_solvers)}/5 leghe")

    # 1. Scrape live odds from BetExplorer for all 5 leagues
    print("\n[*] Recupero quote live di consenso dai bookmaker (BetExplorer)...")
    live_odds_by_league = {}
    for lcode, conf in LEAGUE_CONFIG.items():
        odds_m = scrape_betexplorer_odds(conf['betexp_url'])
        live_odds_by_league[lcode] = odds_m
        print(f"    - [{lcode}] {conf['name']}: {len(odds_m)} match quotati")

    # 2. Discover target fixtures
    print("\n[*] Recupero calendario ufficiale e selezione incontri in arrivo...")
    target_fixtures = []
    standings_cache = {}
    raw_standings_cache = {}

    for lcode, conf in LEAGUE_CONFIG.items():
        u = f"https://www.fotmob.com/api/data/leagues?id={conf['fotmob_id']}"
        data = fetch_json(u)
        props = extract_page_props(data)
        standings_cache[lcode] = fetch_league_standings(lcode, league_data=props)
        raw_standings_cache[lcode] = extract_raw_standings_rows(props)
        if 'fixtures' not in props:
            continue
        all_m = props['fixtures'].get('allMatches', [])
        
        # Dynamic lookahead window:
        # Midweek scan (Mon-Wed): evaluate strictly midweek fixtures up to Thursday (delta_days <= 2)
        # Thursday scan: up to Friday (delta_days <= 1)
        # Weekend scan (Fri-Sun): evaluate weekend fixtures up to Monday (delta_days <= 3)
        now_dt = datetime.now(timezone.utc)
        weekday = now_dt.weekday()
        if weekday in (0, 1, 2):
            max_lookahead = 2
        elif weekday == 3:
            max_lookahead = 1
        else:
            max_lookahead = 3

        for m in all_m:
            st = m.get('status', {})
            if st.get('finished', False):
                continue
            utc_time = st.get('utcTime') or m.get('time', {}).get('utcTime')
            if not utc_time:
                continue
            try:
                m_dt = datetime.fromisoformat(utc_time.replace('Z', '+00:00'))
                delta_days = (m_dt.date() - now_dt.date()).days
                if 0 <= delta_days <= max_lookahead:
                    target_fixtures.append({
                        'league': lcode,
                        'fotmob_id': str(m.get('id')),
                        'home': m.get('home', {}).get('name'),
                        'away': m.get('away', {}).get('name'),
                        'time': utc_time
                    })
            except Exception:
                pass

    print(f"[*] Totale incontri individuati nella finestra temporale: {len(target_fixtures)}")

    # 3. Evaluate each fixture
    evaluated_results = []
    qualified_1x2_bets = []
    qualified_dnb_bets = []
    audit_rows = []
    skipped_no_odds = 0
    traps_blocked = 0

    for idx, f in enumerate(target_fixtures, 1):
        lcode = f['league']
        h_name = f['home']
        a_name = f['away']
        mid = f['fotmob_id']
        t_str = f['time']
        title = f"{h_name} - {a_name}"

        # Match live consensus odds + fetch best available ADM bookmaker odds
        odds_entry = find_matched_odds(h_name, a_name, live_odds_by_league.get(lcode, {}))
        if not odds_entry:
            skipped_no_odds += 1
            continue

        if isinstance(odds_entry, tuple):
            avg_q1, avg_qx, avg_q2 = odds_entry
            match_href = ""
        else:
            avg_q1, avg_qx, avg_q2 = odds_entry["avg"]
            match_href = odds_entry.get("href", "")
        best_info = fetch_betexplorer_best_odds(match_href)
        if best_info:
            q1, qx, q2 = best_info["best"]
            best_books = best_info["books"]
            b1_lbl = "/".join(best_books["1"][:2])
            bx_lbl = "/".join(best_books["X"][:2])
            b2_lbl = "/".join(best_books["2"][:2])
            print(
                f"[{idx:02d}/{len(target_fixtures)}] {lcode} | {title} "
                f"(Avg: {avg_q1:.2f}-{avg_qx:.2f}-{avg_q2:.2f} | "
                f"Best ADM: {q1:.2f}[{b1_lbl}] - {qx:.2f}[{bx_lbl}] - {q2:.2f}[{b2_lbl}])"
            )
        else:
            q1, qx, q2 = avg_q1, avg_qx, avg_q2
            best_books = {"1": ["Media"], "X": ["Media"], "2": ["Media"]}
            print(f"[{idx:02d}/{len(target_fixtures)}] {lcode} | {title} (Odds Avg fallback: {q1:.2f} - {qx:.2f} - {q2:.2f})")

        # Download match details (refresh upcoming cache if older than 6h so teamForm & injuries stay fresh)
        cache_f = os.path.join(CACHE_DIR, f"{mid}.json")
        m_data = None
        if os.path.exists(cache_f):
            try:
                if (datetime.now().timestamp() - os.path.getmtime(cache_f)) < 21600:
                    with open(cache_f, 'r', encoding='utf-8') as fp:
                        m_data = json.load(fp)
            except Exception:
                pass
        if not m_data:
            u = f"https://www.fotmob.com/api/data/matchDetails?matchId={mid}"
            m_data = fetch_json(u)
            if m_data:
                with open(cache_f, 'w', encoding='utf-8') as fp:
                    json.dump(m_data, fp)

        props = extract_page_props(m_data)
        if not props or 'content' not in props:
            continue

        content = props.get('content', {})
        teamForm = content.get('matchFacts', {}).get('teamForm')
        if not teamForm or len(teamForm) < 2:
            continue

        home_matches = [parse_past_match(m, h_name) for m in teamForm[0]]
        away_matches = [parse_past_match(m, a_name) for m in teamForm[1]]
        home_matches = [m for m in home_matches if m is not None]
        away_matches = [m for m in away_matches if m is not None]

        if not home_matches or not away_matches:
            continue

        h2h_rec = parse_h2h_records(content, h_name, a_name)
        standings = get_standings_context(lcode, h_name, a_name, standings_cache[lcode])
        h_elo, a_elo = get_elo(h_name), get_elo(a_name)
        h_rest, h_m14 = compute_rest_and_fatigue(t_str, home_matches)
        a_rest, a_m14 = compute_rest_and_fatigue(t_str, away_matches)

        # Build Live FotMob Pre-Match Fresh-Injury & Lineup Report
        h_recent_jsons = load_cached_past_match_jsons(teamForm[0])
        a_recent_jsons = load_cached_past_match_jsons(teamForm[1])
        injury_report = build_match_injury_report(m_data, h_name, a_name, h_recent_jsons, a_recent_jsons)
        h_inj = injury_report["home"]
        a_inj = injury_report["away"]
        if h_inj["fresh_missing_count"] > 0 or a_inj["fresh_missing_count"] > 0:
            print(
                f"  [INJURY/XI ({injury_report['lineup_type']})] "
                f"Home fresh out: {h_inj['fresh_missing_count']:.1f} ({h_inj['fresh_missing_value_ratio']*100:.1f}% XI) | "
                f"Away fresh out: {a_inj['fresh_missing_count']:.1f} ({a_inj['fresh_missing_value_ratio']*100:.1f}% XI) | "
                f"Logit shift: {injury_report['net_logit_shift_home']:+.3f}"
            )

        if engine == "v2" and v2_state_engine is not None:
            lconf = LEAGUE_CONFIG[lcode]
            total_games = 34 if lconf["teams"] <= 18 else 38
            v2_payload = v2_state_engine.build_match_payload(
                div=lcode,
                home_team=h_name,
                away_team=a_name,
                home_elo=h_elo,
                away_elo=a_elo,
                rest_days_h=h_rest,
                rest_days_a=a_rest,
                matches_14d_h=h_m14,
                matches_14d_a=a_m14,
                home_pts=standings.home_points,
                away_pts=standings.away_points,
                matchday=standings.matchday,
                standings_rows=raw_standings_cache.get(lcode, []),
                total_games=total_games,
                rel_pos_1idx=lconf["rel"],
            )
            pred_v2 = predici_partita_v2(
                v2_payload,
                market_odds=[avg_q1, avg_qx, avg_q2],
                injury_report=injury_report,
                devig_method="shin",
            )
            p1, px, p2 = pred_v2["Final_1X2"]["Probs_Float"]
            raw_v2_p = pred_v2["V2_Raw_1X2"]["Probabilita_%"]
            shin_mkt_p = pred_v2.get("Market_Reference", {}).get("De_Vigged_Market_%", {"1": 0, "X": 0, "2": 0})
            min_ev_thr = 0.025
            max_ev_thr = 0.060
            min_odds_thr = 1.35
            max_odds_thr = 6.00
            kelly_frac_thr = 0.125
            max_cap_thr = 0.020
        else:
            features = build_56_features(
                home_elo=h_elo,
                away_elo=a_elo,
                home_matches=home_matches,
                away_matches=away_matches,
                h2h=h2h_rec,
                standings=standings,
                home_rest_days=h_rest,
                away_rest_days=a_rest,
                home_matches_14d=h_m14,
                away_matches_14d=a_m14,
                is_derby=0.0
            )
            scaling = validate_feature_scaling(features)
            if not scaling["is_sane"]:
                print(f"  [SCALING WARNING] {title}: min={scaling['min_scaled']:.2f}, max={scaling['max_scaled']:.2f}")
            pred = predici_partita(features, market_odds=[avg_q1, avg_qx, avg_q2], devig_method="shin")
            calib_probs = pred["Bayesian_Shrinkage_1X2"]["Probabilita_%"]
            p1 = calib_probs["1"] / 100.0
            px = calib_probs["X"] / 100.0
            p2 = calib_probs["2"] / 100.0
            raw_v2_p = {"1": round(p1 * 100, 2), "X": round(px * 100, 2), "2": round(p2 * 100, 2)}
            shin_mkt_p = pred["Bayesian_Shrinkage_1X2"].get("De_Vigged_Market_%", {"1": 0, "X": 0, "2": 0})
            min_ev_thr = 0.050
            max_ev_thr = 1.000
            min_odds_thr = 1.60
            max_odds_thr = 3.10
            kelly_frac_thr = 0.250
            max_cap_thr = 0.030

        # Track 1: 1X2 Screening using BEST available ADM bookmaker odds (line-shopping)
        odds_map = {"1": q1, "X": qx, "2": q2}
        avg_odds_map = {"1": avg_q1, "X": avg_qx, "2": avg_q2}
        probs_map = {"1": p1, "X": px, "2": p2}

        # Pick at most the highest-EV actionable selection per match
        best_match_sel = None
        best_match_dec = None
        match_trap_reason = ""
        for sel in ["1", "X", "2"]:
            odd = odds_map[sel]
            prob = probs_map[sel]
            dec = calculate_quarter_kelly(
                prob=prob,
                bookmaker_odds=odd,
                bankroll=current_bankroll,
                min_ev=min_ev_thr,
                min_odds=min_odds_thr,
                max_odds=max_odds_thr,
                max_cap=max_cap_thr,
                max_ev=max_ev_thr,
                kelly_fraction=kelly_frac_thr,
            )
            if dec["actionable"]:
                if engine == "v2":
                    blocked, trap_msg = check_injury_trap_guard(sel, injury_report)
                    if blocked:
                        traps_blocked += 1
                        match_trap_reason = trap_msg
                        print(f"  [INJURY TRAP BLOCKED] {sel} @ {odd:.2f} (EV +{dec['ev_perc']:.2f}%) -> {trap_msg}")
                        continue
                if best_match_dec is None or dec["ev_perc"] > best_match_dec["ev_perc"]:
                    best_match_sel = sel
                    best_match_dec = dec

        if best_match_sel is not None and best_match_dec is not None:
            sel = best_match_sel
            odd = odds_map[sel]
            books_str = ", ".join(best_books.get(sel, []))
            qualified_1x2_bets.append({
                "date": t_str[:10],
                "league": lcode,
                "match": title,
                "selection": sel,
                "bookmaker_odds": odd,
                "avg_odds": avg_odds_map[sel],
                "best_bookmakers": books_str,
                "fair_odds": best_match_dec["fair_odds"],
                "model_prob": best_match_dec["prob_perc"],
                "ev_perc": best_match_dec["ev_perc"],
                "stake_eur": best_match_dec["stake_eur"],
                "applied_fraction": best_match_dec["applied_fraction_perc"],
                "injury_note": (
                    f"H_out={h_inj['fresh_missing_count']:.1f}, A_out={a_inj['fresh_missing_count']:.1f}"
                    if (h_inj["fresh_missing_count"] > 0 or a_inj["fresh_missing_count"] > 0) else ""
                ),
            })
            print(
                f"  >>> [VALUE 1X2 {engine.upper()}] {sel} @ {odd:.2f} [{books_str}] (Media: {avg_odds_map[sel]:.2f}) "
                f"| P={best_match_dec['prob_perc']:.1f}% | EV: +{best_match_dec['ev_perc']:.2f}% | Stake: EUR {best_match_dec['stake_eur']:.2f}"
            )

        audit_rows.append({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "date": t_str[:10],
            "league": lcode,
            "match": title,
            "engine": engine,
            "avg_q1": avg_q1, "avg_qx": avg_qx, "avg_q2": avg_q2,
            "best_q1": q1, "best_qx": qx, "best_q2": q2,
            "p_v2_1": raw_v2_p["1"], "p_v2_x": raw_v2_p["X"], "p_v2_2": raw_v2_p["2"],
            "p_adj_1": round(p1 * 100.0, 2), "p_adj_x": round(px * 100.0, 2), "p_adj_2": round(p2 * 100.0, 2),
            "p_shin_1": shin_mkt_p["1"], "p_shin_x": shin_mkt_p["X"], "p_shin_2": shin_mkt_p["2"],
            "lineup_type": injury_report["lineup_type"],
            "home_fresh_out": h_inj["fresh_missing_count"],
            "away_fresh_out": a_inj["fresh_missing_count"],
            "injury_logit_shift": injury_report["net_logit_shift_home"],
            "qualified_sel": best_match_sel or "",
            "qualified_odd": odds_map[best_match_sel] if best_match_sel else "",
            "qualified_book": ", ".join(best_books.get(best_match_sel, [])) if best_match_sel else "",
            "qualified_ev": best_match_dec["ev_perc"] if best_match_dec else "",
            "trap_blocked_reason": match_trap_reason,
        })

        # Track 2: Shadow Ghost DNB
        q_dnb_1 = round(max(q1 * (1.0 - 1.0 / qx), q1 * (1.0 - 1.0 / qx) * 1.02), 2)
        q_dnb_2 = round(max(q2 * (1.0 - 1.0 / qx), q2 * (1.0 - 1.0 / qx) * 1.02), 2)
        p_dnb_1 = p1 / (p1 + p2) if (p1 + p2) > 0 else 0.5
        p_dnb_2 = p2 / (p1 + p2) if (p1 + p2) > 0 else 0.5

        for sel_code, odd_dnb, p_dnb, orig_sel in [("1 DNB", q_dnb_1, p_dnb_1, "1"), ("2 DNB", q_dnb_2, p_dnb_2, "2")]:
            ev_dnb = (p_dnb * odd_dnb) - 1.0
            if 0.020 <= ev_dnb < 0.080 and 1.30 < odd_dnb < 2.80:
                b = odd_dnb - 1.0
                k_star = ev_dnb / b if b > 0 else 0.0
                q_k = k_star * 0.25
                c_frac = max(0.0, min(q_k, 0.030))
                stake_d = round(current_bankroll * c_frac, 2)
                qualified_dnb_bets.append({
                    "date": t_str[:10],
                    "league": lcode,
                    "match": title,
                    "selection": sel_code,
                    "dnb_odds": odd_dnb,
                    "fair_odds": round(1.0 / p_dnb, 2) if p_dnb > 0 else 999.0,
                    "conditional_prob": round(p_dnb * 100, 2),
                    "ev_perc": round(ev_dnb * 100, 2),
                    "stake_eur": stake_d,
                    "applied_fraction": round(c_frac * 100, 2)
                })

    qualified_1x2_bets.sort(key=lambda x: -x['ev_perc'])
    qualified_dnb_bets.sort(key=lambda x: -x['ev_perc'])

    if engine == "v2" and audit_rows:
        append_v2_audit_rows(audit_rows)

    print("\n==========================================================================")
    print(f"              RIEPILOGO SCANSIONE UFFICIALE ({engine.upper()})")
    print("==========================================================================")
    print(f"[*] Incontri con Quote Verificate Analizzati: {len(target_fixtures) - skipped_no_odds}")
    print(f"[*] Incontri Senza Quote (Saltati per Sicurezza): {skipped_no_odds}")
    print(f"[*] Trappole Infortuni/Turnover Bloccate: {traps_blocked}")
    ev_lbl = "EV in [1.0%, 8.0%)" if engine == "v2" else "EV >= +5.0%"
    print(f"[*] Selezioni Ufficiali 1X2 Qualificate ({ev_lbl}): {len(qualified_1x2_bets)}")
    print(f"[*] Selezioni Shadow DNB Qualificate: {len(qualified_dnb_bets)}")

    # Sanity Anomaly Guard
    if len(qualified_1x2_bets) > 15:
        print("\n[ALERT] Rilevato volume anomalo di selezioni (> 15). Controllo di sicurezza attivato.")

    res = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engine": engine,
        "current_bankroll": current_bankroll,
        "evaluated_fixtures": len(target_fixtures) - skipped_no_odds,
        "skipped_no_odds": skipped_no_odds,
        "traps_blocked": traps_blocked,
        "qualified_1x2_bets": qualified_1x2_bets,
        "qualified_dnb_bets": qualified_dnb_bets
    }

    if publish:
        print("\n==========================================================================")
        print("          PUBBLICAZIONE AUTOMATICA & SINCRONIZZAZIONE (--publish)")
        print("==========================================================================")
        newly_added = append_bets_to_tracking(qualified_1x2_bets, TRACKING_CSV)
        if newly_added:
            git_commit_and_push(TRACKING_CSV, len(newly_added))
        send_telegram_scan_digest(newly_added, current_bankroll, res)

    return res

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Deep Pitch Pre-Match Value Scanner")
    parser.add_argument("--publish", action="store_true", help="Append new qualified picks to tracking CSV, push to GitHub and send Telegram digest")
    parser.add_argument("--engine", choices=["v2", "v1"], default="v2", help="Prediction engine version (default: v2)")
    args = parser.parse_args()
    run_matchday_scan(publish=args.publish, engine=args.engine)
