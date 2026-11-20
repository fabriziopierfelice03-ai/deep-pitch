"""
src/elo_fetcher.py
Modulo per l'estrazione e il caching dinamico in tempo reale dei rating ELO da ClubElo.com.
Supporta tutti i club dei top 5 campionati europei (ITA, ENG, ESP, GER, FRA)
con normalizzazione dei nomi, alias e cache locale persistente.
"""

import urllib.request
import re
import json
import os
import time
import unicodedata
from typing import Dict, Optional

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CACHE_FILE = os.path.join(CACHE_DIR, "clubelo_live.json")
CACHE_TTL_HOURS = 12

LEAGUE_URLS = {
    'I1': 'ITA',
    'E0': 'ENG',
    'SP1': 'ESP',
    'D1': 'GER',
    'F1': 'FRA'
}

ALIAS_MAP = {
    "acmilan": "milan",
    "internazionale": "inter",
    "hellasverona": "verona",
    "manchestercity": "mancity",
    "manchesterunited": "manunited",
    "tottenhamhotspur": "tottenham",
    "spurs": "tottenham",
    "newcastleunited": "newcastle",
    "nottinghamforest": "forest",
    "afcbournemouth": "bournemouth",
    "wolverhampton": "wolves",
    "wolverhamptonwanderers": "wolves",
    "brightonandhovealbion": "brighton",
    "leedsunited": "leeds",
    "westbromwichalbion": "westbrom",
    "athleticclub": "athbilbao",
    "athleticbilbao": "athbilbao",
    "atleticomadrid": "atletico",
    "atleticodemadrid": "atletico",
    "celtavigo": "celta",
    "celtadevigo": "celta",
    "deportivoalaves": "alaves",
    "deportivoacoruna": "depor",
    "deportivolacoruna": "depor",
    "realsociedad": "sociedad",
    "realbetis": "betis",
    "rayovallecano": "vallecano",
    "bayernmunchen": "bayern",
    "bayernmunich": "bayern",
    "bayerleverkusen": "leverkusen",
    "borussiadortmund": "dortmund",
    "borussiamonchengladbach": "gladbach",
    "eintrachtfrankfurt": "frankfurt",
    "scfreiburg": "freiburg",
    "tsghoffenheim": "hoffenheim",
    "1fcunionberlin": "unionberlin",
    "mainz05": "mainz",
    "1fckoln": "koln",
    "fckoln": "koln",
    "vfbstuttgart": "stuttgart",
    "hamburgersv": "hamburg",
    "parissaintgermain": "parissg",
    "asmonaco": "monaco",
    "olympiquemarseille": "marseille",
    "olympiquelyonnais": "lyon",
}

def strip_accents(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def clean_key(name: str) -> str:
    name = strip_accents(name.lower())
    clean = re.sub(r'[^a-z0-9]', '', name)
    return ALIAS_MAP.get(clean, clean)

def _decode_json_str(raw: str) -> str:
    """Decodes JSON unicode escapes (e.g. \\u00e9 -> e) extracted via regex."""
    try:
        return json.loads(f'"{raw}"')
    except Exception:
        return raw

def is_cache_valid() -> bool:
    if not os.path.exists(CACHE_FILE):
        return False
    mtime = os.path.getmtime(CACHE_FILE)
    return (time.time() - mtime) < (CACHE_TTL_HOURS * 3600)

def scrape_live_clubelo() -> Dict[str, float]:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    ratings = {}

    def store(raw_esc: str, elo_str: str):
        elo = round(float(elo_str), 2)
        name = _decode_json_str(raw_esc)
        ratings[clean_key(name)] = elo
        ratings[name.lower()] = elo

    for l_code, fed_url in LEAGUE_URLS.items():
        url = f"https://clubelo.com/{fed_url}"
        try:
            req = urllib.request.Request(url, headers=headers)
            html = urllib.request.urlopen(req, timeout=10).read().decode('utf-8', errors='ignore')
            for m in re.finditer(r'\{[^{}]*"Elo":\s*([0-9\.]+)[^{}]*"Name":\s*"([^"]+)"[^{}]*\}', html):
                store(m.group(2), m.group(1))
            for m in re.finditer(r'\{[^{}]*"Name":\s*"([^"]+)"[^{}]*"Elo":\s*([0-9\.]+)[^{}]*\}', html):
                store(m.group(1), m.group(2))
        except Exception as e:
            print(f"Warning: fallito scraping ClubElo per {fed_url}: {e}")
            
    if ratings:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": time.time(),
                "ratings": ratings
            }, f, indent=2)
            
    return ratings

_ELO_CACHE_MEM: Optional[Dict[str, float]] = None

def load_live_elo_database(force_refresh: bool = False) -> Dict[str, float]:
    global _ELO_CACHE_MEM
    if not force_refresh and _ELO_CACHE_MEM is not None:
        return _ELO_CACHE_MEM
    if not force_refresh and is_cache_valid():
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                _ELO_CACHE_MEM = data.get("ratings", {})
                return _ELO_CACHE_MEM
        except Exception:
            pass
            
    res = scrape_live_clubelo()
    if not res and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                res = data.get("ratings", {})
            os.utime(CACHE_FILE, None)
        except Exception:
            pass
    _ELO_CACHE_MEM = res
    return _ELO_CACHE_MEM
def get_live_elo(team_name: str, fallback: float = 1650.0) -> float:
    ratings = load_live_elo_database()
    k = clean_key(team_name)
    if k in ratings:
        return ratings[k]
        
    raw_lower = team_name.lower()
    if raw_lower in ratings:
        return ratings[raw_lower]
        
    for name_key, elo in ratings.items():
        if k in name_key or name_key in k:
            return elo
            
    return fallback

if __name__ == "__main__":
    db = load_live_elo_database(force_refresh=True)
    print(f"Database ELO live pronto: {len(db)} squadre censite.")
    for t in ["Genoa", "Frosinone", "Venezia", "Fiorentina", "Lazio", "Milan"]:
        print(f"  • {t:<15} -> Live Elo: {get_live_elo(t)}")
