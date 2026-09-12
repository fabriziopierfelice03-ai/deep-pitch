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
    "nottinghamforest": "nottingham",
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
    "deportivoacoruna": "depacoruna",
    "deportivolacoruna": "depacoruna",
    "realsociedad": "sociedad",
    "realbetis": "betis",
    "rayovallecano": "vallecano",
    "bayernmunchen": "bayern",
    "bayernmunich": "bayern",
    "bayerleverkusen": "leverkusen",
    "borussiadortmund": "dortmund",
    "borussiamonchengladbach": "mgladbach",
    "eintrachtfrankfurt": "frankfurt",
    "scfreiburg": "freiburg",
    "tsghoffenheim": "hoffenheim",
    "1fcunionberlin": "unionberlin",
    "mainz05": "mainz",
    "1fckoln": "koln",
    "fckoln": "koln",
    "vfbstuttgart": "stuttgart",
    "hamburgersv": "hamburg",
    "parissaintgermain": "psg",
    "asmonaco": "monaco",
    "olympiquemarseille": "marseille",
    "olympique lyonnais": "lyon",
}

def strip_accents(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def clean_key(name: str) -> str:
    name = strip_accents(name.lower())
    clean = re.sub(r'[^a-z0-9]', '', name)
    return ALIAS_MAP.get(clean, clean)

def is_cache_valid() -> bool:
    if not os.path.exists(CACHE_FILE):
        return False
    mtime = os.path.getmtime(CACHE_FILE)
    return (time.time() - mtime) < (CACHE_TTL_HOURS * 3600)

def scrape_live_clubelo() -> Dict[str, float]:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    ratings = {}
    
    for l_code, fed_url in LEAGUE_URLS.items():
        url = f"https://clubelo.com/{fed_url}"
        try:
            req = urllib.request.Request(url, headers=headers)
            html = urllib.request.urlopen(req, timeout=10).read().decode('utf-8', errors='ignore')
            
            m1 = re.finditer(r'\{[^{}]*"Elo":\s*([0-9\.]+)[^{}]*"Name":\s*"([^"]+)"[^{}]*\}', html)
            for m in m1:
                elo = round(float(m.group(1)), 2)
                raw_name = m.group(2)
                k = clean_key(raw_name)
                ratings[k] = elo
                ratings[raw_name.lower()] = elo
                
            m2 = re.finditer(r'\{[^{}]*"Name":\s*"([^"]+)"[^{}]*"Elo":\s*([0-9\.]+)[^{}]*\}', html)
            for m in m2:
                raw_name = m.group(1)
                elo = round(float(m.group(2)), 2)
                k = clean_key(raw_name)
                ratings[k] = elo
                ratings[raw_name.lower()] = elo
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

def load_live_elo_database(force_refresh: bool = False) -> Dict[str, float]:
    if not force_refresh and is_cache_valid():
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("ratings", {})
        except Exception:
            pass
            
    return scrape_live_clubelo()

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
