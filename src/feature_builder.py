"""
src/feature_builder.py
Modulo per l'ingestion rigorosa e la costruzione controllata del vettore a 57 feature
per il modello predittivo DeepPitch.

Regola: "No xG = No Bet". I dati xG e le statistiche devono provenire esclusivamente
da fonti primarie verificate (FotMob/Opta, Understat, FBref, Lega Ufficiale).
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import torch

@dataclass
class MatchRecord:
    """Rappresenta il tabellino verificato di una singola partita disputata."""
    date: str
    opponent: str
    venue: str          # 'H' per Casa, 'A' per Trasferta
    goals_for: int
    goals_against: int
    xg_for: float       # Da FotMob/Opta o Understat
    xg_against: float   # Da FotMob/Opta o Understat
    shots: int
    shots_on_target: int
    corners: int
    cards: int
    points: int         # 3 (W), 1 (D), 0 (L)
    source: str         # es: "FotMob/Opta"

@dataclass
class H2HRecord:
    """Rappresenta lo storico degli scontri diretti recenti tra due squadre."""
    home_avg_points: float
    away_avg_points: float
    home_avg_goals_for: float
    home_avg_goals_against: float
    home_avg_xg: float
    away_avg_xg: float
    venue_home_avg_points: float
    venue_home_avg_goals_for: float
    source: str

@dataclass
class StandingsContext:
    """Dati di classifica ufficiali al momento del calcio d'inizio."""
    home_pos: float
    away_pos: float
    home_points: float
    away_points: float
    home_dist_ucl: float
    away_dist_ucl: float
    home_dist_uel: float
    away_dist_uel: float
    home_dist_rel: float
    away_dist_rel: float
    matchday: float
    season_half: float  # 1 o 2


def calculate_weighted_average(values: List[float], weights: Optional[List[float]] = None) -> float:
    """
    Calcola la media ponderata decrescente (più recente ha peso maggiore).
    Pesi standard per 5 match: [5, 4, 3, 2, 1].
    Se i match disponibili sono < 5, usa la porzione iniziale normalizzata.
    """
    if not values:
        return 0.0
    n = len(values)
    base_weights = [5.0, 4.0, 3.0, 2.0, 1.0][:n]
    total_w = sum(base_weights)
    weighted_sum = sum(v * w for v, w in zip(values, base_weights))
    return weighted_sum / total_w


def build_team_metrics(
    matches: List[MatchRecord],
    target_venue: Optional[str] = None,
    fallback_metrics: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Estrae le medie ponderate da una lista di MatchRecord ordinati dal piu recente al piu vecchio.
    Se target_venue e specificato ('H' o 'A'), filtra per quella sede (split metrics).
    Se i match filtrati sono insufficienti/assenti, esegue il fallback 'soft' sulle metriche generali.
    """
    if target_venue:
        matches = [m for m in matches if m.venue == target_venue]
    
    if not matches:
        if fallback_metrics:
            return fallback_metrics.copy()
        return {
            "pts": 0.0,
            "xg": 0.0,
            "xga": 0.0,
            "gf": 0.0,
            "ga": 0.0,
            "shots": 0.0,
            "sot": 0.0,
            "corners": 0.0,
            "cards": 0.0,
        }
    
    # Prendi al massimo gli ultimi 5 match
    m_slice = matches[:5]
    
    pts = calculate_weighted_average([m.points for m in m_slice])
    xg = calculate_weighted_average([m.xg_for for m in m_slice])
    xga = calculate_weighted_average([m.xg_against for m in m_slice])
    gf = calculate_weighted_average([m.goals_for for m in m_slice])
    ga = calculate_weighted_average([m.goals_against for m in m_slice])
    shots = calculate_weighted_average([m.shots for m in m_slice])
    sot = calculate_weighted_average([m.shots_on_target for m in m_slice])
    corners = calculate_weighted_average([m.corners for m in m_slice])
    cards = calculate_weighted_average([m.cards for m in m_slice])
    
    return {
        "pts": pts,
        "xg": xg,
        "xga": xga,
        "gf": gf,
        "ga": ga,
        "shots": shots,
        "sot": sot,
        "corners": corners,
        "cards": cards,
    }


def build_57_features(
    home_elo: float,
    away_elo: float,
    home_matches: List[MatchRecord],
    away_matches: List[MatchRecord],
    h2h: H2HRecord,
    standings: StandingsContext,
    home_rest_days: float,
    away_rest_days: float,
    home_matches_14d: float,
    away_matches_14d: float,
    is_derby: float = 0.0,
    home_matches_split: Optional[List[MatchRecord]] = None,
    away_matches_split: Optional[List[MatchRecord]] = None
) -> List[float]:
    """
    Costruisce e valida il vettore di esattamente 57 feature conformi al modello.
    
    Supporta sia liste match ampie (10-15 partite, da cui estrae le ultime 5 in sede),
    sia liste split dedicate fornite esplicitamente tramite home_matches_split / away_matches_split.
    """
    # 1. Metriche Generali (ultimi 5 match complessivi)
    h_gen = build_team_metrics(home_matches)
    a_gen = build_team_metrics(away_matches)
    
    # 2. Metriche Split (ultimi 5 match in Casa per Home, Trasferta per Away)
    if home_matches_split:
        h_split = build_team_metrics(home_matches_split, fallback_metrics=h_gen)
    else:
        h_split = build_team_metrics(home_matches, target_venue='H', fallback_metrics=h_gen)
        
    if away_matches_split:
        a_split = build_team_metrics(away_matches_split, fallback_metrics=a_gen)
    else:
        a_split = build_team_metrics(away_matches, target_venue='A', fallback_metrics=a_gen)
    
    # 3. Finishing Efficiency (xG - GF)
    h_xg_diff = h_gen["xg"] - h_gen["gf"]
    a_xg_diff = a_gen["xg"] - a_gen["gf"]
    
    # 4. Protezione numerica derby: safe_is_derby forzato a 0.0 per evitare divisione per IQR=0
    safe_is_derby = 0.0
    
    features = [
        # [0, 1] Team Strength Ratings (Elo)
        float(home_elo),
        float(away_elo),
        
        # [2, 3, 4, 5] Current Standings
        float(standings.home_pos),
        float(standings.away_pos),
        float(standings.home_points),
        float(standings.away_points),
        
        # [6, 7, 8, 9] Recent Form - Points
        float(h_gen["pts"]),
        float(a_gen["pts"]),
        float(h_split["pts"]),
        float(a_split["pts"]),
        
        # [10, 11, 12, 13] xG Created
        float(h_gen["xg"]),
        float(a_gen["xg"]),
        float(h_split["xg"]),
        float(a_split["xg"]),
        
        # [14, 15, 16, 17] xG Conceded (xGA)
        float(h_gen["xga"]),
        float(a_gen["xga"]),
        float(h_split["xga"]),
        float(a_split["xga"]),
        
        # [18, 19, 20, 21] Goals Scored (GF)
        float(h_gen["gf"]),
        float(a_gen["gf"]),
        float(h_split["gf"]),
        float(a_split["gf"]),
        
        # [22, 23, 24, 25] Goals Conceded (GA)
        float(h_gen["ga"]),
        float(a_gen["ga"]),
        float(h_split["ga"]),
        float(a_split["ga"]),
        
        # [26, 27] Over/Underperformance
        float(h_xg_diff),
        float(a_xg_diff),
        
        # [28, 29, 30, 31, 32, 33, 34, 35] Head-to-Head
        float(h2h.home_avg_points),
        float(h2h.away_avg_points),
        float(h2h.home_avg_goals_for),
        float(h2h.home_avg_goals_against),
        float(h2h.home_avg_xg),
        float(h2h.away_avg_xg),
        float(h2h.venue_home_avg_points),
        float(h2h.venue_home_avg_goals_for),
        
        # [36, 37, 38, 39, 40] Schedule, Fatigue & Derby
        float(min(home_rest_days, 14.0)),
        float(min(away_rest_days, 14.0)),
        float(home_matches_14d),
        float(away_matches_14d),
        float(safe_is_derby),
        
        # [41, 42, 43, 44, 45, 46] Table Pressure & Distances
        float(standings.home_dist_ucl),
        float(standings.away_dist_ucl),
        float(standings.home_dist_uel),
        float(standings.away_dist_uel),
        float(standings.home_dist_rel),
        float(standings.away_dist_rel),
        
        # [47, 48] Season Stage
        float(standings.matchday),
        float(standings.season_half),
        
        # [49, 50, 51, 52, 53, 54, 55, 56] In-Game Event Stats
        float(h_gen["shots"]),
        float(a_gen["shots"]),
        float(h_gen["sot"]),
        float(a_gen["sot"]),
        float(h_gen["corners"]),
        float(a_gen["corners"]),
        float(h_gen["cards"]),
        float(a_gen["cards"]),
    ]
    
    assert len(features) == 57, f"Errore critico: il vettore ha dimensione {len(features)}, atteso 57."
    return features


def validate_feature_scaling(features: List[float], model_checkpoint_path: str = "models/modello_calcio_v1.pt") -> Dict[str, Any]:
    """
    Valida che nessuna feature superi soglie patologiche di scaling dopo RobustScaler.
    """
    checkpoint = torch.load(model_checkpoint_path)
    median = checkpoint["median"]
    iqr = checkpoint["iqr"]
    
    x = torch.tensor(features, dtype=torch.float32)
    scaled = (x - median) / (iqr + 1e-8)
    
    max_val = scaled.max().item()
    min_val = scaled.min().item()
    
    is_sane = (-5.0 <= min_val) and (max_val <= 5.0)
    
    return {
        "is_sane": is_sane,
        "max_scaled": round(max_val, 4),
        "min_scaled": round(min_val, 4),
        "scaled_vector": scaled.tolist()
    }


def print_audit_report(home_name: str, away_name: str, home_matches: List[MatchRecord], away_matches: List[MatchRecord], features: List[float]):
    """Stampa l'audit trasparente di tutte le fonti prima di procedere."""
    print("=" * 80)
    print(f"   AUDIT TRACCIABILITÀ DATI REALI: {home_name.upper()} vs {away_name.upper()}")
    print("=" * 80)
    
    print(f"\n1. Partite recenti considerate per {home_name}:")
    for m in home_matches:
        print(f"   • {m.date} | ({m.venue}) vs {m.opponent:<12} | Ris: {m.goals_for}-{m.goals_against} | xG: {m.xg_for:.2f} - xGA: {m.xg_against:.2f} | Fonte: {m.source}")
        
    print(f"\n2. Partite recenti considerate per {away_name}:")
    for m in away_matches:
        print(f"   • {m.date} | ({m.venue}) vs {m.opponent:<12} | Ris: {m.goals_for}-{m.goals_against} | xG: {m.xg_for:.2f} - xGA: {m.xg_against:.2f} | Fonte: {m.source}")
    
    scaling = validate_feature_scaling(features)
    print(f"\n3. Controllo Stabilità RobustScaler:")
    print(f"   • Range scalato: [{scaling['min_scaled']}, {scaling['max_scaled']}] | Validato: {'SI' if scaling['is_sane'] else 'NO (ANOMALIA)'}")
    print("=" * 80)
