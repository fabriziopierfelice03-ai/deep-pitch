# DeepPitch v2: Quantitative Football Pricing, Residual-Offset Ensembling & Real-Time Microstructure Engine

DeepPitch is an end-to-end quantitative modeling, pricing, and execution pipeline for Top-5 European football (`E0`, `I1`, `SP1`, `D1`, `F1`), built on strictly zero-leakage statistical inference, live availability/lineup shock modeling, multi-bookmaker best-odds execution, and a coupled C++20 in-play hedging daemon.

---

## Executive Summary & Architecture

DeepPitch **v2** replaces the legacy single-MLP architecture (`v1`, blind test Brier `0.58463`) with a **five-stage institutional-style quantitative pipeline** that slashes out-of-sample Brier error by **`-0.0110`** (to **`0.57365`** on the 2024/25 blind test set, matching bookmaker accuracy in the Premier League at `0.56587` vs `0.56582`) while remaining **100% automated on free public data feeds** (FotMob/Opta, ClubElo, `football-data.co.uk`, BetExplorer):

```text
  [ Past Pinnacle 1X2 + O2.5 + AHh (t' < t) ]     [ Opta/FotMob Shot-Level npxG + xGOT (t' < t) ]
                       │                                                  │
                       ▼                                                  ▼
     [ Online Kalman Log-λ + Exact 35d Ridge ]        [ Exact Rolling 120d Weighted Poisson-MLE ]
     (Implied Attack/Defense, Steam, Surprise)        (Opponent-Adjusted Dixon-Coles on npxG/xGOT)
                       │                                                  │
                       └───────────────────────┬──────────────────────────┘
                                               ▼
                        [ OOF-Calibrated Generative Base p_base (T = 0.903) ]
                                               │
            ┌──────────────────────────────────┴──────────────────────────────────┐
            ▼                                                                     ▼
  [ Exact Table Motivation ]                                      [ Orthogonal Residual Features ]
  (Math Champ / Relegated /                                       (Market-vs-xG, Finishing, GK,
   Locked UCL / Dead Rubber)                                       Sharp Steam, Rest, Congestion)
            │                                                                     │
            └──────────────────┬──────────────────────────────────────────────────┘
                               ▼
         [ 5-Head Residual-Offset Ensemble (init_score / base_margin = log p_base) ]
         ├── Head 0: Generative Dixon-Coles Base            (weight 0.15)
         ├── Head 1: Grouped-Ridge Offset Linear            (weight 0.35)
         ├── Head 2: Residual-Offset LightGBM (5-seed)      (weight 0.15)
         ├── Head 3: Residual-Offset XGBoost  (5-seed)      (weight 0.20)
         └── Head 4: Residual-Offset PyTorch MLP (10-seed)  (weight 0.15)
                               │  (SLSQP Simplex + OOF Temperature Scaling T = 1.013)
                               ▼
            [ Live FotMob Fresh-Injury & Predicted/Confirmed Lineup Delta ]
            • Separates FRESH starter shocks (played last 5, now out/rotated)
              from long-term injuries already priced into Past-Market ratings
            • Bounded Logit Adjustment (±0.16) + Adverse-Selection Trap Guard
                               │
                               ▼
            [ 11-Book ADM Best-Odds Line Shopping + Quarter-Kelly Sizing ]
            • Band: EV ∈ [1.0%, 8.0%) | Odds ∈ (1.35, 6.00) | Hard Cap: 3.0%
            • Auto-Audit & CLV Tracking (data/v2_scan_audit.csv + tracking_2026_2027.csv)
                               │
                               ▼
            [ Autonomous C++20 Live Risk Sentinel (Private Execution Engine) ]
```

---

## Out-of-Sample Quantitative Benchmarks

### 1. Probabilistic Accuracy (Brier Score — Strictly Zero Current-Match Odds Leakage)

Evaluated on the exact **16,268 Top-5 historical dataset** (`2015–2025`) using **5-Fold Expanding-Window Out-of-Fold CV** (`8,140` OOF matches from `2019–2024`) with the `2024/25` season (`1,627` matches) held 100% blind, plus a brand-new blind evaluation on the live **`2026/27` season** (`Aug–Nov 2026`):

| Model / Benchmark | Val `2023/24` (n=1,627) | Blind Test `2024/25` (n=1,627) | Delta vs `v1` (Test) |
| :--- | :---: | :---: | :---: |
| **DeepPitch `v1`** (Legacy 56-feat MLP) | `0.58296` | `0.58463` | — |
| **Block #1 + #2** (Multi-scale EWMA + Opta `npxG/xGOT` + DC, *no past market*) | `0.57547` | `0.57985` | `-0.00478` |
| **Generative Base** (62% Past-Market + 26% Exact Opta `npxG` DC + 11% Exact 35d Mkt) | `0.57432` | `0.57472` | `-0.00991` |
| **DeepPitch `v2` Production 5-Head Ensemble** | **`0.57307`** | **`0.57365`** | **`-0.01098`** |
| **DeepPitch `v2` Best Single Head (`Offset_Linear`)** | **`0.57280`** | **`0.57323`** | **`-0.01140`** |
| **Bookmaker Consensus Avg (Shin De-Vig)** | `0.57193` | `0.56945` | *Market Benchmark* |
| **Pinnacle Opening Odds (Shin De-Vig)** | `0.57177` | `0.56921` | *Sharp Open Benchmark* |
| **Pinnacle Closing Odds (Shin De-Vig)** | `0.56965` | `0.56684` | *Gold Standard* |
| **Pinnacle Open + `v2` Orthogonal Residuals (`npxG/xGOT/steam/rest`)** | — | **`0.56908`** | **`-0.00013` below Pinnacle Open** |

**League-by-League Breakdown on Blind Test `2024/25` (`v2` vs Bookmaker Avg Shin):**
* **Premier League (`E0`, n=339):** `0.56587` vs `0.56582` (**`+0.00005` — virtual parity**)
* **Serie A (`I1`, n=358):** `0.57191` vs `0.56918` (`+0.00273`)
* **Ligue 1 (`F1`, n=302):** `0.56749` vs `0.56401` (`+0.00348`)
* **Regular Season (`Matchdays 1–30`, n=1,241):** `0.57447` vs `0.57239` (**`+0.00208`**)

### 2. Live Blind Validation on `2026/27` (`Aug–Nov 2026`) — Impact of Fresh Injuries & Lineups

Tested strictly out-of-sample on `136` completed `2026/27` Top-5 matches with prior FotMob lineup histories cached in `data/scanner_cache/` and live `football-data.co.uk` `2627` odds:

| Configuration (Blind `2026/27`) | Brier All (`n=136`) | Brier Impacted (`n=94`) | Est. ADM-Best (`EV ∈ [1%, 8%)`) | World Max (`EV ∈ [1%, 8%)`) |
| :--- | :---: | :---: | :---: | :---: |
| **`v2` without Injuries/Lineups** | `0.55369` | `0.54503` | `42` bets \| `45.2%` WR \| `+5.44u` | `38` bets \| `47.4%` WR \| `+6.22u` |
| **`v2` WITH Fresh Injuries + TrapGuard** | **`0.55265`** (`-0.00104`) | **`0.54332`** (`-0.00171`) | `36` bets (`2` traps blocked) \| **`50.0%` WR** \| `+4.71u` | `36` bets (`3` traps blocked) \| **`58.3%` WR** \| **`+16.76u`** |
| **Bookmaker Consensus Avg (Shin)** | `0.55060` | `0.53928` | — | — |

### 3. Six-Season Out-of-Sample Economic Evaluation (`2019–2025`, `EV ∈ [1%, 8%)`)

Why **`EV ∈ [1.0%, 8.0%)`**? Empirical diagnostics across 9,768 OOS matches reveal classic **stale-price adverse selection** in the extreme `EV ≥ 8%` tail: when a pre-match model disagrees with Pinnacle by `> 8%`, it is disproportionately due to breaking matchday news ($\mathcal{F}_t \setminus \mathcal{F}_{t-1}$: late injuries, suspensions, or Champions League rotation). Filtering the `[0%, 1%)` vig-noise floor and the `≥ 8%` toxic tail — combined with the live FotMob injury guard — isolates genuine positive-CLV edge:

| Season (OOS, `v2` @ Est. ADM-Best, `EV ∈ [1%, 8%)`) | Bets | Win Rate | Flat PnL | Flat Yield | CLV vs PinClose |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `2019/20` (OOF Fold 1) | 763 | 39.1% | `+1.00u` | `+0.13%` | `+0.19%` |
| `2020/21` (OOF Fold 2 — *COVID empty stadiums*) | 777 | 36.7% | `-49.00u` | `-6.31%` | `-0.05%` |
| `2021/22` (OOF Fold 3) | 841 | 35.9% | `+10.78u` | `+1.28%` | `+0.45%` |
| `2022/23` (OOF Fold 4) | 848 | 34.7% | `+40.22u` | `+4.74%` | `+0.96%` |
| `2023/24` (Validation) | 817 | 37.6% | `+42.58u` | `+5.21%` | `+0.85%` |
| `2024/25` (Blind Test) | 743 | 40.6% | `+49.56u` | `+6.67%` | `+0.18%` |
| **All 6 Seasons (`2019–2025`)** | **4,789** | **37.3%** | **`+95.14u`** | **`+1.99%`** | **`+0.44%`** |
| **Post-COVID 4 Seasons (`2021–2025`)** | **3,249** | **37.1%** | **`+143.14u`** | **`+4.41%`** | **`+0.62%`** |

*(On World Max best-odds across all 6 seasons `2019–2025`: `5,084` bets, **`+123.36u (+2.43%)`**, **`CLV = +2.90%`**, Quarter-Kelly `€1,000 → €1,781`; Post-COVID `2021–2025`: **`+3.05%`**, Quarter-Kelly **`€2,181`**.)*

---

## Mathematical & Statistical Methodology

### 1. Exact Rolling Weighted Poisson-MLE Dixon-Coles (`npxG` & `xGOT`)
Raw rolling averages fail to adjust for opponent strength and suffer from penalty variance (`~0.77 xG` noise per penalty). DeepPitch v2 solves a time-decayed Ridge-regularized Poisson MLE via **Newton-Raphson (IRLS)** strictly on past matches ($t_k < t$, half-life $\tau = 120$ days):

$$\lambda_H = \exp\!\left(\mu + \tfrac{1}{2}\text{HFA} + a_H - d_A\right), \qquad \lambda_A = \exp\!\left(\mu - \tfrac{1}{2}\text{HFA} + a_A - d_H\right)$$

with exponential decay weights $w_k = 2^{-(t - t_k)/\tau}$ and **Dixon-Coles (1997)** low-score correlation adjustment $\tau_\rho(x,y)$ ($\rho = -0.065$) on $(0,0), (1,0), (0,1), (1,1)$.

### 2. Past Market-Implied Strength (Zero Current-Match Leakage)
For every **completed** past fixture $t' < t$, DeepPitch v2 analytically inverts Pinnacle's historical `1X2`, `Over/Under 2.5`, and `Asian Handicap` lines into implied expected goals $(\lambda_{\text{mkt},H}, \lambda_{\text{mkt},A})$:
1. Total implied goals $L = \lambda_H + \lambda_A$ by inverting $P(G_{\text{tot}} \ge 3) = 1 - e^{-L}(1 + L + L^2/2) = p_{\text{O2.5}}$.
2. Implied goal margin $d = \lambda_H - \lambda_A$ via Dixon-Coles inversion of $(p_1 - p_2)$ blended with `−AHh`.
3. Recursive Kalman-style state updates on $\log \lambda_{\text{mkt}}$ plus an exact 35-day exponentially-weighted Ridge solver, automatically capturing squad value, transfers, and persistent injuries without match-outcome noise.

### 3. Residual-Offset Learning & Grouped-Ridge Regularization
In low-SNR domains like football, feeding 100+ raw features into tree/neural models induces variance explosion. Instead, all ML heads initialize from the calibrated generative base logits $z_{\text{base}} = \log(p_{\text{base}})$ (`init_score` in LightGBM, `base_margin` in XGBoost, additive skip in PyTorch/Linear) and learn strictly regularized residual corrections:

$$z_{\text{final}} = \alpha \cdot \log(p_{\text{base}}) + b + f_\theta(x_{\text{mot}}, x_{\text{res}})$$

In `Offset_Linear`, two distinct L2 penalties prevent over-shrinking sparse high-impact end-of-season motivation signals (`ex_champ_diff`, `ex_locked_diff`, `ex_drel_diff`, `ex_off_diff` with $\lambda_{\text{mot}} = 100$) while heavily shrinking noisy all-season residuals ($\lambda_{\text{res}} = 7000$).

### 4. Live FotMob Fresh-Injury & Lineup Delta ($\mathcal{F}_t \setminus \mathcal{F}_{t-1}$)
Long-term absences are already priced into past market lines. `src/injury_lineup.py` cross-references FotMob's pre-match `unavailable` list and `predicted`/`confirmed` XI against each team's **last 5 completed starting XIs** (already cached locally):
* Isolates **freshly missing regular starters** (started $\ge 2$ of last 5 matches and now injured, suspended, doubtful, or rotated out).
* Applies a bounded, conservative logit adjustment (`±0.16` max) weighted by missing XI market value and starter count.
* Triggers a hard **Adverse-Selection Trap Guard** whenever a candidate pick backs a team freshly missing $\ge 2$ regular starters or $\ge 18\%$ of starting XI market value.

### 5. Shin (1993) De-Vigging, 11-Book ADM Line Shopping & Quarter-Kelly Sizing
* **Shin De-Vigging:** Accounts for favorite-longshot bias by solving numerically for the informed-trading parameter $z \in [0, 1)$ such that $\sum_c p_c(z) = 1$.
* **ADM Best-Odds Execution:** Queries BetExplorer's live AJAX bookmaker feed across 11 active Italian ADM bookmakers (`888sport`, `bet365`, `bwin.it`, `Eurobet.it`, `GoldBet`, `Lottomatica`, `Netwin`, `Planetwin365`, `Sisal`, `SNAI`, `William Hill`), cutting effective execution overround from `~6.5%` average down to `~3.6–4.4%`.
* **Quarter-Kelly Capital Allocation:**
  $$f^* = \frac{p \cdot q - 1}{q - 1}, \qquad \text{Stake Fraction} = \min\!\left(0.25 \cdot f^*, \; 0.030\right)$$

---

## Autonomous In-Play Risk Sentinel (C++20 Engine)

While DeepPitch v2 handles pre-match pricing and selection, open positions are monitored in real time by **Live Sentinel**, a decoupled private **C++20** daemon (`live-sentinel`) ingesting 1Hz live odds feeds:
* **Mathematical Green Book Hedging:** Locks risk-free profit $S_{\text{hedge}} = \text{round}_{0.50}\!\left(\frac{S_0 \cdot O_0}{O_{\text{opp}}(t)}\right)$ protected by an analytical Red Book break-even guard $O_{\text{opp}}(t) > \frac{O_0}{O_0 - 1}$.
* **Two-Level Microstructure State Machine:** Continuous economic activation gate $A(t) \ge 0.50$ paired with Track A (extreme asymmetry bypass) and Track B (4-minute rolling velocity, pressure stasis trigger, and anti-panic circuit breaker).
* **Atomic Sync:** Synchronizes settled hedges to `tracking_2026_2027.csv` and dispatches sub-second Telegram execution receipts.

---

## Repository Structure

```text
├── src/
│   ├── train_v2.py             # DeepPitch v2 training & 5-fold expanding OOF calibration -> bundle export
│   ├── v2_state_engine.py      # Live auto-sync football-data + FotMob npxG/xGOT + exact rolling solvers
│   ├── injury_lineup.py        # Live FotMob fresh-injury, suspension, rotation delta & trap guard
│   ├── predict_v2.py           # 5-Head Residual-Offset v2 inference engine
│   ├── scanner.py              # Unified pre-match Top-5 scanner (v2 default, 11-book ADM best-odds, Telegram, CLV audit)
│   ├── decision_engine.py      # EV band [1%, 8%) qualification & Quarter-Kelly sizing (3% cap)
│   ├── elo_fetcher.py          # Live ClubElo scraper with 12h local cache
│   ├── settle_matches.py       # Automated match settlement & bankroll ledger updater
│   ├── feature_builder.py      # Legacy v1 56-feature deterministic builder
│   ├── predict.py              # Legacy v1 inference + Shin (1993) de-vigging utility
│   └── train_and_export.py     # Legacy v1 PyTorch training script
├── models/
│   ├── deep_pitch_v2_bundle.pkl # Production DeepPitch v2 serialized 5-head ensemble & scalers
│   └── modello_calcio_v1.pt     # Legacy v1 checkpoint
├── backtest/
│   ├── backtest.cpp            # Native C++20 backtester & 10,000-run dual Monte Carlo stress tester
│   └── backtest2.0.csv         # Historical benchmark dataset
├── tests/
│   ├── test_v2_pipeline.py     # Unit & E2E tests for v2 ensemble, injuries, motivation & EV guard
│   ├── test_scanner.py         # Scanner, odds matching & tracking idempotency tests
│   ├── test_decision_engine.py # Quarter-Kelly & cap tests
│   └── test_feature_builder.py # Feature vector & Shin de-vigging tests
├── tracking_2026_2027.csv      # Public live paper-trading ledger (compatible with live-sentinel)
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pre-Match Top-5 Scanner (DeepPitch v2)
```bash
# Dry-run scan across E0, I1, SP1, D1, F1 (v2 engine + live injuries + 11-book ADM best-odds)
python src/scanner.py

# Production run: appends picks to tracking_2026_2027.csv, syncs GitHub & sends Telegram digest
python src/scanner.py --publish

# Legacy comparison mode (v1)
python src/scanner.py --engine v1
```

### 3. Run the Full Test Suite
```bash
pytest tests/ -v
```

### 4. Compile & Run the C++20 Monte Carlo Backtester
```bash
g++ -O3 -std=c++20 backtest/backtest.cpp -o backtest/backtest
./backtest/backtest
```

---

## Disclaimer

*This repository is maintained strictly for academic research, statistical modeling, and quantitative sports analytics. Historical backtests and paper-trading audits do not guarantee future profitability. Nothing herein constitutes financial or wagering advice.*
