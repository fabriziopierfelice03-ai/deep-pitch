# DeepPitch: Quantitative Multi-Task Football Value Betting Framework

DeepPitch is an end-to-end institutional-grade quantitative modeling and algorithmic wagering pipeline designed to isolate and exploit structural market inefficiencies (Value Bets) across the top 5 European domestic leagues: Premier League (`E0`), Ligue 1 (`F1`), LaLiga (`SP1`), Serie A (`I1`), and Bundesliga (`D1`).

The repository operates on a **dual-stack quantitative architecture**:
1. **Machine Learning Core (Python/PyTorch):** A joint **Multi-Task Neural Network** regularized with learned entity embeddings, producing concurrent continuous expected goals ($\hat{\lambda}_{\text{home}}, \hat{\lambda}_{\text{away}}$) and categorical outcome logits ($P_1, P_X, P_2$).
2. **High-Performance Simulation & Execution Engine (C++20):** An ultra-fast, cache-friendly backtester and **10,000-run Monte Carlo stress tester** assessing path-dependent drawdown and ruin probabilities across **16,266 historical matches in under 250 milliseconds**.
3. **Live Empirical Audit:** Transparent pre-match paper trading tracked in real time for the 2026/2027 season with zero lookahead bias.

---

## ⚡ High-Performance C++ Backtesting Engine (16,000+ Matches)

To rigorously validate model pricing over long horizons without Python interpreter overhead, DeepPitch includes a native **C++20 backtesting and Monte Carlo simulation engine** located in [`backtest/`](./backtest).

```
  [ 16,266 Historical Matches (2014-2026) ]
                     │
                     ▼
       [ C++ Vectorized Ingestion ]
                     │
        Filter: 1.60 < Odds < 3.10
        Edge:   EV > +13.0%
                     │
                     ▼
        [ 2,265 Qualified Value Bets ]
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
 [ Quarter-Kelly Staking ]  [ 10,000 Monte Carlo Shuffles ]
 (Max 1.0% Bankroll Cap)   (Ruin Probability & Drawdown Tails)
         │                       │
         ▼                       ▼
  +65.46% Bankroll ROI        0.00% Risk of Ruin
  +66.38 Flat PnL Units       37.42% Median Drawdown
```

### 1. Capital Allocation & Risk Management
The backtester applies a fractional **Quarter Kelly Criterion** with a hard risk ceiling to eliminate gambler's ruin while maximizing logarithmic geometric growth:

$$f^* = \frac{p \cdot q - 1}{q - 1}, \quad \text{Stake Fraction} = \min\left(0.25 \cdot f^*, \; 0.01\right)$$

* $p$: Model-derived true win probability.
* $q$: Decimal odds offered by the market.
* **Hard Capital Cap:** Individual position sizes never exceed **1.0%** of current bankroll under any circumstance.
* **Selection Filters:** Matches are only executed if $1.60 < \text{Odds} < 3.10$ and Expected Value $\text{EV} = (p \cdot q) - 1 > +13\%$.

### 2. Historical Backtest Performance (16,266 Matches)
Simulated across all top-5 European league matches from 2014 to 2026 in [`backtest/backtest2.0.csv`](./backtest/backtest2.0.csv):

| Metric | Result | Analytical Context |
| :--- | :--- | :--- |
| **Historical Matches Evaluated** | **16,266** | Complete dataset span across 5 European leagues |
| **Total Bets Placed ($\text{EV} > 13\%$)** | **2,265** | 13.9% market selection selectivity |
| **Bets Won / Win Rate** | **994 / 43.89%** | Solid strike rate on plus-money average prices |
| **Average Odds Taken** | **2.414** | Focus on under-the-radar value in the 1.60–3.10 range |
| **Initial Capital** | **€1,000.00** | Starting backtest bankroll |
| **Peak Bankroll** | **€1,912.01** | High-water mark during the simulation |
| **Final Bankroll** | **€1,654.57** | **+65.46% Net Capital Growth** |
| **Historical Maximum Drawdown** | **29.84%** | Controlled drawdown via Quarter-Kelly capping |
| **Flat Staking P&L** | **+66.38 units** | Independent proof of positive statistical expectation |

### 3. Monte Carlo Robustness Stress Test (10,000 Runs)
Historical execution paths are prone to sequence bias (e.g., encountering winning streaks early). To test model fragility, the C++ engine runs **10,000 randomized permutations** of the 2,265 qualified value bets, assessing path-dependent drawdown distributions and worst-case tail risks:

| Stress Test Metric | Value | Risk Interpretation |
| :--- | :--- | :--- |
| **Probability of Ruin ($< €200$)** | **0.00%** | Zero runs breached the 80% loss threshold across 10,000 trials |
| **Optimistic Drawdown ($5^{\text{th}}$ percentile)** | **27.28%** | Favorable variance sequence |
| **Median Drawdown ($50^{\text{th}}$ percentile)** | **37.42%** | Expected drawdown profile under standard randomness |
| **Pessimistic Drawdown ($95^{\text{th}}$ percentile)** | **51.84%** | Severe adverse variance cluster |
| **Worst Absolute Drawdown** | **72.99%** | Extreme tail event over 10,000 reshuffled permutations |
| **Engine Execution Speed** | **~200 ms** | 16,266 match parses + 10,000 Monte Carlo runs combined |

### 4. Compiling and Running the C++ Engine

```bash
# Compile with C++20 and full -O3 vectorization
g++ -O3 -std=c++20 backtest/backtest.cpp -o backtest/backtest

# Execute simulation (auto-detects dataset in current or parent directory)
./backtest/backtest
```

---

## 📈 Live Paper Trading Ledger (Season 2026/2027)

To ensure zero lookahead bias and absolute empirical validation, all value bets are logged and committed to this repository **prior to the kickoff of each matchday**.

| Metric | Recorded Value |
| :--- | :--- |
| **Starting Bankroll** | €1,000.00 |
| **Current Bankroll** | **€988.05** |
| **Absolute P&L** | **-€11.95** |
| **Yield / ROI** | **-1.20%** (Bankroll) / **-1.27%** (Yield on Turnover) |
| **Record** | 11W – 18L (37.9% Win Rate) |
| **Pending Bets** | 9 active (Weekend 1X2 Selections: 18–20 Sep 2026) |
| **Active Portfolio Heat** | €88.92 (9.00% Bankroll) |
| **Sample Start** | Matchday 2 |
| **Full Audit Trail** | [`tracking_2026_2027.csv`](./tracking_2026_2027.csv) |

---

## 🧠 Model Architecture & Pipeline Flow

The neural model accepts a **59-dimensional input vector** (56 domain-engineered numerical features + a 3-dimensional learned league entity embedding). A shared latent representation regularizes both heads simultaneously:

```
                    [ 56 Match Features ]  ──┐
                                             ├──> [ 59-dim Dense Vector ]
[ League Identifier ] ──> [ League Embedding (5x3) ] ──┘
                                   │
                                   ▼
                    [ Linear (59 -> 25) + Tanh ]
                                   │
                    [ Dropout (p=0.10) ]
                                   │
                    [ Linear (25 -> 16) + Tanh ]  <-- Shared Latent Core (k)
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
        [ Head 1: xG Regression ]          [ Head 2: 1X2 Classification ]
        Linear (16 -> 2)                   Linear (16 -> 3)
        Softplus Activation                Raw Logits
                 │                                   │
                 ▼                                   ▼
          Predicted xG                        Direct Softmax
       (Home xG, Away xG)                   P(1), P(X), P(2)
                 │                                   │
                 ▼                                   │
      [ 7x7 Poisson Matrix ]                         │
    (Secondary Markets: O/U, BTTS)                   │
                 │                                   │
                 └───────────────┬───────────────────┘
                                 ▼
                    [ Consensus Fair Odds Engine ]
                                 │
                                 ▼
                     [ Expected Value (EV+) ]
```

### 1. Robust Input Normalization
Features are normalized using empirical Median and Interquartile Range ($IQR$) computed strictly on the training partition:

$$\tilde{x} = \frac{x - \text{Median}(X_{\text{train}})}{IQR(X_{\text{train}}) + 1e-8}, \quad IQR = Q_{75} - Q_{25}$$

### 2. Multi-Task Joint Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SmoothL1}}(\hat{y}_{\text{xG}}, y_{\text{xG}}) + 0.30 \cdot \mathcal{L}_{\text{CrossEntropy}}(\hat{z}_{1X2}, y_{1X2})$$

* **Smooth L1 Loss ($\beta=1.0$):** Resists large outlier scores while maintaining stable gradient flow around zero.
* **Cross-Entropy Loss:** Acts as a discrete anchor, preserving discriminative match-outcome boundaries in the shared latent representation.

---

## 🔬 The 56-Feature Pre-Match Specification

Every match vector is deterministically assembled into the exact 56-feature layout below:

| Indices | Category | Description |
| :--- | :--- | :--- |
| `[0 - 1]` | **Team Strength (Elo)** | `home_elo`, `away_elo`: Pre-match Elo ratings |
| `[2 - 5]` | **League Standings** | Table position and total points accumulated in the current season |
| `[6 - 9]` | **Recent Form (Points)** | Weighted average points across last 5 matches (Overall and Venue-specific) |
| `[10 - 17]` | **Advanced xG & xGA** | Weighted average xG created and conceded over the last 5 matches (Overall + Venue) |
| `[18 - 25]` | **Actual Goals (GF & GA)** | Weighted average actual goals scored and conceded over the last 5 matches |
| `[26 - 27]` | **Finishing Efficiency** | $xG - Goals$ differential (identifies regression to the mean and overperformance) |
| `[28 - 35]` | **Head-to-Head (H2H)** | Points, goals, and xG averages across the last 5 direct meetings (Overall + Venue) |
| `[36 - 39]` | **Fatigue & Schedule Context** | Rest days (capped at 14) and matches played in the last 14 days |
| `[40 - 45]` | **Table Pressure** | Point distances to Champions League, Europa League, and Relegation zones |
| `[46 - 47]` | **Season Progression** | Current round/matchday number and season half (`1` for 1–19, `2` for 20–38) |
| `[48 - 55]` | **In-Game Event Averages** | Weighted average shots, shots on target, corners, and cards over the last 5 matches |

---

## 🤖 Pre-Match Feature Retrieval via Agentic AI

Manually maintaining 56 continuous parameters for 40+ matches every weekend is inefficient and error-prone. Rather than relying on fragile scrapers vulnerable to DOM updates, anti-bot protections, and API rate limits, feature ingestion is handled by an **autonomous Agentic AI workflow**:

1. **Autonomous Search & Ingestion:** Every Friday, an external agent queries verified football statistics providers (FBref, Understat, Transfermarkt, FotMob/Opta).
2. **Deterministic Calculation:** The agent aggregates 5-game decay-weighted averages, calculates context parameters (table distances, fatigue thresholds), and validates type boundaries.
3. **Structured Vector Export:** The agent outputs a verified, zero-null Python list of exactly 56 floats ready for direct ingestion by `predict.py`.

A ready-to-use extraction prompt is provided in [`agent_prompt_template.md`](./agent_prompt_template.md).

---

## ⚖️ Pricing Engine & Value Betting Logic

Market odds pricing requires selecting the right probability model for each specific market:

### 1. 1X2 Market (Direct Softmax Head)
Standard Poisson simulations notoriously underestimate low-scoring draws ($0\text{-}0, 1\text{-}1$) due to the assumption of independence between home and away goals. DeepPitch extracts 1X2 probabilities directly from the categorical output head:

$$P(y = c) = \frac{e^{\hat{z}_c}}{\sum_{j \in \{1, X, 2\}} e^{\hat{z}_j}}$$

### 2. Derivative Markets (Bivariate Poisson Simulation)
Predicted continuous xG outputs ($\hat{\lambda}_{\text{home}}, \hat{\lambda}_{\text{away}}$) generate an independent scoreline joint probability matrix across a $7 \times 7$ grid:

$$P(\text{Home}=i, \text{Away}=j) = \frac{\hat{\lambda}_H^i e^{-\hat{\lambda}_H}}{i!} \times \frac{\hat{\lambda}_A^j e^{-\hat{\lambda}_A}}{j!}$$

This distribution calculates fair odds for:
* **Over / Under 2.5 Goals:** $1 - \sum_{i+j < 2.5} P(i, j)$
* **Both Teams to Score (BTTS):** $\sum_{i \ge 1, j \ge 1} P(i, j)$

### 3. Execution Threshold & Strategy Filters
A market selection is tagged as an actionable Value Bet and executed in the strategy only when it satisfies all empirical criteria validated by the C++ simulation engine:

1. **Strict Edge Filter:**
   $$\text{EV} = (P_{\text{model}} \times \text{Odds}_{\text{Bookmaker}}) - 1 > +13.0\% \quad (+0.13)$$
2. **Odds Filtering Band:**
   $$1.60 < \text{Odds}_{\text{Bookmaker}} < 3.10$$
3. **Quarter-Kelly Sizing with Hard Ceiling:**
   $$f^* = \frac{p \cdot q - 1}{q - 1}, \quad \text{Stake Fraction} = \min\left(0.25 \cdot f^*, \; 0.01\right)$$
   Every position is strictly bounded by a **1.0%** bankroll ceiling to eliminate gambler's ruin.

---

## 📁 Repository Structure

```text
├── backtest/
│   ├── backtest.cpp                # Ultra-fast C++20 backtester & Monte Carlo simulator
│   └── backtest2.0.csv             # 16,266 historical matches dataset (odds, probs, results)
├── models/
│   └── modello_calcio_v1.pt        # Serialized weights, embeddings & scaler parameters
├── data/
│   └── matches_sample.csv          # Sample subset of the enriched historical dataset
├── src/
│   ├── train_and_export.py         # Full PyTorch training routine & checkpoint exporter
│   ├── feature_builder.py          # Deterministic feature assembly & validation engine
│   ├── predict.py                  # Standalone inference & dual pricing engine
│   └── decision_engine.py          # Value bet qualification & Quarter-Kelly sizing engine
├── tests/
│   └── test_feature_builder.py     # Deterministic feature assembly test suite
├── agent_prompt_template.md        # Prompt schema for Agentic AI data extraction
├── tracking_2026_2027.csv          # Live, timestamped paper trading log
├── requirements.txt                # Minimal environment dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Run the C++ Backtesting & Monte Carlo Engine

```bash
# Compile and run the C++ engine
g++ -O3 -std=c++20 backtest/backtest.cpp -o backtest/backtest
./backtest/backtest
```

### 2. Run Python Match Inference

```python
from src.predict import predici_partita

# League identifier: 'I1' (Serie A), 'E0' (Premier League), etc.
# feature_56: 56-element float vector assembled via Agentic AI
sample_vector = [
    1845.0, 1720.0,            # [0-1] Elo
    2.0, 7.0, 48.0, 36.0,      # [2-5] Table standings & points
    # ... Remaining 50 features in exact order
]

report = predici_partita("I1", sample_vector)

print("Predicted xG:", report["xG_Predetti"])
print("Neural 1X2 Probabilities:", report["Rete_Softmax_1X2"]["Probabilita_%"])
print("Neural 1X2 Fair Odds:", report["Rete_Softmax_1X2"]["Fair_Odds"])
print("Poisson Over 2.5 Fair Odds:", report["Poisson_Stats"]["Under_Over_2.5"]["Quota_Over"])
```

---

## ⚖️ Disclaimer

*This repository is maintained strictly for academic research, statistical modeling, and quantitative sports analytics. Historical paper trading yields and backtest results do not guarantee future profitability. Nothing herein constitutes financial or wagering advice.*
