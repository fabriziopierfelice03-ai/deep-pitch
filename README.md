# deep-pitch
End-to-end quantitative football betting framework in native PyTorch. Multi-task learning (xG regression + 1X2 logits) with custom entity embeddings, Poisson fair odds modeling, and an Agentic AI feature-ingestion pipeline. Verified via public paper trading.


# DeepPitch: Quantitative Multi-Task Football Value Betting Framework

DeepPitch is an end-to-end quantitative modeling pipeline designed to isolate and exploit market inefficiencies (Value Bets) across the top 5 European domestic leagues: Premier League (`E0`), Ligue 1 (`F1`), LaLiga (`SP1`), Serie A (`I1`), and Bundesliga (`D1`).

The engine pairs a native **Multi-Task PyTorch Neural Network** (simultaneous xG regression and 1X2 classification) with a **bivariate Poisson exact-score simulator** and an **autonomous Agentic AI pre-match feature retrieval workflow**.

---

## 📈 Live Paper Trading Ledger (Season 2026/2027)

To ensure zero lookahead bias and absolute empirical validation, all value bets are logged and committed to this repository **prior to the kickoff of each matchday**.

| Metric | Recorded Value |
| --- | --- |
| **Starting Bankroll** | €1,000.00 |
| **Current Bankroll** | **€1,117.00** |
| **Absolute P&L** | **+€117.00** |
| **Yield / ROI** | **+11.70%** |
| **Record** | 2W – 1L (66.7% Win Rate) |
| **Sample Start** | Matchday 2 |
| **Full Audit Trail** | [`tracking_2026_2027.csv`](https://www.google.com/search?q=./tracking_2026_2027.csv) |

---

## 🧠 Model Architecture & Pipeline Flow

The model accepts a **60-dimensional input vector** (57 domain-engineered numerical features + a 3-dimensional learned league entity embedding). A shared latent representation regularizes both heads simultaneously.

```
                    [ 57 Match Features ]  ──┐
                                             ├──> [ 60-dim Dense Vector ]
[ League Identifier ] ──> [ League Embedding (5x3) ] ──┘
                                   │
                                   ▼
                    [ Linear (60 -> 25) + Tanh ]
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
* **Cross-Entropy Loss:** Acts as a discrete anchor, preserving discriminative match-outcome boundaries in the latent layer $k$.

---

## 🔬 The 57-Feature Pre-Match Specification

Every match vector is deterministically assembled into the exact 57-feature layout below:

| Indices | Category | Description |
| --- | --- | --- |
| `[0 - 1]` | **Team Strength (Elo)** | `home_elo`, `away_elo`: Pre-match Elo ratings |
| `[2 - 5]` | **League Standings** | Table position and total points accumulated in the current season |
| `[6 - 9]` | **Recent Form (Points)** | Weighted average points across last 5 matches (Overall and Venue-specific) |
| `[10 - 17]` | **Advanced xG & xGA** | Weighted average xG created and conceded over the last 5 matches (Overall + Venue) |
| `[18 - 25]` | **Actual Goals (GF & GA)** | Weighted average actual goals scored and conceded over the last 5 matches |
| `[26 - 27]` | **Finishing Efficiency** | $xG - Goals$ differential (identifies regression to the mean and overperformance) |
| `[28 - 35]` | **Head-to-Head (H2H)** | Points, goals, and xG averages across the last 5 direct meetings (Overall + Venue) |
| `[36 - 40]` | **Fatigue & Context** | Rest days (capped at 14), matches played in last 14 days, and `is_derby` flag |
| `[41 - 46]` | **Table Pressure** | Point distances to Champions League, Europa League, and Relegation zones |
| `[47 - 48]` | **Season Progression** | Current round/matchday number and season half (`1` for 1–19, `2` for 20–38) |
| `[49 - 56]` | **In-Game Event Averages** | Weighted average shots, shots on target, corners, and cards over the last 5 matches |

---

## 🤖 Pre-Match Feature Retrieval via Agentic AI

Manually maintaining 57 continuous parameters for 40+ matches every weekend is inefficient and error-prone. Rather than relying on fragile scrapers vulnerable to DOM updates, anti-bot protections, and API rate limits, feature ingestion is handled by an **autonomous Agentic AI workflow**:

1. **Autonomous Search & Ingestion:** Every Friday, an external web-browsing agent queries verified football statistics providers (FBref, Understat, Transfermarkt).
2. **Deterministic Calculation:** The agent aggregates 5-game decay-weighted averages, calculates context parameters (table distances, fatigue thresholds), and validates type boundaries.
3. **Structured Vector Export:** The agent outputs a verified, zero-null Python list of exactly 57 floats ready for direct ingestion by `predict.py`.

A ready-to-use extraction prompt is provided in [`agent_prompt_template.md`](https://www.google.com/search?q=./agent_prompt_template.md).

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

### 3. Execution Threshold

A bet is tagged as positive Expected Value ($\text{EV}+$) only when:


$$\text{EV} = (P_{\text{model}} \times \text{Odds}_{\text{Bookmaker}}) - 1 \ge +0.05$$

---

## 📁 Repository Structure

```text
├── models/
│   └── modello_calcio_v1.pt        # Serialized weights, embeddings & scaler parameters
├── data/
│   └── matches_sample.csv          # Sample subset of the 2014-2026 enriched dataset
├── src/
│   ├── train_and_export.py         # Full PyTorch training routine & checkpoint exporter
│   └── predict.py                  # Standalone inference & dual pricing engine
├── agent_prompt_template.md        # Prompt schema for Agentic AI data extraction
├── tracking_2026_2027.csv          # Live, timestamped paper trading log
├── requirements.txt                # Minimal environment dependencies
└── README.md

```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/YOUR_USERNAME/deep-pitch-quant.git
cd deep-pitch-quant
pip install -r requirements.txt

```

### 2. Run Match Inference

```python
from src.predict import predici_partita

# League identifier: 'I1' (Serie A), 'E0' (Premier League), etc.
# feature_57: 57-element float vector assembled via Agentic AI
sample_vector = [
    1845.0, 1720.0,            # [0-1] Elo
    2.0, 7.0, 48.0, 36.0,      # [2-5] Table standings & points
    # ... Remaining 51 features in exact order
]

report = predici_partita("I1", sample_vector)

print("Predicted xG:", report["xG_Predetti"])
print("Neural 1X2 Probabilities:", report["Rete_Softmax_1X2"]["Probabilita_%"])
print("Neural 1X2 Fair Odds:", report["Rete_Softmax_1X2"]["Fair_Odds"])
print("Poisson Over 2.5 Fair Odds:", report["Poisson_Stats"]["Under_Over_2.5"]["Quota_Over"])

```

---

## ⚖️ Disclaimer

*This repository is maintained strictly for academic research, statistical modeling, and quantitative sports analytics. Historical paper trading yields do not guarantee future profitability. Nothing herein constitutes financial or wagering advice.*
