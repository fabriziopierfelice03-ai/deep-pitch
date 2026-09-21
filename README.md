# DeepPitch: Quantitative Football Value Betting with Bayesian Shrinkage & C++20 Monte Carlo Engine

DeepPitch is an end-to-end institutional-grade quantitative modeling and algorithmic wagering pipeline designed to detect and exploit structural market inefficiencies (Value Bets) across European football leagues.

The framework operates on a **modernized dual-stack quantitative architecture**:
1. **Machine Learning Core (Python/PyTorch):** A streamlined, universal **1X2 Categorical Neural Network** (`56 -> 25 -> 16 -> 3`) regularized with AdamW, label smoothing, and early stopping on validation Brier score.
2. **Empirical Bayesian Shrinkage Layer:** A calibrated logistic regression layer that blends the neural network's predictive alpha with de-vigged market consensus odds:
   $$\text{logit}(p_{\text{calib}}) = 0.0806 + 0.6173 \cdot \text{logit}(p_{\text{model}}) + 0.5035 \cdot \text{logit}(p_{\text{book}})$$
   yielding a balanced weighting of **55.1% Model Alpha** and **44.9% Market Consensus**.
3. **100% Blind Out-of-Sample Test Set Validation:** Validated on **1,627 untouched out-of-sample fixtures** (10% temporal split), achieving **+17.08% net ROI** across 384 qualified bets at $\text{EV} > 5.5\%$ with a **22.29% maximum drawdown**.
4. **High-Performance Simulation & Execution Engine (C++20):** An ultra-fast, cache-friendly native backtester featuring a **10,000-run Monte Carlo stress test** (Bootstrap resampling and trade-order permutation) executing 16,000+ fixtures in milliseconds.
5. **Live Empirical Audit:** Transparent pre-match paper trading tracked in real time for the 2026/2027 season with zero lookahead bias.

---

## ⚡ High-Performance C++20 Backtesting & Monte Carlo Engine

To rigorously stress-test strategy viability without lookahead bias or data contamination, DeepPitch evaluates performance exclusively on the **100% blind out-of-sample test set (1,627 matches)** using a native **C++20 engine** located in [`backtest/`](./backtest).

```
          [ 16,266 Historical Matches (2014-2026) ]
                             │
            Strict 90% Temporal Split (Cutoff)
                             │
                             ▼
      [ 1,627 Blind Out-of-Sample Test Matches (10%) ]
                             │
               Filter: 1.60 < Odds < 3.10
               Edge:   EV > +5.5% (Target Strategy)
                             │
                             ▼
              [ 384 Qualified Value Bets ]
              (183 Won | 47.66% Win Rate | Avg Odds: 2.24)
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
     [ Quarter-Kelly Staking ]    [ 10,000 Monte Carlo Runs ]
     (1.5% Bankroll Hard Cap)     (Bootstrap & Permutation)
               │                           │
               ▼                           ▼
        +17.08% Net ROI             0.00% Risk of Ruin
        +11.90 Flat PnL Units       25.19% Median Drawdown
        22.29% Max Drawdown         €1,167 Median Bankroll
```

### 1. Capital Allocation & Risk Management
The backtester applies a fractional **Quarter-Kelly Criterion** with a hard risk ceiling to eliminate gambler's ruin while maximizing geometric bankroll growth:

$$f^* = \frac{p \cdot q - 1}{q - 1}, \quad \text{Stake Fraction} = \min\left(0.25 \cdot f^*, \; 0.015\right)$$

* $p$: Model probability calibrated via Bayesian Shrinkage.
* $q$: Decimal odds offered by the market.
* **Hard Capital Cap:** Individual position sizes never exceed **1.5%** of current bankroll under any circumstance.
* **Selection Filters:** Matches are only executed if $1.60 < \text{Odds} < 3.10$ and Expected Value $\text{EV} = (p \cdot q) - 1 > +5.5\%$.

### 2. Out-of-Sample Sensitivity Analysis (1,627 Test Matches)
The table below illustrates the strategy's sensitivity across various EV thresholds on the virgin out-of-sample test set:

| EV Threshold | Bets | Won | Win Rate | Avg Odds | Final Bankroll | ROI (%) | Max Drawdown | Flat PnL |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2.0%** | 556 | 259 | 46.58% | 2.23 | €1,061.48 | +6.15% | 25.16% | +1.03u |
| **3.0%** | 504 | 235 | 46.63% | 2.24 | €1,083.98 | +8.40% | 26.26% | +1.96u |
| **4.0%** | 455 | 216 | 47.47% | 2.24 | €1,158.86 | +15.89% | 26.20% | +9.22u |
| **5.0%** | 407 | 191 | 46.93% | 2.24 | €1,087.33 | +8.73% | 24.13% | +4.99u |
| **5.5% (Target)** | **384** | **183** | **47.66%** | **2.24** | **€1,170.77** | **+17.08%** | **22.29%** | **+11.90u** |
| **6.0%** | 367 | 174 | 47.41% | 2.25 | €1,134.11 | +13.41% | 24.35% | +9.58u |
| **7.0%** | 336 | 158 | 47.02% | 2.25 | €1,078.20 | +7.82% | 23.78% | +6.20u |
| **8.0%** | 302 | 139 | 46.03% | 2.26 | €1,003.88 | +0.39% | 25.61% | +0.95u |

### 3. Dual-Module Monte Carlo Stress Testing (10,000 Runs)

To evaluate tail risk and path dependency across thousands of possible trajectories, the C++ engine executes two complementary 10,000-iteration Monte Carlo procedures:

#### Module A: Bootstrap Resampling (Sampling with Replacement)
Simulates 10,000 alternative seasons by sampling 384 bets with replacement from the observed empirical trade distribution:

| Metric | Simulated Value | Risk Interpretation |
| :--- | :---: | :--- |
| **Probability of Ruin ($< €200$)** | **0.00%** | Zero runs breached the 80% loss barrier |
| **Pessimistic Bankroll ($5^{\text{th}}$ percentile)** | **€701.90** | Unfavorable variance regime over a full season |
| **Median Bankroll ($50^{\text{th}}$ percentile)** | **€1,167.04** | Central expectation aligns closely with realized +17.08% return |
| **Optimistic Bankroll ($95^{\text{th}}$ percentile)** | **€1,977.36** | High-water mark under positive variance |
| **Median Maximum Drawdown** | **25.19%** | Expected drawdown depth across simulated paths |
| **Pessimistic Drawdown ($95^{\text{th}}$ percentile)** | **43.58%** | Drawdown boundary in adverse volatility regimes |
| **Worst-Case Absolute Drawdown** | **67.31%** | Theoretical tail scenario across 3,840,000 total simulated wagers |

#### Module B: Trade-Order Permutation (Reshuffling without Replacement)
Quantifies sequence risk and losing-streak clustering by randomly shuffling the chronological order of the 384 observed historical bets:

| Permutation Metric | Value | Risk Interpretation |
| :--- | :---: | :--- |
| **Mild Drawdown ($5^{\text{th}}$ percentile)** | **17.20%** | Favorable dispersion of wins and losses |
| **Median Drawdown ($50^{\text{th}}$ percentile)** | **24.65%** | Standard sequence-dependent drawdown |
| **Severe Drawdown ($95^{\text{th}}$ percentile)** | **35.60%** | Clustered adverse variance |
| **Worst-Case Permutation Drawdown** | **48.41%** | Maximum clustering of losing bets in 10,000 random sequences |

### 4. Compiling and Running the C++ Engine

```bash
# Compile with C++20 and full -O3 optimization
g++ -O3 -std=c++20 backtest/backtest.cpp -o backtest/backtest

# Execute simulation (auto-detects dataset in current or parent directory)
./backtest/backtest
```

---

## ⚖️ Bayesian Shrinkage Layer via Logistic Regression

In sports betting markets, deep neural networks trained exclusively on match data can suffer from localized overconfidence and longshot bias. Conversely, bookmaker odds embed massive aggregate liquidity and collective market wisdom, but carry bookmaker overround (vigorish) and public sentiment distortions.

To harvest true predictive alpha while ensuring rigorous calibration, DeepPitch incorporates an **Empirical Bayesian Shrinkage Layer** fitted via logistic regression on 14,400 historical training matches:

```
[ 56 Match Features ] ──> [ Neural Net (56 -> 25 -> 16 -> 3) ] ──> Raw Logits ──> p_model
                                                                                     │
                                                                                     ├──> [ Bayesian Shrinkage Layer ] ──> Calibrated p
[ Market Odds (1X2) ] ──> [ De-Vigging (Overround Removal) ]   ──> Implied Odds ──> p_book   (Model 55.1% | Market 44.9%)
```

### Mathematical Formulation:
1. **De-vigging Bookmaker Implied Probabilities:**
   $$I_k = \frac{1}{q_k}, \quad s = \sum_{j \in \{1, X, 2\}} I_j, \quad p_{k, \text{book}} = \frac{I_k}{s}$$

2. **Logit Transformation:**
   $$\text{logit}(p) = \ln\left(\frac{p}{1 - p}\right)$$

3. **Empirical Bayesian Combination:**
   $$z_k = \alpha + \beta_{\text{model}} \cdot \text{logit}(p_{k, \text{model}}) + \beta_{\text{book}} \cdot \text{logit}(p_{k, \text{book}})$$
   where:
   * $\alpha = 0.0806$ (Calibration Intercept)
   * $\beta_{\text{model}} = 0.6173 \implies \mathbf{55.1\%}$ relative weight
   * $\beta_{\text{book}} = 0.5035 \implies \mathbf{44.9\%}$ relative weight

4. **Normalized Calibrated Distribution:**
   $$p_k^{\text{calib}} = \frac{\sigma(z_k)}{\sum_j \sigma(z_j)}, \quad \text{Fair Odds}_k = \frac{1}{p_k^{\text{calib}}}$$

This Bayesian formulation ensures that the model only takes positions when its statistical edge is strong enough to deviate meaningfully from the market consensus.

---

## 🧠 Neural Network Architecture & Training Pipeline

### 1. Architectural Streamlining
In earlier iterations, the network incorporated league entity embeddings and an auxiliary xG regression head. Extensive empirical ablation revealed:
* **Removal of League Entity Embeddings:** Entity embeddings artificially fragmented the dataset and prevented the model from generalizing to international fixtures, cup competitions, or unseen leagues. The streamlined 56-feature vector is fully universal across world football.
* **Removal of Auxiliary xG Regression Head:** Multi-task loss balancing between continuous xG and discrete 1X2 cross-entropy introduced conflicting gradient dynamics. Eliminating the xG head concentrated 100% of network capacity on the true wagering objective: the 1X2 categorical distribution.

### 2. Network Specifications
```
Input Layer:        56 Domain-Engineered Numerical Features
Hidden Layer 1:     Linear(56 -> 25) + Tanh Activation + Dropout(p=0.10)
Hidden Layer 2:     Linear(25 -> 16) + Tanh Activation
Output Layer:       Linear(16 -> 3) (Unbounded Logits for 1, X, 2)
```

### 3. Training Protocol
* **Dataset:** 16,266 historical matches (2014–2026).
* **Temporal Split:** Strict 80% Train (12,987 matches), 10% Validation (1,623 matches), and 10% Test (1,627 matches).
* **Optimization:** `AdamW` ($\text{lr} = 0.003$, $\text{weight\_decay} = 0.008$).
* **Loss Function:** `CrossEntropyLoss(label_smoothing=0.06)` with gradient norm clipping ($\text{max\_norm} = 1.0$).
* **Early Stopping:** Monitored strictly on **Validation Brier Score** with patience of 800 steps, preserving the test set completely pristine.

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

## 📁 Repository Structure

```text
├── backtest/
│   ├── backtest.cpp                # High-performance C++20 backtester & dual Monte Carlo engine
│   └── backtest2.0.csv             # 16,266 matches dataset with calibrated model probabilities
├── models/
│   └── modello_calcio_v1.pt        # Serialized weights (56 -> 25 -> 16 -> 3) & scaler parameters
├── data/
│   └── matches_sample.csv          # Sample subset of the enriched historical dataset
├── src/
│   ├── train_and_export.py         # PyTorch training routine with 80/10/10 split & early stopping
│   ├── feature_builder.py          # Deterministic feature assembly & validation engine
│   ├── predict.py                  # Match inference with Bayesian Shrinkage calibration
│   └── decision_engine.py          # Value bet qualification & Quarter-Kelly sizing engine
├── tests/
│   └── test_feature_builder.py     # Deterministic feature assembly & calibration test suite
├── agent_prompt_template.md        # Prompt schema for Agentic AI data extraction
├── tracking_2026_2027.csv          # Live, timestamped paper trading log
├── requirements.txt                # Minimal environment dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Run the C++ Backtesting & Monte Carlo Engine
```bash
# Compile and execute C++ backtest
g++ -O3 -std=c++20 backtest/backtest.cpp -o backtest/backtest
./backtest/backtest
```

### 2. Run Python Match Inference with Bayesian Shrinkage
```python
from src.predict import predici_partita, stampa_report_partita

# 56 domain features assembled via src.feature_builder
features_56 = [
    1845.0, 1720.0,            # [0-1] Elo
    2.0, 7.0, 48.0, 36.0,      # [2-5] Table standings & points
    # ... Remaining 50 features in exact order
]

# Market odds [Home, Draw, Away]
market_odds = [2.70, 3.30, 2.60]

# Run prediction with Bayesian calibration
report = predici_partita(features_56, market_odds=market_odds)
stampa_report_partita(report, "Roma", "Milan")
```

### 3. Run Unit Tests
```bash
py -3.14 -m pytest tests/
```

---

## ⚖️ Disclaimer

*This repository is maintained strictly for academic research, statistical modeling, and quantitative sports analytics. Historical backtest results and simulations do not guarantee future profitability. Nothing herein constitutes financial or wagering advice.*
