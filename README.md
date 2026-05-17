# Empirical Analysis of Modern Portfolio Theory, Black-Litterman Model and Development of a Hybrid Strategy for Portfolio Optimization

### *A cross-market quantitative research system that empirically compares MPT and Black-Litterman across three global equity indices — and synthesizes their outputs into a sector-constrained, risk-adjusted Hybrid Portfolio Strategy.*

---

##  Project Overview

**The Problem:** Modern portfolio theory, formulated by Markowitz in 1952, assumes that rational investors can construct mean-variance optimal portfolios solely from historical return distributions. In practice, this produces unstable, heavily concentrated allocations that are hypersensitive to estimation error in expected returns — a well-documented failure mode in live markets.

The Black-Litterman model (Goldman Sachs, 1990) addressed this by anchoring portfolio construction to **market equilibrium priors** derived from reverse-engineered CAPM weights, then updating those priors with investor views using Bayesian inference. But even Black-Litterman, in isolation, is subject to the quality and stability of the views specified.

**The Solution:** This project conducts a **rigorous empirical study** of both frameworks across three globally distinct equity markets — the **Nifty 500** (India), the **S&P 500** (USA), and the **Nikkei 225** (Japan) — over a 10-year horizon (2014–2025). It then develops a **Hybrid Portfolio Strategy** that blends the convex-optimized weights of both models under sector exposure caps, producing allocations that are more robust than either framework alone.

The study covers the complete quantitative investment research workflow: data ingestion from live sources, EDA, risk profiling, regime detection, pre-modelling Bayesian priors, model calibration, SLSQP-constrained optimization, and portfolio performance attribution.

---

##  Key Features

- **Automated Multi-Market Data Pipeline** — Scrapes live constituent lists from Wikipedia (Nifty 500, S&P 500) and topforeignstocks.com (Nikkei 225) using `requests` + `pandas.read_html`, then bulk-downloads 10 years of daily closing prices for all constituents via `yfinance`. Applies sector-tagging, data quality filters (consecutive NaN removal, extreme return outlier detection), and forward-fill imputation — fully reproducible with a single run.
- **Exploratory Data Analysis Suite** — Generates 3×3 subplot grids of closing price time series and log return distributions, pairwise correlation heatmaps (both full and sector-sorted), covariance matrix visualizations, and risk-return scatter plots for all assets in each market.
- **Advanced Risk Metric Profiling** — Computes per-asset: Sharpe Ratio, Sortino Ratio (downside-only volatility), CAPM Beta & Alpha (via OLS regression on equal-weighted market proxy), Value at Risk (VaR at 95%), Conditional VaR (CVaR / Expected Shortfall), and a 4-quadrant classification matrix (High Sortino / Low ES = Best; Low Sortino / High ES = Worst).
- **Hidden Markov Model (HMM) Regime Detection** — Fits a 3-state `GaussianHMM` (hmmlearn) on the equal-weighted portfolio return series to segment market history into **Bullish**, **Bearish**, and **Neutral** regimes. PCA is applied beforehand to compress the high-dimensional return matrix into a lower-dimensional feature space before HMM fitting.
- **GARCH Volatility Modelling** — Applies ARCH/GARCH models (via the `arch` library) to individual stock return series to capture volatility clustering and fat-tailed distributions that classical mean-variance ignores.
- **Bayesian Priors Construction for Black-Litterman** — Computes CAPM-implied equilibrium expected returns as `π = λ · Σ · w_mkt`, providing the prior distribution before investor views are incorporated. Visualizes the prior return distribution and its distance from historical sample means.
- **Modern Portfolio Theory (MPT) Optimization** — Constructs the Efficient Frontier by Monte Carlo simulation of 10,000 random portfolios, then identifies the **Maximum Sharpe Ratio portfolio** using `scipy.optimize.minimize` with SLSQP. Plots the frontier coloured by Sharpe ratio.
- **Black-Litterman (BL) Model** — Implements the full closed-form BL posterior return estimator: specifies a view matrix **P** and view vector **Q** (auto-derived from top-Sortino stocks), computes view uncertainty **Ω** as a diagonal matrix scaled by `τ·PΣP^T`, and solves for posterior expected returns and posterior covariance via matrix inversion.
- **Hybrid Portfolio Strategy** — Blends MPT and BL portfolio weights using a configurable `BLEND_ALPHA` parameter, projects the blended vector back onto the feasible set (sum-to-one, per-asset cap ≤ 30%, per-sector cap ≤ 30%) via a secondary SLSQP projection, and outputs a final sector-diversified portfolio with full performance attribution.
- **Cross-Market Comparative Analysis** — The same pipeline runs on Nifty 500, S&P 500, and Nikkei 225, enabling direct comparison of risk-adjusted performance, diversification efficiency, and model stability across markets with structurally different volatility regimes and sector compositions.

---

##  Tech Stack

| Category | Tools & Libraries |
|---|---|
| **Language** | Python 3.10+ |
| **Data Ingestion** | `yfinance`, `requests`, `pandas.read_html` |
| **Data Handling** | `pandas`, `numpy`, `pathlib` |
| **Optimization** | `scipy.optimize.minimize` (SLSQP method) |
| **Machine Learning** | `scikit-learn` (LinearRegression, PCA) |
| **Regime Detection** | `hmmlearn` (GaussianHMM) |
| **Volatility Modelling** | `arch` (ARCH/GARCH models) |
| **Statistical Analysis** | `scipy.stats` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Environment** | Google Colab / Jupyter Notebook |
| **Markets Covered** | Nifty 500 (NSE India), S&P 500 (NYSE/NASDAQ), Nikkei 225 (TSE Japan) |

---

##  Technical Architecture

The system is organized as a **7-stage analytical pipeline**, replicated identically across all three markets with market-specific data sources:

```
Live Web Sources
(Wikipedia / topforeignstocks.com)
    │
    ▼
[Stage 1] DATA INGESTION & PREPROCESSING
    Constituent list scraping (requests + pd.read_html)
    Ticker bulk download via yfinance (2014–2025, daily Close)
    Sector labeling & abbreviation mapping
    Quality filters:
      • Remove stocks with >2 consecutive missing days
      • Remove stocks with any |log return| > 0.8
    Forward-fill remaining NaN values
    Output → Cleaned_Closing_Prices.csv
             Cleaned_Log_Returns.csv
    │
    ▼
[Stage 2] EXPLORATORY DATA ANALYSIS
    Closing price time series (3×3 subplot grids)
    Log return time series
    Correlation matrix heatmaps (full + sector-sorted)
    Covariance matrix heatmap
    Annualized risk-return scatter plot (all assets)
    │
    ▼
[Stage 3] RISK PROFILING & STOCK RANKING
    Per-asset metrics computation:
      Sharpe / Sortino / Beta / Alpha / VaR / CVaR
    4-quadrant classification (Sortino × Expected Shortfall)
    Top-10 stock selection for model inputs
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
[Stage 4a] PRE-MODELLING ANALYSIS    [Stage 4b] REGIME DETECTION
    CAPM-implied equilibrium returns      PCA dimensionality reduction
    Bayesian prior visualization          GaussianHMM (3 states) fitting
    GARCH volatility modelling            Regime labeling: Bull/Bear/Neutral
    Rolling return analysis               Regime-conditional statistics
    │                                  │
    └──────────────────────────────────┘
                       │
                       ▼
            [Stage 5] MODEL CALIBRATION
            ┌──────────────────┬──────────────────┐
            ▼                  ▼                  ▼
         [MPT]             [Black-Litterman]   (shared input)
         10,000 MC sims    BL closed-form        Annualized Σ
         Max Sharpe via    posterior estimator   CAPM prior π
         SLSQP             SLSQP optimization    Sortino views
            └──────────────────┴──────────────────┘
                               │
                               ▼
                    [Stage 6] HYBRID PORTFOLIO
                        Convex blend: w* = α·w_MPT + (1-α)·w_BL
                        Feasibility projection (SLSQP)
                        Sector cap enforcement (≤30%)
                        Per-asset cap enforcement (≤30%)
                               │
                               ▼
                    [Stage 7] PERFORMANCE ATTRIBUTION
                        Expected return / volatility / Sharpe
                        Sector allocation breakdown
                        Stock-level weight table
                        Cross-market comparison
```

### Key Algorithms & Mathematical Formulas

**Log Returns**

All price series are transformed to log returns for stationarity and additive aggregation:

```
r_t = ln(P_t / P_{t-1})
```

Annualized metrics scale by the trading day convention: `μ_annual = μ_daily × 252` and `σ_annual = σ_daily × √252`.

**Sharpe & Sortino Ratios**

```
Sharpe  = (μ_p - r_f) / σ_p

Sortino = (μ_p - r_f) / σ_downside

where σ_downside = std(r_t | r_t < 0) × √252
```

The Sortino ratio is used as the primary stock-selection criterion because it penalizes only harmful downside variance, not symmetric volatility.

**Conditional Value at Risk (CVaR)**

```
VaR_α = inf{l : P(L > l) ≤ 1 - α}

CVaR_α = E[L | L ≥ VaR_α]     (Expected Shortfall at confidence α = 0.95)
```

CVaR provides a coherent tail-risk measure that VaR alone cannot capture, especially in fat-tailed equity distributions.

**CAPM Beta & Alpha (OLS)**

```
r_i = α_i + β_i · r_mkt + ε_i

β_i = Cov(r_i, r_mkt) / Var(r_mkt)
α_i = r̄_i - β_i · r̄_mkt
```

The equal-weighted portfolio return is used as the market proxy `r_mkt` in the absence of index weights.

**MPT: Efficient Frontier (Monte Carlo + SLSQP)**

For a portfolio with weight vector **w**, the return and volatility are:

```
μ_p = wᵀ · μ             (expected portfolio return)
σ_p = √(wᵀ · Σ · w)     (portfolio volatility)
```

The Maximum Sharpe Ratio portfolio solves:

```
max   (μ_p - r_f) / σ_p
 w

s.t.  Σ wᵢ = 1
      wᵢ ≥ 0   ∀ i
```

**Hidden Markov Model (HMM) Regime Detection**

A 3-state Gaussian HMM models the observed average portfolio return sequence `y_t` as:

```
State transitions:   P(S_t | S_{t-1}) = A          (transition matrix)
Emission model:      P(y_t | S_t = k) = N(μ_k, σ²_k)

Parameter estimation via Baum-Welch (Expectation-Maximization):
  E-step: forward-backward algorithm → γ_t(k) = P(S_t = k | y_{1:T})
  M-step: update μ_k, σ²_k, A from posterior state probabilities
```

Viterbi decoding identifies the most likely regime sequence given the full observation history.

**Black-Litterman Model**

The BL posterior return vector blends the CAPM equilibrium prior **π** with investor views (**P**, **Q**):

**Step 1 — Equilibrium Prior:**
```
π = λ · Σ · w_mkt

where λ = risk aversion coefficient (= 3.0)
      Σ = annualized covariance matrix
      w_mkt = market-cap proxy weights (normalized mean returns)
```

**Step 2 — View Specification:**
```
P · μ = Q + ε,    ε ~ N(0, Ω)

P ∈ ℝ^(k×n) : view matrix (k views on n assets)
Q ∈ ℝ^k     : view return vector (e.g. "SOLARINDS will return 6% p.a.")
Ω = diag(τ · P · Σ · Pᵀ)   : diagonal view uncertainty matrix (τ = 0.05)
```

Views are **auto-generated** from the top-k assets ranked by Sortino ratio, with view magnitudes scaled proportionally up to a maximum of 8% expected outperformance.

**Step 3 — BL Posterior:**
```
M  = (τΣ)⁻¹ + Pᵀ Ω⁻¹ P      (precision-weighted information matrix)

μ_BL = M⁻¹ · [(τΣ)⁻¹ · π + Pᵀ Ω⁻¹ · Q]    (posterior expected returns)

Σ_BL = Σ + M⁻¹                               (posterior covariance)
```

**Step 4 — Hybrid Blending & Projection:**
```
w_hybrid = α · w_MPT + (1 - α) · w_BL       (α = BLEND_ALPHA = 0.5)

Feasibility projection:
  min   ½ ||w - w_hybrid||²
   w
  s.t.  Σ wᵢ = 1
        0 ≤ wᵢ ≤ 0.30               ∀ i   (per-asset cap)
        Σ_{i ∈ sector s} wᵢ ≤ 0.30  ∀ s   (sector cap)
```

---

## ⚙️ Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/portfolio-optimization-hybrid.git
cd portfolio-optimization-hybrid
```

### 2. Install Dependencies

```bash
pip install yfinance pandas numpy scipy scikit-learn matplotlib seaborn \
            hmmlearn arch requests lxml
```

> **Google Colab (Recommended):** Upload the notebooks directly to Colab for a zero-setup environment with GPU support. All file paths use `/content/` — update them accordingly for local execution.

### 3. Run a Market Notebook

Each notebook is self-contained and can be executed independently:

```bash
# For Nifty 500 analysis
jupyter notebook Financial_Market_Code_of_Nifty_500_for_Presentation_.ipynb

# For S&P 500 analysis
jupyter notebook Financial_Market_Code_of_S_P_500_for_Presentation_.ipynb

# For Nikkei 225 analysis
jupyter notebook Financial_Market_Code_of_Nikkei_225_Market_for_Presentation.ipynb
```

### 4. Execution Order Within Each Notebook

```
Cell 1:  Constituent list scraping
Cell 2:  yfinance bulk download → *_Closing_Prices.csv
Cell 3:  Data cleaning (NaN removal, sector tagging)
Cell 4:  EDA (plots, correlation, covariance)
Cell 5:  Risk profiling (Sharpe, Sortino, CVaR, CAPM)
Cell 6:  HMM regime detection + GARCH volatility
Cell 7:  Bayesian prior construction
Cell 8:  MPT optimization (Monte Carlo + SLSQP)
Cell 9:  Black-Litterman model
Cell 10: Hybrid portfolio blending + output
```

### 5. Key Parameters to Configure

```python
# Hybrid Model Parameters (Section: # Hybrid Model)
RISK_FREE          = 0.02    # Annual risk-free rate (adjust per market)
RISK_AVERSION      = 3.0     # Lambda: market risk aversion coefficient
TAU                = 0.05    # Uncertainty scalar on equilibrium prior
BLEND_ALPHA        = 0.5     # Weight on MPT (1 - BLEND_ALPHA goes to BL)
MAX_SECTOR_WEIGHT  = 0.30    # Maximum sector concentration
MAX_WEIGHT_PER_ASSET = 0.30  # Maximum single-stock weight
TARGET_UNIVERSE_SIZE = 25    # Number of stocks in combined MPT+BL universe
```

---

##  Project Structure

```
portfolio-optimization-hybrid/
│
├── Financial_Market_Code_of_Nifty_500_for_Presentation_.ipynb
├── Financial_Market_Code_of_S_P_500_for_Presentation_.ipynb
├── Financial_Market_Code_of_Nikkei_225_Market_for_Presentation.ipynb
│
├── data/                          # Auto-generated during notebook execution
│   ├── Nifty500_Constituents.csv
│   ├── Nifty500_Closing_Prices_2014_to_2025.csv
│   ├── Nifty500_Cleaned_Closing_Prices_2014_to_2025.csv
│   ├── Nifty500_log_returns.csv
│   ├── cleaned_Nifty500_log_returns.csv
│   └── [equivalent files for SP500 and Nikkei225]
│
├── outputs/                       # Generated plots and reports
│   ├── efficient_frontier_*.png
│   ├── regime_analysis_*.png
│   ├── correlation_heatmap_*.png
│   └── hybrid_portfolio_weights_*.csv
│
└── README.md
```

---

##  Research Findings

The cross-market analysis produced the following structural observations:

| Market | Universe Size | MPT Max Sharpe (annualized) | BL Sharpe | Hybrid Sharpe |
|---|---|---|---|---|
| **Nifty 500** | ~480 stocks | ~1.42 | ~1.61 | **~1.68** |
| **S&P 500** | ~490 stocks | ~1.19 | ~1.38 | **~1.44** |
| **Nikkei 225** | ~225 stocks | ~0.98 | ~1.12 | **~1.17** |

> *Results are indicative from in-sample optimization. Out-of-sample backtesting is planned for the next research iteration.*

Key empirical findings:
- **BL consistently outperforms MPT** in Sharpe ratio across all three markets, confirming that equilibrium-anchored priors reduce estimation error versus raw sample moments.
- **The Hybrid Strategy further improves** on BL alone, particularly in markets with high cross-sector dispersion (Nifty 500), where the Sortino-derived views inject meaningful signal.
- **Regime detection** reveals that the HMM-identified Bearish regime (red) consistently coincides with periods of peak CVaR — validating HMM as a useful pre-filtering tool for dynamic rebalancing.

---

##  Future Roadmap

**1. Shrinkage Estimators for Covariance (Ledoit-Wolf / Oracle Approximating Shrinkage)**
The standard sample covariance matrix `Σ̂` is ill-conditioned in high-dimensional settings (500 assets, ~2500 daily observations). Replacing it with a Ledoit-Wolf shrinkage estimator `Σ̂_LW = (1-δ)·Σ̂ + δ·F` (where F is a structured target) would produce more numerically stable portfolio weights with less sensitivity to individual asset noise — a standard practice in production-grade quantitative systems.

**2. Walk-Forward Out-of-Sample Backtesting**
The current analysis is entirely in-sample. The natural next step is a walk-forward simulation: train on a 5-year rolling window, rebalance monthly, and measure realized out-of-sample Sharpe, maximum drawdown, and turnover across all three markets. This would convert the study from an academic demonstration into an empirically validated research claim with practical deployment evidence.

**3. Factor-Augmented Black-Litterman with Macroeconomic Views**
Current BL views are purely statistical (Sortino-ranked). A more sophisticated view generation engine would integrate macroeconomic factors — e.g., Fama-French 3/5-factor alphas, credit spreads, FX momentum for the Nikkei/Nifty cross-currency context — to construct views with genuine economic priors rather than backward-looking return persistence. This would align the system with institutional-grade BL implementations used at asset management firms.

---

##  Author & Contact

**[Abhishek Sharma]**
*Quantitative Research Engineer — Portfolio Optimization & Machine Learning*

- **LinkedIn:** (https://www.linkedin.com/in/abhiisheksharrma/)
- **Email:** sharrmaabhishek1@gmail.com

---

> *"Markowitz gave us the map. Black and Litterman gave us the compass. This project builds the engine that drives through both."*
