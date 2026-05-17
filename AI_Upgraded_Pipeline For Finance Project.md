# AI-Upgraded Portfolio Optimization Pipeline
## From Classical Quant Research to Production-Grade Hybrid AI System

---

> *"As an AI Research Intern, here's my upgraded pipeline — a complete, production-ready architecture that transforms your classical MPT + Black-Litterman research into a deployable, AI-augmented portfolio optimization system."*

---

## Section 1: Analysis Summary

### Strengths of the Original Project

The original three-notebook system represents a rigorous and well-structured piece of quantitative research. Its greatest strengths are the **end-to-end reproducibility** of the data pipeline (live constituent scraping → yfinance bulk download → deterministic cleaning rules) and the **theoretical depth** of the modelling stack — the closed-form Black-Litterman posterior implementation, the SLSQP-constrained hybrid projection, and the HMM regime detector are all production-quality components rarely seen in portfolio research at this level. The cross-market comparative design (Nifty 500, S&P 500, Nikkei 225 simultaneously) provides genuine empirical breadth that most single-market studies lack. The risk profiling layer — Sharpe, Sortino, CVaR, the four-quadrant ES/Sortino classifier — is well-calibrated for institutional-grade stock screening.

### Identified Gaps and Upgrade Opportunities

Three structural gaps limit the system's real-world impact. First, **all results are in-sample**: there is no walk-forward backtesting, so reported Sharpe ratios cannot be attributed to genuine predictive skill versus in-sample curve-fitting — a critical distinction for any production deployment. Second, the **Black-Litterman views are purely statistical** (backward-looking Sortino scores), meaning the system has no forward-looking signal; replacing these with ML-predicted excess returns would transform it from a descriptive to a predictive system. Third, the **covariance matrix uses the raw sample estimator**, which is ill-conditioned at 480+ assets and will produce unstable weights; shrinkage is non-negotiable at this dimensionality. The upgrade pipeline below addresses all three gaps while preserving every reusable component from the original work.

---

## Section 2: Kaggle Training Pipeline

### Master Architecture Overview

```
[Phase 0]  Environment & Data Infrastructure
     │
     ▼
[Phase 1]  Multi-Market Data Ingestion & Unified Preprocessing
     │
     ▼
[Phase 2]  Feature Engineering (Technical + Macro + Cross-Market)
     │
     ▼
[Phase 3]  AI Model Training — Return Prediction (BL View Generator)
     │         ├── XGBoost Baseline
     │         ├── LSTM Temporal Model
     │         └── iTransformer (Multivariate Attention)
     ▼
[Phase 4]  AI-Augmented Portfolio Construction
     │         ├── Shrinkage Covariance (Ledoit-Wolf)
     │         ├── Black-Litterman with AI Views
     │         ├── MPT Max-Sharpe (SLSQP)
     │         └── Hybrid Blending + Feasibility Projection
     ▼
[Phase 5]  Backtesting Engine (Walk-Forward, No Look-Ahead)
     │
     ▼
[Phase 6]  Benchmark Comparison & Performance Attribution
     │
     ▼
[Phase 7]  Model Serialization & Output Artifacts
```

---

### Phase 0: Kaggle Environment Setup

**Step 0.1 — Kaggle Notebook Configuration**

- Runtime: GPU (P100 or T4) — required for LSTM/iTransformer training
- Accelerator: GPU
- Internet: ON (for yfinance API calls during data ingestion)
- Persistence: Enable to cache downloaded price data between runs
- RAM allocation: High-memory mode preferred for 500-stock covariance matrices

**Step 0.2 — Dependency Declaration**

Declare a requirements block at the top of the notebook with pinned versions for full reproducibility. Core packages required beyond Kaggle defaults:

- `yfinance` — price data ingestion
- `hmmlearn` — regime detection (carry over from original)
- `arch` — GARCH volatility estimation (carry over)
- `scikit-learn` — preprocessing, Ledoit-Wolf shrinkage, cross-validation
- `xgboost` — gradient boosting return predictor
- `torch` — LSTM and iTransformer architectures
- `optuna` — hyperparameter optimization
- `pyfolio-reloaded` or manual implementation — tearsheet metrics
- `plotly` — interactive visualizations for outputs
- `joblib` — model serialization

**Step 0.3 — Reproducibility Locks**

- Set global random seeds (`numpy.random.seed`, `torch.manual_seed`, `random.seed`) to a single shared constant at the top of the notebook
- Document the Kaggle dataset snapshot date in a metadata cell so results are reproducible against a fixed data universe

---

### Phase 1: Multi-Market Data Ingestion & Unified Preprocessing

**Step 1.1 — Constituent Universe Construction**

For each of the three markets, replicate the original scraping approach with one structural upgrade: save a **versioned snapshot** of the constituent list with a timestamp. This prevents survivorship bias from index rebalancing contaminating the historical backtest — if a stock was added to Nifty 500 in 2022, it should only appear in the universe from 2022 onwards, not backfilled to 2014.

Markets and sources:
- **Nifty 500**: Wikipedia HTML table (original approach) + NSE official PDF for historical constituent changes
- **S&P 500**: Wikipedia HTML table (original approach) + CRSP constituent history (if available)
- **Nikkei 225**: topforeignstocks.com (original approach) + TSE official constituent change log

**Step 1.2 — OHLCV Bulk Download**

Extend the original closing-price-only approach to full OHLCV (Open, High, Low, Close, Volume) — all five fields are needed for the technical indicator feature engineering in Phase 2.

Download parameters:
- Date range: 2014-01-01 to 2024-12-31 (reserve 2024 as final out-of-sample test year)
- Frequency: Daily
- Currency alignment: Convert Nikkei 225 prices to a common currency (USD) using JPY/USD FX rate from yfinance (`JPYUSD=X`) for cross-market comparability — or keep in local currency and normalize returns (preferred, avoids FX modelling complexity)

**Step 1.3 — Data Quality Pipeline (Upgraded from Original)**

Carry over original rules (consecutive NaN removal, |log return| > 0.8 filter) and add:

- **Survivorship bias handling**: Tag delisted stocks; do not remove them from historical periods where they were active
- **Corporate action adjustment**: Verify yfinance returns adjusted close (it does by default); flag any split/dividend artifacts via Z-score spike detection
- **Cross-market date alignment**: Create a unified calendar of trading dates per market; do not impute across market holidays — keep each market's native calendar for single-market models, align only for cross-market feature computation
- **Minimum history filter**: Require at least 756 trading days (~3 years) of clean data before a stock is eligible for the model universe — prevents noise from newly listed companies

**Step 1.4 — Return Computation & Normalization**

Compute three representations of returns, each serving a different downstream purpose:

- **Log returns** `r_t = ln(P_t / P_{t-1})`: primary signal for all statistical models (carry over from original)
- **Excess returns** `e_t = r_t - r_f_daily`: input to CAPM and BL models
- **Cross-sectional z-scored returns** `z_t = (r_t - μ_cross) / σ_cross`: normalized rank-based return for ML features, prevents scale dominance by high-volatility markets

Store all three representations as separate DataFrames with consistent datetime indexing.

---

### Phase 2: Feature Engineering

This is the primary differentiator from the original project. Features are organized into four families, each providing a distinct signal type for the AI return predictor.

**Step 2.1 — Technical Indicator Features (Per Asset)**

Compute all indicators using a strict **point-in-time** rule: any indicator computed on day `t` uses only data available through close of day `t`. No look-ahead.

Momentum features:
- Returns over 5, 10, 21, 63, 126, 252 trading day windows
- Rate of Change (ROC) at 1-week, 1-month, 3-month
- Relative Strength Index (RSI, 14-day)
- MACD signal line and histogram (12/26/9 EMA)

Volatility features:
- Rolling realized volatility at 21 and 63-day windows
- Parkinson volatility estimator (uses High/Low, more efficient than close-to-close)
- GARCH(1,1) conditional volatility forecast (carry over arch from original — run per-asset)
- Average True Range (ATR, 14-day)

Volume features:
- 21-day volume z-score (detects unusual activity)
- On-Balance Volume (OBV) momentum
- Volume-weighted return (distinguishes conviction moves)

Price structure features:
- Distance from 52-week high (momentum/mean-reversion signal)
- Bollinger Band position (price relative to 2σ bands)
- 50-day / 200-day SMA crossover signal (binary)

**Step 2.2 — Risk & Factor Features (Per Asset)**

These carry over from the original risk profiling stage but are computed on rolling windows for time-varying estimates rather than the static full-sample values used in the original:

- Rolling 63-day Sharpe Ratio
- Rolling 63-day Sortino Ratio
- Rolling 63-day CAPM Beta (vs. equal-weighted market proxy)
- Rolling 63-day CAPM Alpha
- Rolling 63-day CVaR at 95% confidence
- Rolling 21-day maximum drawdown

**Step 2.3 — Cross-Sectional Rank Features**

For each trading day, rank each stock within its market universe and sector on the key metrics (return, Sharpe, Sortino, volatility, Beta). Normalize to [0, 1]. These rank features are more stable than raw metric values and help the model learn relative attractiveness rather than absolute levels.

**Step 2.4 — Market Regime Features (Cross-Asset)**

Carry over the HMM regime detector from the original system, but run it in a rolling fashion:
- Fit GaussianHMM on a rolling 252-day window of equal-weighted portfolio returns
- Regime state on day `t` is estimated using only data through day `t-1` (one-day lag to prevent look-ahead)
- Output: one-hot encoded regime state vector `[Bullish, Bearish, Neutral]` as features
- Add regime transition probability matrix diagonal elements (persistence) as continuous features

Cross-market correlation regime features:
- Rolling 63-day correlation between Nifty 500, S&P 500, and Nikkei 225 equal-weighted returns
- VIX level (downloadable from yfinance as `^VIX`) as a global risk sentiment feature
- USDINR and USDJPY FX returns as cross-market linkage features

**Step 2.5 — Feature Matrix Assembly & Selection**

- **Target variable**: Forward 21-day excess log return (the value the model is trained to predict as a BL view)
- **Feature lag**: All features are lagged by 1 day relative to the target to ensure strict point-in-time compliance
- **Dimensionality control**: Apply feature importance screening using a preliminary LightGBM model; retain the top 60 features by gain importance to prevent the curse of dimensionality in the LSTM
- **Normalization**: Apply `RobustScaler` (median/IQR-based, robust to outliers) per feature across the training window — fit the scaler only on training data, apply to validation/test

---

### Phase 3: AI Model Training — Return Prediction (BL View Generator)

The goal of Phase 3 is to produce **forward-looking expected excess return predictions** for each stock, which replace the static Sortino-ranked views in the original BL implementation. Three models are trained in sequence: a gradient boosting baseline, a sequential deep learning model, and a state-of-the-art multivariate attention model.

**Step 3.1 — Walk-Forward Cross-Validation Design**

This is the most critical methodological safeguard. All models are trained and evaluated using **TimeSeriesSplit** (expanding window):

```
Training design (avoiding look-ahead bias):

Fold 1:  Train: 2014-2018 | Validate: 2019      | Test: [held out]
Fold 2:  Train: 2014-2019 | Validate: 2020      | Test: [held out]
Fold 3:  Train: 2014-2020 | Validate: 2021      | Test: [held out]
Fold 4:  Train: 2014-2021 | Validate: 2022      | Test: [held out]
Fold 5:  Train: 2014-2022 | Validate: 2023      | Test: [held out]
Final:   Train: 2014-2023 | Test:     2024       | [report results]
```

Each fold's scaler, model, and hyperparameters are fit independently on training data. Test year 2024 is locked away and touched only once at the very end.

**Step 3.2 — Model A: XGBoost Gradient Boosting (Baseline)**

Architecture purpose: Fast, interpretable baseline that establishes a minimum performance floor and provides feature importance rankings.

Key design choices:
- Input: Flat feature vector (all 2.5 features at time `t` for stock `i`)
- Target: 21-day forward excess return (regression)
- Loss function: Huber loss (robust to return outlier days)
- Regularization: L1 + L2 (`alpha` + `lambda` parameters)
- Hyperparameter search: Optuna with 100 trials, optimizing validation Information Coefficient (IC) — the rank correlation between predicted and realized returns. IC is the standard evaluation metric in quant research because it measures directional accuracy across the return distribution, not absolute prediction error.
- Ensemble: Train 5 models (one per cross-validation fold) and average predictions (ensemble reduces variance)
- Output per stock per day: scalar predicted 21-day excess return

Evaluation metrics:
- Information Coefficient (Rank IC): target ≥ 0.05 (industry baseline for daily stock prediction)
- IC Information Ratio (ICIR): IC mean / IC std across time; target ≥ 0.5
- Hit Rate: fraction of days where predicted direction matches realized direction; target ≥ 54%

**Step 3.3 — Model B: LSTM Sequential Model**

Architecture purpose: Capture temporal dependencies in return patterns that the XGBoost flat-feature approach cannot model.

Architecture design:
- Input shape: `(sequence_length=63, n_features=60)` — 63 trading days of features per stock
- Layer stack:
  - LSTM Layer 1: 128 hidden units, return sequences = True, dropout = 0.2
  - LSTM Layer 2: 64 hidden units, return sequences = False, dropout = 0.2
  - BatchNorm Layer
  - Dense Layer: 32 units, ReLU activation
  - Dropout: 0.3
  - Output Layer: 1 unit, linear activation (regression)
- Loss: Huber loss
- Optimizer: AdamW with cosine annealing learning rate schedule
- Batch construction: Sample random (stock, time) pairs from training universe — each sample is a 63-day window ending at day `t` with target being day `t+21` excess return
- Early stopping: Monitor validation IC, patience = 10 epochs

**Step 3.4 — Model C: iTransformer (Primary Model)**

Architecture purpose: The iTransformer (Liu et al., 2024) is a state-of-the-art multivariate time series forecaster that inverts the conventional Transformer architecture — it applies attention across **variates (stocks)** rather than across time steps. This is theoretically ideal for portfolio prediction because cross-stock attention captures the interdependencies between assets that drive covariance structure.

Architecture design:
- Input: Multivariate panel `(n_stocks, sequence_length=63, n_features)` — models all stocks jointly
- Key innovation: Token = one stock's full time series, so attention weights encode cross-stock relationships
- Positional encoding: Learnable, applied along the variate dimension
- Attention heads: 8
- Transformer blocks: 3
- Feed-forward dimension: 256
- Dropout: 0.1
- Output: Per-stock 21-day forward excess return predictions
- Cross-market variant: Train one iTransformer jointly on all three markets with market-ID embeddings — this allows the model to learn cross-market signal transmission (e.g., S&P 500 moves predicting Nikkei 225 next-day returns)

**Step 3.5 — Model Ensembling & View Generation**

Combine predictions from all three models using a **stacking approach**:
- Level-0 predictions: XGBoost, LSTM, iTransformer predicted 21-day excess returns per stock
- Level-1 meta-learner: Ridge regression trained on validation fold IC-weighted average of Level-0 predictions
- Output: Final ensemble predicted excess return vector `Q_AI` per stock per rebalancing date

Uncertainty quantification:
- Compute prediction standard deviation across the three models for each stock
- Use this dispersion as the diagonal of the BL view uncertainty matrix `Ω` — stocks where models agree get lower `Ω` (stronger view), stocks where models disagree get higher `Ω` (weaker view). This is a rigorous improvement over the original τ-scaled diagonal approach.

---

### Phase 4: AI-Augmented Portfolio Construction

**Step 4.1 — Ledoit-Wolf Shrinkage Covariance**

Replace the raw sample covariance matrix from the original system:

- Use `sklearn.covariance.LedowtWolf` (Oracle Approximating Shrinkage variant)
- Alternatively, use the Constant Correlation shrinkage target (Ledoit-Wolf 2004), which is empirically superior for equity return matrices
- Annualize: `Σ_LW = LedoitWolf_daily × 252`
- Validate: Confirm condition number of `Σ_LW` is ≪ condition number of raw `Σ̂` — a well-conditioned covariance matrix is the primary output check

**Step 4.2 — AI-Augmented Black-Litterman**

Upgrade the original BL implementation with three changes:

1. **AI-generated views**: Replace Sortino-ranked static views with the ensemble model predictions `Q_AI` — the full vector of predicted 21-day excess returns for all stocks in the universe becomes the BL view vector
2. **Model-uncertainty view matrix Ω**: Diagonal elements of `Ω` are set to the cross-model prediction dispersion computed in Step 3.5, not the fixed `τ·PΣP^T` scalar
3. **Full P matrix**: Because AI predictions are available for all stocks simultaneously, construct `P` as the identity matrix (each stock has its own absolute return view) rather than the sparse relative views in the original

The equilibrium prior `π = λ · Σ_LW · w_mkt` is retained from the original, as is the closed-form posterior formula.

**Step 4.3 — MPT Optimization with Shrinkage Covariance**

Run the original SLSQP Maximum Sharpe optimization but feed it `Σ_LW` instead of the raw covariance matrix. This single change is expected to substantially stabilize the MPT weights. Retain the original Monte Carlo Efficient Frontier visualization (10,000 random portfolios) for interpretability.

**Step 4.4 — Hybrid Portfolio Blending**

Retain the original convex blending and SLSQP projection approach:

```
w_hybrid = BLEND_ALPHA × w_MPT + (1 - BLEND_ALPHA) × w_BL_AI

Feasibility projection:
  min  ½ ||w - w_hybrid||²
  s.t. Σ wᵢ = 1
       0 ≤ wᵢ ≤ MAX_WEIGHT_PER_ASSET
       Σ_{sector s} wᵢ ≤ MAX_SECTOR_WEIGHT
```

Add a **regime-conditional BLEND_ALPHA** — in Bullish regimes (HMM state 0), tilt toward MPT weights (`α = 0.6`); in Bearish regimes, tilt toward BL AI weights (`α = 0.4`) since the equilibrium prior is more defensive. This regime-switching blend is a natural extension of the existing HMM component.

---

### Phase 5: Walk-Forward Backtesting Engine

This is the most critical addition absent from the original project.

**Step 5.1 — Backtesting Design Principles**

Strict rules enforced throughout:
- **No look-ahead bias**: Every portfolio constructed on day `t` uses only information available at close of day `t-1`
- **Realistic transaction costs**: Apply 10 bps (0.1%) one-way cost for Nifty 500, 5 bps for S&P 500, 15 bps for Nikkei 225 (reflecting typical retail/institutional bid-ask spreads)
- **Rebalancing frequency**: Monthly (first trading day of each month), matching the 21-day return prediction horizon
- **Minimum weight filter**: Any weight below 0.5% is set to zero and the remainder renormalized — avoids unexecutable micro-positions

**Step 5.2 — Walk-Forward Execution Loop**

For each rebalancing date in the backtest period (2019-2024):

1. Fetch the live training data window (expanding from 2014)
2. Re-fit the scaler on training data only
3. Generate AI model predictions for the current universe
4. Compute `Σ_LW` on training data
5. Run BL-AI + MPT + Hybrid construction
6. Determine regime state from HMM (using data through `t-1`)
7. Set regime-conditional BLEND_ALPHA
8. Apply transaction cost deduction based on turnover from previous portfolio
9. Record daily portfolio returns using next-month realized returns
10. Log all intermediate values (weights, predicted returns, realized returns, regime) for attribution analysis

**Step 5.3 — Backtest Output Data Structure**

For each rebalancing date, record:
- Portfolio weights vector (all stocks)
- Expected portfolio return (ex-ante, from model)
- Realized portfolio return (ex-post, next month)
- Turnover (L1 norm of weight change vs. previous period)
- Regime state at time of construction
- BL/MPT/Hybrid split weights (for attribution)

---

### Phase 6: Benchmark Comparison & Performance Attribution

**Step 6.1 — Benchmark Portfolios**

Compute the following benchmark return series over the same backtest period:

- **Market Index**: Equal-weighted index of the full constituent universe (replicates holding everything)
- **Pure MPT**: Max-Sharpe portfolio with shrinkage covariance, no BL adjustment, no AI
- **Pure BL (Original)**: BL with Sortino-ranked views (the original project's approach)
- **BL-AI (Ablation)**: BL with AI views but without hybrid blending
- **Hybrid AI (Full System)**: Complete upgraded system — the primary result

**Step 6.2 — Performance Metrics**

Compute for each strategy over the full backtest and sub-periods (pre-COVID, COVID, post-COVID, 2022 rate hike cycle, 2023-2024):

Primary metrics:
- **Annualized Return**: `(1 + r̄_daily)^252 - 1`
- **Annualized Volatility**: `σ_daily × √252`
- **Sharpe Ratio**: `(return - r_f) / volatility` — target: outperform pure MPT and pure BL by ≥10% Sharpe improvement
- **Maximum Drawdown**: `max{[max(V_s) - V_t] / max(V_s) : s ≤ t}` over the backtest period
- **Calmar Ratio**: `annualized_return / |max_drawdown|`
- **Sortino Ratio**: Annualized return / downside volatility

Secondary metrics:
- **Turnover**: Average monthly L1 norm of weight changes (lower is better for costs)
- **Information Ratio vs. Index**: `(return_strategy - return_index) / tracking_error`
- **Hit Rate**: Fraction of months where strategy beats equal-weighted index
- **Average IC over backtest period**: Cross-sectional rank correlation between predicted and realized returns

**Step 6.3 — Attribution Analysis**

Decompose realized portfolio performance into:
- **Selection effect**: Returns attributable to stock picking (weight × stock return vs. benchmark)
- **Allocation effect**: Returns attributable to sector tilts
- **Regime effect**: Performance differential in Bullish vs. Bearish vs. Neutral HMM states (validates the regime-conditional blending)
- **AI view quality**: Correlation between `Q_AI` predictions and realized next-month returns (IC over time)

---

### Phase 7: Model Serialization & Output Artifacts

**Step 7.1 — Saved Models**

Serialize and save to Kaggle output directory:
- `xgb_model_final.json` — XGBoost model trained on full 2014-2023 data
- `lstm_model_final.pt` — PyTorch LSTM state dict
- `itransformer_final.pt` — PyTorch iTransformer state dict
- `feature_scaler.joblib` — Fitted RobustScaler for inference
- `hmm_model.pkl` — Fitted GaussianHMM (3-state)
- `ledoitwolf_params.json` — Shrinkage coefficient δ for each market
- `backtest_weights.csv` — Full historical weight matrix (all rebalancing dates × all stocks)
- `backtest_returns.csv` — Daily portfolio returns for all strategies
- `performance_summary.csv` — Metrics table (all strategies × all metrics)

**Step 7.2 — Visualization Outputs**

Generate and save as high-DPI PNG and interactive Plotly HTML:
- Cumulative return curves (all 5 strategies + benchmark)
- Rolling 63-day Sharpe ratio comparison
- Efficient Frontier with MPT and BL-AI optimal portfolios marked
- Feature importance chart (XGBoost top 20 features)
- Regime timeline (HMM state sequence over backtest period coloured by Bull/Bear/Neutral)
- Weight heatmap (sector allocations over time)
- IC decay chart (shows how predictive power decays from t+1 to t+63)
- Maximum Drawdown underwater chart

---

## Section 3: Streamlit Dashboard — Structural Process

### Overall Architecture

The dashboard is a 4-panel, single-page Streamlit application organized as:

```
app.py  (main entry, routing, layout)
├── components/
│   ├── sidebar.py          (all user inputs)
│   ├── data_loader.py      (cached data fetching)
│   ├── model_inference.py  (load models + generate portfolio)
│   ├── portfolio_engine.py (BL + MPT + Hybrid construction)
│   ├── backtest_runner.py  (simulate forward/historical performance)
│   └── charts.py           (all Plotly figure factories)
├── models/                 (saved model artifacts from Kaggle)
├── data/                   (cached market data)
├── requirements.txt
└── .streamlit/config.toml  (theme + layout settings)
```

### Panel 1: Sidebar — User Input Controls

Control group 1 — Market Selection:
- Market dropdown: `Nifty 500 (India)` / `S&P 500 (USA)` / `Nikkei 225 (Japan)` / `All Markets (Cross-Market)`
- Universe size slider: 10 to 100 stocks (controls how many stocks from the AI-ranked list to include)
- Sector filter: Multi-select to include/exclude sectors

Control group 2 — Portfolio Parameters:
- Risk tolerance: `Conservative (Sharpe priority)` / `Balanced` / `Aggressive (Return priority)`
  - Maps to MAX_WEIGHT_PER_ASSET: 0.15 / 0.25 / 0.40
  - Maps to target volatility constraint: 10% / 15% / 25% annualized
- Time horizon: `1 Month` / `3 Months` / `6 Months` / `1 Year`
- Rebalancing frequency: `Monthly` / `Quarterly`

Control group 3 — Model Configuration:
- BL Blend Alpha: slider 0.0 → 1.0 (0.5 default)
- AI Model: `XGBoost` / `LSTM` / `iTransformer` / `Ensemble`
- Show regime overlay: toggle

Control group 4 — Action:
- `Generate Portfolio` button (primary CTA)
- `Compare with Benchmark` toggle
- `Download Results` button (exports CSV)

### Panel 2: Portfolio Summary — Top Row

Tab A — Allocation View:
- Pie chart: Portfolio weights by stock (Plotly)
- Stacked bar: Sector allocation breakdown
- Data table: Stock | Sector | Weight | Expected Return | Sharpe | Beta | Sortino — sortable, filterable

Tab B — Risk Dashboard:
- Gauge chart: Portfolio-level expected Sharpe ratio
- Metric cards: Expected return / volatility / max drawdown / CVaR
- Correlation heatmap: Selected stocks only (seaborn-style in Plotly)

### Panel 3: Efficient Frontier — Middle Row

- Interactive Plotly scatter of 5,000 simulated random portfolios coloured by Sharpe ratio
- Optimal MPT portfolio marked (red star)
- BL-AI portfolio marked (blue diamond)
- Hybrid portfolio marked (gold circle — the recommended portfolio)
- Current risk tolerance boundary shown as vertical line
- Hover tooltip on each marker: shows return / risk / Sharpe / top-3 holdings

### Panel 4: Performance Simulation — Bottom Row

Tab A — Historical Backtest (if date range overlaps 2019-2024):
- Cumulative return line chart: Strategy vs. Market Index vs. Pure MPT vs. Pure BL (all on same axis)
- Rolling 63-day Sharpe ratio chart
- Drawdown underwater chart
- Performance metrics table (Sharpe / Max DD / Calmar / Sortino)

Tab B — Forward Simulation:
- Monte Carlo simulation of forward portfolio paths (500 scenarios)
  - Uses historical return and covariance structure + AI expected returns
  - Shows 5th / 25th / 50th / 75th / 95th percentile fan chart
- Expected portfolio value evolution (starting from $10,000 notional)
- Probability of loss over the selected time horizon

Tab C — Regime Analysis:
- Current regime state (Bullish / Bearish / Neutral) displayed as colored badge
- Historical regime timeline over last 252 days
- Regime-conditional performance statistics

### Caching & Performance Strategy

- `@st.cache_data` with TTL = 3600 seconds on all data fetching functions
- `@st.cache_resource` (no TTL) on model loading — models are loaded once per session
- Efficient frontier simulation runs in a background thread to avoid blocking the UI
- Lazy loading: Panels 3 and 4 only render after the user clicks `Generate Portfolio`

### Error Handling Design

- Data fetch failure: Show cached historical data with a warning banner ("Using cached data from [date]")
- Model inference failure: Fall back to the rule-based Sortino-ranked BL approach from the original project with a user notification
- Insufficient universe: If fewer than 10 stocks pass quality filters, alert user and suggest relaxing sector filter
- Invalid parameter combinations: Inline validation before generating portfolio (e.g., universe size > available stocks)

---

## Section 4: Deployment & Scalability

### Streamlit Cloud Deployment Process

Step-by-step:

1. Create a public GitHub repository with the structure defined in Section 3
2. Place the serialized model artifacts from Kaggle into `models/` (or use Streamlit secrets + cloud storage link for large files)
3. Create `requirements.txt` with pinned versions of all dependencies (use `pip freeze` from the local VS Code virtual environment)
4. Create `.streamlit/config.toml` with theme settings (dark/light, primary color, font)
5. Push to GitHub; connect to Streamlit Cloud via streamlit.io → `New App` → select repo
6. Configure Secrets in Streamlit Cloud dashboard (API keys for yfinance premium / Alpha Vantage if used)
7. Set environment variables for model artifact paths
8. Trigger deploy; Streamlit Cloud builds the Docker container automatically

For large model files (iTransformer weights ~200MB):
- Upload to Hugging Face Hub or Google Cloud Storage
- Use presigned URL in `data_loader.py` with download-and-cache on first run

### Docker Containerization (Local + Production)

Dockerfile structure:
- Base image: `python:3.11-slim`
- System dependencies: `build-essential`, `libgomp1` (XGBoost requirement)
- Copy requirements and install in a single layer
- Copy application code
- Expose port 8501 (Streamlit default)
- Entrypoint: `streamlit run app.py --server.headless true`

docker-compose.yml addition:
- Mount a `./data` volume for persistent caching of downloaded price data
- Add a `redis` service for production-grade caching (replacing `@st.cache_data` for multi-user deployments)

### API Key Management

- Never hardcode API keys in source code
- Use Streamlit secrets (`st.secrets`) for cloud deployment
- Use `.env` + `python-dotenv` for local VS Code development
- Create a `secrets.toml.example` template in the repo (no actual keys)

---

## Risks & Mitigations

### Risk 1: Overfitting & In-Sample Bias

**Risk**: The AI model learns spurious patterns specific to the 2014-2023 training period, producing inflated backtest Sharpe ratios that collapse out-of-sample.

**Mitigations**:
- Strict TimeSeriesSplit walk-forward design (Phase 5) — no training data leaks into the test period
- Feature importance analysis to detect suspiciously high-importance features (data snooping signals)
- IC decay analysis: a genuinely predictive model shows IC > 0 at 1-day, 5-day, and 21-day horizons; a spurious one shows very high 1-day IC but near-zero at 21 days
- Regime-stratified evaluation: performance must hold across Bullish, Bearish, and Neutral regimes, not just aggregate
- Penalty for excessive turnover in the optimization objective — overfitted models tend to produce high-turnover portfolios

### Risk 2: Data Drift & Non-Stationarity

**Risk**: Market structure changes (COVID regime, 2022 rate shock, AI sector bubble) cause the model's learned patterns to become stale.

**Mitigations**:
- Expanding window retraining (the walk-forward design inherently adapts)
- Add a **model staleness detector**: if rolling 63-day IC falls below 0.02 for 3 consecutive months, trigger automatic retraining
- Feature engineering using log returns and z-scores (vs. raw prices) improves stationarity
- HMM regime detection provides a soft switch that adjusts blend weights in response to structural breaks

### Risk 3: Survivorship Bias

**Risk**: Using only current Nifty 500 / S&P 500 / Nikkei 225 constituents and backtesting from 2014 means the historical universe contains only survivors — companies that were not delisted, bankrupt, or removed. This artificially inflates backtest returns.

**Mitigations**:
- Timestamp constituent list snapshots (Step 1.1) and only include stocks in the universe for periods when they were actually index members
- If historical constituent data is unavailable, clearly disclose survivorship bias limitation in the notebook documentation
- Apply a conservative 50 bps additional performance haircut to all reported Sharpe ratios to account for estimated survivorship bias

### Risk 4: Regulatory & Ethical Considerations

**Disclosure requirements**:
- All outputs are for research and educational purposes only — prominently display this in the dashboard
- Predicted returns are not investment advice; include a regulatory disclaimer on the Streamlit app home screen
- Cross-border deployment of the dashboard may require financial services licensing in some jurisdictions

**Data ethics**:
- All data sources used (yfinance, Wikipedia) are publicly available; no private or non-public information is used
- The model does not use alternative data (social media sentiment, satellite data) that could raise fairness concerns in regulated contexts
- Backtest results must clearly disclose: in-sample period, out-of-sample period, transaction cost assumptions, and survivorship bias status

### Risk 5: Computational Budget on Kaggle

**Risk**: iTransformer training on 500 stocks × 2500 days × 60 features may exceed Kaggle's 9-hour GPU session limit.

**Mitigations**:
- Pre-compute and cache feature matrices before model training (save as compressed `.npz`)
- Use mixed-precision training (FP16) for the iTransformer — reduces memory by ~40% and speeds training by ~2×
- If time budget is tight, train LSTM on Kaggle and load pre-trained XGBoost; iTransformer can be trained in parallel on Colab Pro
- Use gradient checkpointing in the iTransformer to reduce VRAM requirements when working with the full multi-market panel

---

## Glossary of Key Terms

| Term | Definition |
|---|---|
| IC (Information Coefficient) | Rank correlation between predicted and realized returns; primary ML evaluation metric in quant finance |
| ICIR | IC Information Ratio = mean(IC) / std(IC); measures consistency of predictive signal |
| Ledoit-Wolf Shrinkage | Regularized covariance estimator that shrinks the sample matrix toward a structured target |
| iTransformer | Inverted-variate Transformer (Liu et al., 2024): applies attention across stocks, not time steps |
| Walk-Forward Backtesting | Rolling train/test split that prevents test data from contaminating training |
| Survivorship Bias | Overestimation of returns caused by excluding failed/delisted stocks from historical data |
| Calmar Ratio | Annualized return divided by maximum drawdown; penalizes strategies with large tail losses |
| HMM | Hidden Markov Model; probabilistic sequence model used here for market regime detection |
| SLSQP | Sequential Least-Squares Quadratic Programming; gradient-based constrained optimizer from scipy |
| BL | Black-Litterman model; Bayesian framework for blending CAPM equilibrium returns with investor views |

---

*Pipeline Version: 1.0 | Research Status: Design Complete, Implementation Pending*
*Original Project Credits: Quantitative analysis notebooks for Nifty 500, S&P 500, Nikkei 225 (2014–2025)*
