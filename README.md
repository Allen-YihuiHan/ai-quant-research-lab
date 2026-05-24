# AI Quant Research Lab

A rigorous, end-to-end quantitative research framework built for learning.
The goal is not to deploy a live trading bot — it is to build the complete
research loop: data → factors → evaluation → backtest → risk analysis → ML alpha.

## Project structure

```
ai-quant-research-lab/
├── configs/            YAML config files and universe definitions
├── data/               Raw, processed, and cached market data (git-ignored)
├── notebooks/          Jupyter notebooks — one per module
├── reports/            Generated figures and factor reports
├── src/quantlab/       Core library (importable Python package)
│   ├── data/           Data download and caching
│   ├── features/       Returns, technical indicators, factor construction
│   ├── evaluation/     Performance metrics, IC/RankIC, factor evaluation
│   ├── backtest/       Backtest engine and portfolio analytics
│   ├── models/         ML alpha models (Ridge → LightGBM → TFT)
│   └── utils/          Config loader, logger
└── tests/              pytest unit tests
```

## Module roadmap

| Module | Topic | Status |
|--------|-------|--------|
| 0 | Environment & project skeleton | ✅ done |
| 1 | Data download + returns + performance metrics | ✅ done |
| 2 | Technical factors + no-lookahead checks | 🔲 |
| 3 | Backtest engine (daily rebalancing, TopK) | 🔲 |
| 4 | Factor evaluation: IC, RankIC, IC Decay, group returns | 🔲 |
| 5 | Multi-factor strategy | 🔲 |
| 6 | ML Alpha: Ridge → LightGBM → (TFT) | 🔲 |
| 7 | Qlib integration & Alpha158 comparison | 🔲 |
| 8 | Automated research report | 🔲 |

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install the package in editable mode
pip install -e ".[dev]"

# 3. Run tests
pytest

# 4. Launch the first notebook
jupyter notebook notebooks/01_data_loader.ipynb
```

## Stock universes

| Key | Stocks | Use |
|-----|--------|-----|
| `us_phase1` | 8 (AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, SPY) | Modules 1–3, smoke tests |
| `us_sp100` | ~96 S&P 100 stocks | Modules 4+ (cross-sectional IC needs 30+ stocks) |

Switch universe by editing `configs/default.yaml`:

```yaml
universe: us_sp100   # change here
```

## Key design rules

1. **No lookahead.** Features may only use data up to and including day *t*.
   Labels (forward returns) may use day *t+1* onward — but never to construct
   the day-*t* signal.
2. **Signal delay.** Signals generated at close on day *t* are executed at
   open or close on day *t+1*.
3. **Time-based splits.** ML train/val/test is always chronological — never
   random `train_test_split`.
4. **Costs always on.** Every backtest includes transaction costs and turnover.
