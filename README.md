# Financial ML Trading Model

An end-to-end **finance + CS** project that combines:
- **LSTM price prediction**
- **Technical indicators** (RSI, MACD, Moving Averages)
- **Backtesting** with performance metrics (ROI, Sharpe, Max Drawdown)
- **Plots** for model and strategy performance

> Built as a simple Python backend project suitable for a 2nd‑year CS resume.

## Features
- Download historical stock data using `yfinance`
- Engineer technical indicators
- Train an LSTM model for next‑day price prediction
- Generate trading signals
- Backtest strategy on historical data
- Compute performance metrics
- Save plots to `/outputs`

## Quick Start

```bash
# 1) Create virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the pipeline
python src/main.py --ticker AAPL --start 2018-01-01 --end 2024-12-31
```

## Outputs
- `outputs/price_predictions.png`
- `outputs/strategy_equity_curve.png`
- `outputs/strategy_drawdown.png`

## Resume Line (example)
> Built an ML-driven trading system with LSTM price prediction, technical indicators (RSI/MACD), and backtesting metrics (ROI, Sharpe, max drawdown), implemented in Python with data visualization.

---

**Disclaimer:** This project is for educational purposes only and not financial advice.
