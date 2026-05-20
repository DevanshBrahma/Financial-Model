import argparse

from data import load_price_data
from indicators import add_indicators
from model import train_lstm_and_predict
from backtest import backtest_strategy
from metrics import compute_metrics
from plotting import plot_predictions, plot_equity_curve, plot_drawdown


def main():
    parser = argparse.ArgumentParser(description="ML Trading Model")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    args = parser.parse_args()

    # 1) Load data
    df = load_price_data(args.ticker, args.start, args.end)

    # 2) Add indicators
    df = add_indicators(df)

    # 3) Train LSTM + predict
    df, preds = train_lstm_and_predict(df)

    # 4) Backtest strategy
    equity_curve = backtest_strategy(df)

    # 5) Metrics
    m = compute_metrics(equity_curve)
    print("\nPerformance Metrics")
    for k, v in m.items():
        print(f"{k}: {v:.4f}")

    # 6) Plots
    plot_predictions(df, preds)
    plot_equity_curve(equity_curve)
    plot_drawdown(equity_curve)


if __name__ == "__main__":
    main()
