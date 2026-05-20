import os

import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_predictions(df):
    _ensure_output_dir()
    plt.figure(figsize=(10, 5))
    plt.plot(df["Close"], label="Actual Close")
    plt.plot(df["Predicted_Close"], label="Predicted Close")
    plt.title("LSTM Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/price_predictions.png")
    plt.close()


def plot_equity_curve(equity_curve):
    _ensure_output_dir()
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve, label="Equity Curve")
    plt.title("Strategy Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/strategy_equity_curve.png")
    plt.close()


def plot_drawdown(equity_curve):
    _ensure_output_dir()
    drawdown = (equity_curve / equity_curve.cummax()) - 1
    plt.figure(figsize=(10, 5))
    plt.plot(drawdown, label="Drawdown")
    plt.title("Strategy Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/strategy_drawdown.png")
    plt.close()
