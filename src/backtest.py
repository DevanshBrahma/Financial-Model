import numpy as np


def backtest_strategy(df):
    # Basic strategy: if predicted close > actual close -> buy, else sell
    df["Signal"] = np.where(df["Predicted_Close"] > df["Close"], 1, -1)
    df["Returns"] = df["Close"].pct_change()
    df["Strategy_Returns"] = df["Signal"].shift(1) * df["Returns"]

    df["Equity_Curve"] = (1 + df["Strategy_Returns"].fillna(0)).cumprod()
    return df["Equity_Curve"]
