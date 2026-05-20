import numpy as np

LONG = 1
SHORT = -1


def backtest_strategy(df):
    # Basic strategy: LONG when predicted close > actual close, otherwise SHORT.
    df["Signal"] = np.where(df["Predicted_Close"] > df["Close"], LONG, SHORT)
    df["Returns"] = df["Close"].pct_change()
    df["Strategy_Returns"] = df["Signal"].shift(1) * df["Returns"]

    df["Equity_Curve"] = (1 + df["Strategy_Returns"].fillna(0)).cumprod()
    return df["Equity_Curve"]
