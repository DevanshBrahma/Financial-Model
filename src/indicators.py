import pandas as pd


def add_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    loss = loss.replace(0, pd.NA)
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df.loc[loss.isna(), "RSI"] = 100
    return df


def add_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    return df


def add_moving_averages(df, windows=(20, 50)):
    for w in windows:
        df[f"SMA_{w}"] = df["Close"].rolling(window=w).mean()
    return df


def add_indicators(df):
    df = add_rsi(df)
    df = add_macd(df)
    df = add_moving_averages(df)
    df = df.dropna()
    return df
