import yfinance as yf


def load_price_data(ticker: str, start: str, end: str):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.dropna()
    return df
