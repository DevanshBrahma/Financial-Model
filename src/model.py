import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential


def create_sequences(data, seq_length=60):
    """Create sliding-window sequences for time-series prediction.

    Args:
        data: 2D numpy array of shape (n_samples, 1) with scaled close prices.
        seq_length: Number of previous timesteps included in each input sample.

    Returns:
        Tuple `(X, y)` where `X` has shape (n_sequences, seq_length, 1) before
        final reshaping and `y` has shape (n_sequences, 1).
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length : i])
        y.append(data[i])
    return np.array(X), np.array(y)


def train_lstm_and_predict(df, seq_length=60, epochs=5, batch_size=32):
    """Train an LSTM on close prices and append inverse-scaled predictions.

    Args:
        df: Price dataframe containing a `Close` column.
        seq_length: Number of timesteps per training sequence.
        epochs: Number of training epochs.
        batch_size: Training batch size.

    Returns:
        A tuple `(df, preds)` where `df` is trimmed to prediction rows and includes
        `Predicted_Close`, and `preds` is the inverse-scaled predictions array.
    """
    close_prices = df[["Close"]].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    X, y = create_sequences(scaled, seq_length)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential(
        [
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            LSTM(50),
            Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    preds = model.predict(X, verbose=0)
    preds = scaler.inverse_transform(preds)

    df = df.iloc[seq_length:].copy()
    df["Predicted_Close"] = preds

    return df, preds
