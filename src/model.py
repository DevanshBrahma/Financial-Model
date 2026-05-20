import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential


def create_sequences(data: np.ndarray, seq_length: int = 60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length : i])
        y.append(data[i])
    return np.array(X), np.array(y)


def train_lstm_and_predict(df, seq_length: int = 60, epochs: int = 5, batch_size: int = 32):
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

    out = df.iloc[seq_length:].copy()
    out["Predicted_Close"] = preds

    return out, preds
