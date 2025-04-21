import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import pytz
def fetch_bitcoin_data():
    print("Fetching latest data (7d @ 5m)...")
    df = yf.download('BTC-USD', period='7d', interval='5m')
    if df.empty:
        raise RuntimeError("No data fetched. Check connection.")
    df.reset_index(inplace=True)
    df.to_csv('bitcoin_data.csv', index=False)
    return df
def prepare_data(sequence_length=60):
    df = pd.read_csv('bitcoin_data.csv', parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Close'], inplace=True)

    prices = df[['Close']].values
    scaler = MinMaxScaler((0, 1))
    scaled = scaler.fit_transform(prices)

    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i, 0])
        y.append(scaled[i, 0])

    if not X:
        raise RuntimeError(f"Need ≥{sequence_length+1} rows, got {len(scaled)}")

    X = np.array(X).reshape(-1, sequence_length, 1)
    y = np.array(y)
    return X, y, scaler, df.index
def build_model(input_shape):
    m = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])
    m.compile(optimizer=Adam(1e-3), loss='mean_squared_error')
    return m
def main():
    fetch_bitcoin_data()
    X, y, scaler, timestamps = prepare_data(sequence_length=60)
    try:
        model = load_model('model.h5')
        print("Loaded existing model.h5")
    except OSError:
        print("Training new model…")
        model = build_model((X.shape[1], 1))
        model.fit(X, y, epochs=20, batch_size=32, verbose=1)
        model.save('model.h5')
        print("Model saved as model.h5")
    last_seq = X[-1].reshape(1, X.shape[1], 1)
    scaled_pred = model.predict(last_seq, verbose=0)
    price_pred = scaler.inverse_transform(scaled_pred)[0][0]
    last_ts = timestamps[-1]
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize('UTC')
    local_tz = pytz.timezone('America/New_York')
    local_time = last_ts.astimezone(local_tz)
    next_time = local_time + pd.Timedelta(minutes=5)

    print(f"Predicted BTC price at {next_time.strftime('%Y-%m-%d %H:%M:%S %Z')}: "
          f"${price_pred:.2f}")

if __name__ == "__main__":
    main()
