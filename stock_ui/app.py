# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import yfinance as yf
from datetime import date, timedelta

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Predictor (TensorFlow model)")

MODEL_PATH = "models/stock_model.h5"
SCALER_PATH = "models/scaler.pkl"

@st.cache_resource
def load_model_and_scaler(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    # load TF model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    # load scaler (joblib)
    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler from {scaler_path}: {e}")
    return model, scaler

# Load
try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    st.error(str(e))
    st.stop()

# Try to infer sequence length from model input shape (if possible)
inferred_seq_len = None
try:
    input_shape = model.input_shape  # e.g. (None, 60, 1)
    if input_shape and len(input_shape) >= 2 and input_shape[1] is not None:
        inferred_seq_len = int(input_shape[1])
except Exception:
    inferred_seq_len = None

DEFAULT_SEQ_LEN = 60
st.sidebar.header("Model / Data settings")
st.sidebar.write(f"Loaded model: {MODEL_PATH}")
st.sidebar.write(f"Inferred seq_len: {inferred_seq_len if inferred_seq_len else 'unknown'}")
seq_len_ui = st.sidebar.number_input("Sequence length (seq_len)", min_value=1, value=inferred_seq_len or DEFAULT_SEQ_LEN, step=1)
lookback_days = st.sidebar.number_input("History lookback days (yfinance)", min_value=seq_len_ui, value=max(365, seq_len_ui*3))

st.sidebar.markdown("---")
st.sidebar.write("Scaler info:")
if hasattr(scaler, "min_"):
    st.sidebar.write(f"scaler type: {type(scaler)}")
    # show brief scaler info (do not print long arrays)
    try:
        st.sidebar.write(f"scale sample: {getattr(scaler, 'scale_', None)[:3]}")
    except Exception:
        pass

# Utility: fetch history
@st.cache_data
def fetch_history(symbol: str, lookback_days: int):
    end = date.today()
    start = end - timedelta(days=lookback_days)
    df = yf.download(symbol, start=start, end=end)
    return df

def predict_stock(model, scaler, hist_df, n_days, seq_len):
    """
    hist_df: dataframe with Date index and 'Close' column
    seq_len: number of past timesteps model expects (e.g., 60)
    """
    if hist_df is None or hist_df.empty:
        return None

    close = hist_df["Close"].values.reshape(-1, 1)

    # Scale using saved scaler. If your model was trained on scaled values,
    # model outputs should also be scaled; we inverse_transform at the end.
    scaled = scaler.transform(close)

    # If history shorter than seq_len, pad at beginning by repeating first value
    if scaled.shape[0] < seq_len:
        pad_amt = seq_len - scaled.shape[0]
        pad = np.repeat(scaled[0:1, :], pad_amt, axis=0)
        scaled = np.vstack((pad, scaled))

    last_seq = scaled[-seq_len:]  # shape (seq_len, 1)
    current_seq = last_seq.reshape(1, seq_len, 1)  # (1, seq_len, 1)

    preds_scaled = []
    for _ in range(n_days):
        p = model.predict(current_seq, verbose=0)
        # normalize different possible output shapes
        out = float(np.squeeze(p))
        preds_scaled.append([out])
        # append predicted scaled value and shift window
        current_seq = np.append(current_seq[:, 1:, :], np.array(out).reshape(1, 1, 1), axis=1)

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)

    # Inverse scale back to original price units.
    try:
        preds = scaler.inverse_transform(preds_scaled).flatten()
    except Exception:
        # If scaler cannot inverse_transform (rare), just return scaled predictions
        preds = preds_scaled.flatten()

    last_date = hist_df.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_days)
    pred_df = pd.DataFrame({"Date": future_dates, "Predicted": preds}).set_index("Date")
    return pred_df

# UI controls
col1, col2 = st.columns([2,1])
with col1:
    symbol = st.text_input("Stock symbol (ticker)", "AAPL").upper()
    n_days = st.slider("Days to predict", 1, 30, 7)
    predict_btn = st.button("Predict")

with col2:
    show_summary = st.checkbox("Show model summary")
    show_scaler = st.checkbox("Show scaler details")

if show_summary:
    s = []
    model.summary(print_fn=lambda x: s.append(x))
    st.text("\n".join(s))

if show_scaler:
    st.write(scaler)

if predict_btn:
    with st.spinner("Fetching history and running predictions..."):
        hist = fetch_history(symbol, int(lookback_days))
        if hist is None or hist.empty:
            st.error("No historical data found for that ticker. Check the symbol or your internet connection.")
        else:
            st.subheader(f"Historical Close — {symbol}")
            st.line_chart(hist["Close"])
            pred_df = predict_stock(model, scaler, hist, n_days, int(seq_len_ui))
            if pred_df is None:
                st.error("Prediction failed.")
            else:
                st.subheader("Predicted prices")
                st.line_chart(pred_df["Predicted"])
                st.dataframe(pred_df)
                csv = pred_df.to_csv().encode()
                st.download_button("Download predictions CSV", csv, file_name=f"{symbol}_predictions.csv")
