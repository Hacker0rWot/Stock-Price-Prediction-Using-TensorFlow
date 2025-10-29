# 📈 Stock Price Predictor (LSTM + Streamlit)

A **deep learning-based stock price prediction app** built using **TensorFlow LSTM** and deployed with **Streamlit**. This project predicts future stock prices using historical data from Yahoo Finance and provides an interactive dashboard to visualize predictions.

---

## 🔹 Project Overview

- Predicts stock closing prices using a **Long Short-Term Memory (LSTM) neural network**.
- Uses historical stock data from **Yahoo Finance**.
- Provides **interactive visualization** of historical and predicted prices.
- Supports **multi-day predictions** (1–30 days).
- Saves predictions as a downloadable CSV file.

**Key Features:**
- LSTM-based model with 2 stacked layers.
- Data scaled using `MinMaxScaler`.
- Streamlit app for user-friendly interface.
- Real-time fetching of stock data and autoregressive multi-day prediction.

---

## 🗂️ Repository Structure
stock-price-predictor/
│
├── app.py # Main Streamlit app
├── models/ # Pretrained model & scaler
│ ├── stock_model.h5 # Trained LSTM model
│ └── scaler.pkl # MinMaxScaler used during training
├── README.md # Project documentation
└── requirements.txt # Python dependencies

---

## ⚙️ Requirements

- Python 3.9+
- TensorFlow
- Pandas
- NumPy
- Streamlit
- yfinance
- scikit-learn
- joblib
- matplotlib, seaborn (optional, for visualization)

Install dependencies via:

```bash
pip install -r requirements.txt

```



## This project uses pre-trained files for prediction:

1.stock_model.h5

This is the trained LSTM neural network saved after training on historical stock data.

Contains all learned weights of the model.

Used in the Streamlit app to predict future stock prices without retraining.

2.scaler.pkl

This is the MinMaxScaler object used to scale the data during training.

Scales price values to the range [0,1] before feeding them to the LSTM.

Required to transform input data and inverse-transform predictions back to actual stock prices.

Ensures consistency between training and inference.

🔹 Why we load pre-trained models

Avoids retraining the model every time — saves time and computation.

Maintains scaling consistency: the model was trained on scaled data, so input must be scaled in the same way.

Allows the Streamlit app to immediately predict stock prices using new ticker data.

## **Load the models and run Streamlit app using Streamlit run app.py**
