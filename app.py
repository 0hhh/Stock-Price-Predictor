import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import joblib
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load model & scaler
@st.cache_resource
def load_trained_model():
    return load_model("stock_prediction_model.keras")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_trained_model()
scaler = load_scaler()

# Streamlit UI
st.title("📈 Stock Market Predictor (LSTM)")
stock = st.text_input("Enter Stock Symbol", "GOOG")
start = "2012-01-01"
end = "2022-12-31"

# Fetch data
data = yf.download(stock, start, end)
st.subheader("Raw Stock Data")
st.write(data.tail())

# Train/test split
train_size = int(len(data) * 0.80)
data_train = data[:train_size][["Close"]]
data_test = data[train_size:][["Close"]]

# Prepare test data
past_100 = data_train.tail(100)
data_test_full = pd.concat([past_100, data_test], ignore_index=True)
data_test_scaled = scaler.transform(data_test_full)

x_test, y_test = [], []
for i in range(100, data_test_scaled.shape[0]):
    x_test.append(data_test_scaled[i - 100:i])
    y_test.append(data_test_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predictions
y_pred = model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Metrics
mae = mean_absolute_error(y_test, y_pred)
# rmse = mean_squared_error(y_test, y_pred, squared=False)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.metric("MAE", f"{mae:.2f}")
st.metric("RMSE", f"{rmse:.2f}")

# Plot moving averages
st.subheader("Moving Averages")
ma50 = data.Close.rolling(50).mean()
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()

fig1 = plt.figure(figsize=(10, 6))
plt.plot(data.Close, label="Close", color="g")
plt.plot(ma50, label="MA50", color="r")
plt.plot(ma100, label="MA100", color="b")
plt.plot(ma200, label="MA200", color="m")
plt.legend()
st.pyplot(fig1)

# Plot predictions
st.subheader("Predicted vs Original Stock Price")
fig2 = plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Original Price", color="g")
plt.plot(y_pred, label="Predicted Price", color="r")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)

# Download predictions
st.subheader("Download Predicted Results")
results = pd.DataFrame({"Original": y_test.flatten(), "Predicted": y_pred.flatten()})
st.download_button("Download CSV", results.to_csv(index=False), "predictions.csv", "text/csv")
