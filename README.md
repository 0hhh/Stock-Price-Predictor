# 🧦📈Stock Price Predictor 

A simple web app to predict stock prices using an LSTM neural network, built with Streamlit and Keras.  
You can enter any stock symbol, view historical data, moving averages, and compare predicted vs actual prices.

## Features

- Download historical stock data using [yfinance](https://github.com/ranaroussi/yfinance)
- Preprocess and scale data for LSTM
- Predict closing prices using a trained LSTM model
- Visualize moving averages and predictions
- Download prediction results as CSV
- Interactive UI with [Streamlit](https://streamlit.io/)

## Setup

1. **Clone the repository**
   ```
   git clone https://github.com/0hhh/stock-price-predictor.git
   cd stock-price-predictor
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```
   Or manually:
   ```
   pip install numpy pandas yfinance matplotlib streamlit joblib scikit-learn keras tensorflow
   ```

3. **Train the model (optional)**
   - Run `model.ipynb` in Jupyter to train and save the model (`stock_prediction_model.keras`) and scaler (`scaler.pkl`).

4. **Run the app**
   ```
   streamlit run app.py
   ```
   If `streamlit` is not recognized, use:
   ```
   python -m streamlit run app.py
   ```

## Files

- `model.ipynb` — Jupyter notebook for training the LSTM model
- `app.py` — Streamlit web app for prediction and visualization
- `stock_prediction_model.keras` — Saved LSTM model (generated after training)
- `scaler.pkl` — Saved scaler for data normalization
- `requirements.txt` — Python dependencies

## Example

![App Screenshot](screenshot.png)

## License

MIT License

---

**Note:**  
- Make sure you are using Python 3.10 or compatible version.
- If you encounter DLL or import errors, check your Python environment and package versions.
