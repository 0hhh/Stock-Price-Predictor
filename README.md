# 📈 Stock Price Predictor 

A web app to predict stock prices using an LSTM neural network, built with Streamlit and Keras.
You can enter any stock symbol, view historical data, moving averages, and compare predicted vs actual prices.

**Live Demo:**  
[https://stock-predictor-ai.streamlit.app](https://stock-predictor-ai.streamlit.app)

## Features

- Download historical stock data using yfinance
- Preprocess and scale data for LSTM
- Predict closing prices using a trained LSTM model
- Visualize moving averages and predictions
- Download prediction results as CSV
- Interactive UI with Streamlit

## File Structure

```
stock-price-predictor/
│── app.py                      # Streamlit web app for prediction and visualization
│── model.ipynb                 # Jupyter notebook for training the LSTM model
│── stock_prediction_model.keras # Saved LSTM model (generated after training)
│── scaler.pkl                  # Saved scaler for data normalization
│── requirements.txt            # Python dependencies
│── README.md                   # project description
```

## Setup

1. **Clone the repository**
   ```
   git clone https://github.com/0hhh/stock-price-predictor.git
   cd stock-prediction-app
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Train the model (optional)**
   - Run  `model.ipynb` to train and save the model (`stock_prediction_model.keras`) and scaler (`scaler.pkl`).

4. **Run the app locally**
   ```
   streamlit run app.py
   ```
   Or, if `streamlit` is not recognized:
   ```
   python -m streamlit run app.py
   ```

## Example

<img width="1918" height="1029" alt="image" src="https://github.com/user-attachments/assets/434ec1a4-3ba5-4a03-b70c-66c3be70f622" />

<img width="501" height="733" alt="image" src="https://github.com/user-attachments/assets/38cf2866-8525-43cc-9280-36d87de7ed9b" />

## License

MIT License

---

**Note:**  
- Use Python 3.10 or compatible version.
- For issues with DLL or imports, check your Python environment and package versions.
  
