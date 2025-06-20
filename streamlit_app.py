import streamlit as st
from model_arima import forecast_arima
from model_lstm import forecast_lstm
from model_prophet import forecast_prophet
from data_preprocessing import load_data

st.set_page_config(page_title="Stock Forecasting Dashboard", layout="wide")

st.title("📈 Stock Market Forecasting")

ticker = st.text_input("Enter Stock Symbol (e.g. AAPL, TSLA)", "AAPL")
model_choice = st.selectbox("Select Forecasting Model", ["ARIMA", "LSTM", "Prophet"])

if st.button("Run Forecast"):
    df = load_data(ticker)
    st.subheader("📊 Forecast Result")

    if model_choice == "ARIMA":
        forecast = forecast_arima(df)
    elif model_choice == "LSTM":
        forecast = forecast_lstm(df)
    else:
        forecast = forecast_prophet(df)

    st.line_chart(forecast)
