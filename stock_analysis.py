import streamlit as st
import plotly.graph_objs as go
from financial_data import get_stock_data
from ml_models import predict_stock_price_lstm

def stock_analysis_interface():
    st.title('📈 Stock Analysis and Prediction')
    ticker = st.text_input("Enter stock ticker (e.g., AAPL, GOOGL):")
    if ticker:
        data, info = get_stock_data(ticker)
        if not data.empty:
            display_stock_info(data, info)
            st.plotly_chart(plot_stock_chart(data, ticker))
            
            predictions = predict_stock_price_lstm(data)
            st.plotly_chart(plot_predictions(data, predictions, ticker))

def display_stock_info(data, info):
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
    col2.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,}")
    col3.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}")

def plot_stock_chart(data, ticker):
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])
    fig.update_layout(title=f"{ticker} Stock Price",
                      xaxis_title="Date",
                      yaxis_title="Price (USD)")
    return fig

def plot_predictions(historical_data, predictions, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Close'], name='Historical Close'))
    fig.add_trace(go.Scatter(x=predictions['Date'], y=predictions['Predicted_Close'], name='Predicted Close'))
    fig.update_layout(title=f"{ticker} Stock Price Prediction",
                      xaxis_title="Date",
                      yaxis_title="Price (USD)")
    return fig