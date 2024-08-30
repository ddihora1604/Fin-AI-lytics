import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime
import os
import json
import time
from requests.exceptions import ReadTimeout

st.set_page_config(page_title="AI-Driven Personal Finance Manager", page_icon="💰", layout="wide")

# Constants
MODEL_NAME = "facebook/opt-1.3b"
DATA_DIR = "user_data"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def load_ai_model_with_retry(max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            return tokenizer, model
        except ReadTimeout:
            if attempt < max_retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                st.error("Failed to load AI model after several attempts.")
                raise

tokenizer, model = load_ai_model_with_retry()

def generate_ai_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data, stock.info

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

def predict_stock_price(data, days=30):
    data['Date'] = pd.to_datetime(data.index)
    data['Date'] = data['Date'].map(datetime.toordinal)
    
    X = data[['Date', 'Open', 'High', 'Low', 'Volume']]
    y = data['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    last_date = data['Date'].iloc[-1]
    future_dates = pd.date_range(start=datetime.fromordinal(last_date), periods=days+1)[1:]
    future_dates_ordinal = future_dates.map(datetime.toordinal)
    
    future_features = np.array([[date, data['Open'].iloc[-1], data['High'].iloc[-1], 
                                 data['Low'].iloc[-1], data['Volume'].iloc[-1]] 
                                for date in future_dates_ordinal])
    future_features_scaled = scaler.transform(future_features)
    
    predictions = model.predict(future_features_scaled)
    
    return pd.DataFrame({'Date': future_dates, 'Predicted_Close': predictions})

def plot_predictions(historical_data, predictions, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Close'], name='Historical Close'))
    fig.add_trace(go.Scatter(x=predictions['Date'], y=predictions['Predicted_Close'], name='Predicted Close'))
    fig.update_layout(title=f"{ticker} Stock Price Prediction",
                      xaxis_title="Date",
                      yaxis_title="Price (USD)")
    return fig

def track_expenses(expenses):
    df = pd.DataFrame(expenses)
    fig = px.pie(df, values='amount', names='category', title='Expense Distribution')
    return fig

def optimize_portfolio(tickers, initial_weights):
    data = yf.download(tickers, period="1y")['Adj Close']
    returns = data.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = results[1,i] / results[0,i]
        weights_record.append(weights)
    
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights_record[max_sharpe_idx], index=tickers, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2) for i in max_sharpe_allocation.allocation]
    
    return max_sharpe_allocation, sdp, rp

def save_user_data(user_id, data):
    with open(os.path.join(DATA_DIR, f"{user_id}.json"), "w") as f:
        json.dump(data, f)

def load_user_data(user_id):
    try:
        with open(os.path.join(DATA_DIR, f"{user_id}.json"), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"expenses": [], "portfolio": {}}

def main():
    # Sidebar
    with st.sidebar:
        st.title('🤖 AI Finance Assistant')
        selected = st.selectbox(
            "Choose a feature",
            ["Chat", "Stock Analysis", "Expense Tracking", "Portfolio Management"]
        )

    # User Authentication (simplified)
    user_id = st.text_input("Enter your user ID:")
    if not user_id:
        st.warning("Please enter a user ID to continue.")
        return

    user_data = load_user_data(user_id)

    if selected == "Chat":
        st.header("Chat with AI")
        user_input = st.text_area("You:", "")
        if st.button("Send"):
            response = generate_ai_response(user_input)
            st.text_area("AI:", response, height=200)

    elif selected == "Stock Analysis":
        st.header("Stock Analysis")
        ticker = st.text_input("Enter stock ticker:", "AAPL")
        if st.button("Analyze"):
            data, info = get_stock_data(ticker)
            st.write(info)
            st.plotly_chart(plot_stock_chart(data, ticker))
            future_predictions = predict_stock_price(data)
            st.plotly_chart(plot_predictions(data, future_predictions, ticker))

    elif selected == "Expense Tracking":
        st.header("Track Your Expenses")
        category = st.text_input("Category")
        amount = st.number_input("Amount", min_value=0.0, format="%.2f")
        if st.button("Add Expense"):
            user_data.setdefault("expenses", []).append({"category": category, "amount": amount})
            save_user_data(user_id, user_data)
        if user_data.get("expenses"):
            st.plotly_chart(track_expenses(user_data["expenses"]))

    elif selected == "Portfolio Management":
        st.header("Optimize Your Portfolio")
        tickers = st.text_input("Enter tickers separated by commas:", "AAPL,MSFT,GOOGL").split(',')
        initial_weights = [1/len(tickers)] * len(tickers)
        if st.button("Optimize"):
            tickers = [ticker.strip() for ticker in tickers]
            allocation, sdp, rp = optimize_portfolio(tickers, initial_weights)
            st.write(f"Optimal Portfolio Allocation:\n{allocation}")
            st.write(f"Expected Return: {rp:.2f}%")
            st.write(f"Expected Volatility: {sdp:.2f}%")

if __name__ == "__main__":
    main()
