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
from datetime import datetime, timedelta
import os
import json

st.set_page_config(page_title="AI-Driven Personal Finance Manager", page_icon="💰", layout="wide")

# Constants
MODEL_NAME = "facebook/opt-1.3b"
DATA_DIR = "user_data"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Load AI model
@st.cache_resource
def load_ai_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_ai_model()

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
        st.title('💬 AI Financial Assistant')
        user_input = st.text_input("Ask about finances, stocks, or insurance:")
        if user_input:
            response = generate_ai_response(user_input)
            st.write(response)

    elif selected == "Stock Analysis":
        st.title('📈 Stock Analysis and Prediction')
        ticker = st.text_input("Enter stock ticker (e.g., AAPL, GOOGL):")
        if ticker:
            data, info = get_stock_data(ticker)
            st.write(f"Current Price: ${data['Close'].iloc[-1]:.2f}")
            st.plotly_chart(plot_stock_chart(data, ticker))
            
            predictions = predict_stock_price(data)
            st.plotly_chart(plot_predictions(data, predictions, ticker))

    elif selected == "Expense Tracking":
        st.title('💸 Expense Tracking')
        
        expenses = user_data.get("expenses", [])
        
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Amount", min_value=0.0)
        with col2:
            category = st.selectbox("Category", ["Food", "Transport", "Entertainment", "Utilities", "Other"])
        
        if st.button("Add Expense"):
            expenses.append({"amount": amount, "category": category})
            user_data["expenses"] = expenses
            save_user_data(user_id, user_data)
            st.success("Expense added successfully!")
        
        if expenses:
            st.plotly_chart(track_expenses(expenses))

    elif selected == "Portfolio Management":
        st.title('📊 Portfolio Management')
        
        portfolio = user_data.get("portfolio", {})
        
        # Display current portfolio
        st.subheader("Current Portfolio")
        for stock, quantity in portfolio.items():
            st.write(f"{stock}: {quantity} shares")
        
        # Add new stock to portfolio
        st.subheader("Add Stock to Portfolio")
        new_stock = st.text_input("Enter stock ticker:")
        new_quantity = st.number_input("Enter quantity:", min_value=1, step=1)
        if st.button("Add to Portfolio"):
            if new_stock in portfolio:
                portfolio[new_stock] += new_quantity
            else:
                portfolio[new_stock] = new_quantity
            user_data["portfolio"] = portfolio
            save_user_data(user_id, user_data)
            st.success(f"Added {new_quantity} shares of {new_stock} to your portfolio.")
        
        # Portfolio Optimization
        if portfolio:
            st.subheader("Portfolio Optimization")
            if st.button("Optimize Portfolio"):
                tickers = list(portfolio.keys())
                initial_weights = np.array([portfolio[ticker] for ticker in tickers])
                initial_weights = initial_weights / np.sum(initial_weights)
                
                optimized_portfolio, sdp, rp = optimize_portfolio(tickers, initial_weights)
                
                st.write("Optimized Portfolio Allocation:")
                st.dataframe(optimized_portfolio)
                st.write(f"Expected annual return: {rp*100:.2f}%")
                st.write(f"Annual volatility: {sdp*100:.2f}%")
                st.write(f"Sharpe Ratio: {rp/sdp:.2f}")

if __name__ == "__main__":
    main()