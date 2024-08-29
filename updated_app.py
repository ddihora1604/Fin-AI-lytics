import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime, timedelta
import os
import json
import replicate
import requests
from dotenv import load_dotenv
import statsmodels.api as sm
import pandas_datareader as pdr

# Load environment variables
load_dotenv()

# Constants
DATA_DIR = "user_data"

# Get API keys from environment variables
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Ensure environment variables are set
if not REPLICATE_API_TOKEN:
    raise ValueError("Replicate API token is not set. Please check your .env file.")
if not ALPHA_VANTAGE_API_KEY:
    raise ValueError("Alpha Vantage API key is not set. Please check your .env file.")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize Replicate client
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

def generate_ai_response(prompt):
    try:
        output = replicate_client.run(
            "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            input={"prompt": prompt, "max_length": 200, "temperature": 0.7}
        )
        return ''.join(output)
    except Exception as e:
        st.error(f"Error generating AI response: {e}")
        return "Error generating response."

def get_financial_news():
    try:
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get('feed', [])[:5]  # Return the top 5 news items
    except requests.RequestException as e:
        st.error(f"Error fetching financial news: {e}")
        return []

def get_stock_data(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        return data, stock.info
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame(), {}

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

def predict_stock_price_lstm(data, days=30):
    # Prepare data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

    # Create training dataset
    training_data_len = int(np.ceil( len(scaled_data) * .95 ))
    train_data = scaled_data[0:training_data_len, :]
    x_train, y_train = [], []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create testing dataset
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Predict future prices
    last_60_days = scaled_data[-60:]
    X_future = []
    for i in range(days):
        X_future.append(last_60_days[-60:])
        prediction = model.predict(np.array(X_future).reshape(1, 60, 1))
        last_60_days = np.append(last_60_days, prediction)
        X_future = []

    future_predictions = scaler.inverse_transform(last_60_days[-days:].reshape(-1, 1))

    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days)
    predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_predictions.flatten()})

    return predictions_df

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
    expense_fig = px.pie(df, values='amount', names='category', title='Expense Distribution')
    return expense_fig

def calculate_portfolio_metrics(returns):
    portfolio_return = np.mean(returns)
    portfolio_volatility = np.std(returns)
    sharpe_ratio = portfolio_return / portfolio_volatility
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    _, p_value = stats.shapiro(returns)
    cumulative_return = (1 + returns).prod() - 1
    return portfolio_return, portfolio_volatility, sharpe_ratio, skewness, kurtosis, p_value, cumulative_return

def portfolio_return(weights, returns):
    return np.sum(returns.mean() * weights) * 252

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

def negative_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
    p_return = portfolio_return(weights, returns)
    p_volatility = portfolio_volatility(weights, cov_matrix)
    return -(p_return - risk_free_rate) / p_volatility

def optimize_portfolio(returns, cov_matrix, risk_free_rate):
    num_assets = len(returns.columns)
    args = (returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    
    result = minimize(negative_sharpe_ratio, num_assets*[1./num_assets], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

def global_minimum_variance(cov_matrix):
    num_assets = cov_matrix.shape[0]
    args = (cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    
    result = minimize(lambda w: portfolio_volatility(w, cov_matrix), num_assets*[1./num_assets],
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

def fama_french_3_factor(returns, start_date, end_date):
    try:
        # Fetch Fama-French factors
        ff_factors = pdr.get_data_famafrench('F-F_Research_Data_Factors_daily', start=start_date, end=end_date)[0]
        ff_factors = ff_factors.div(100)  # Convert to decimal format
        
        # Align the dates
        aligned_data = pd.concat([returns, ff_factors], axis=1).dropna()
        
        # Prepare the data for regression
        y = aligned_data.iloc[:, 0]  # Portfolio returns
        X = sm.add_constant(aligned_data.iloc[:, 1:])  # Factors + constant
        
        # Run the regression
        model = sm.OLS(y, X).fit()
        return model
    except Exception as e:
        st.error(f"Error in Fama-French model: {e}")
        return None

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
    st.set_page_config(page_title="AI-Driven Personal Finance Manager", page_icon="💰", layout="wide")

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
        user_input = st.text_input("Ask about finances, stocks, insurance, or request financial news:")
        if user_input:
            if "financial news" in user_input.lower():
                news_items = get_financial_news()
                st.write("Here are the latest financial news items:")
                for item in news_items:
                    st.subheader(item['title'])
                    st.write(f"Source: {item['source']}")
                    st.write(f"Summary: {item['summary']}")
                    st.write(f"URL: {item['url']}")
                    st.write("---")
            else:
                response = generate_ai_response(user_input)
                st.write(response)

    elif selected == "Stock Analysis":
        st.title('📈 Stock Analysis and Prediction')
        ticker = st.text_input("Enter stock ticker (e.g., AAPL, GOOGL):")
        if ticker:
            data, info = get_stock_data(ticker)
            if not data.empty:
                st.write(f"Current Price: ${data['Close'].iloc[-1]:.2f}")
                st.plotly_chart(plot_stock_chart(data, ticker))
                
                predictions = predict_stock_price_lstm(data)
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
        st.title('📊 Advanced Portfolio Management')
        
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
        
        # Portfolio Analysis
        if portfolio:
            st.subheader("Portfolio Analysis")
            
            # Fetch historical data
            tickers = list(portfolio.keys())
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
            
            # Calculate returns
            returns = data.pct_change().dropna()
            
            # Calculate portfolio metrics
            weights = np.array([portfolio[ticker] for ticker in tickers])
            weights = weights / np.sum(weights)
            portfolio_returns = returns.dot(weights)
            
            p_return, p_volatility, sharpe_ratio, skewness, kurtosis, p_value, cumulative_return = calculate_portfolio_metrics(portfolio_returns)
            
            st.write(f"Portfolio Return: {p_return:.2%}")
            st.write(f"Portfolio Volatility: {p_volatility:.2%}")
            st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            st.write(f"Skewness: {skewness:.2f}")
            st.write(f"Kurtosis: {kurtosis:.2f}")
            st.write(f"Shapiro-Wilk p-value: {p_value:.4f}")
            st.write(f"Cumulative Return: {cumulative_return:.2%}")
            
            # Plot portfolio returns
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=returns.index, y=portfolio_returns, mode='lines', name='Portfolio Returns'))
            fig.update_layout(title='Portfolio Returns Over Time', xaxis_title='Date', yaxis_title='Returns')
            st.plotly_chart(fig)
            
            # Markowitz Portfolio Optimization
            st.subheader("Markowitz Portfolio Optimization")
            risk_free_rate = st.number_input("Enter risk-free rate:", min_value=0.0, max_value=1.0, value=0.02, step=0.001)
            
            cov_matrix = returns.cov()
            optimal_weights = optimize_portfolio(returns, cov_matrix, risk_free_rate)
            
            st.write("Optimal Portfolio Weights:")
            for ticker, weight in zip(tickers, optimal_weights):
                st.write(f"{ticker}: {weight:.2%}")
            
            # Global Minimum Variance Portfolio
            st.subheader("Global Minimum Variance Portfolio")
            gmv_weights = global_minimum_variance(cov_matrix)
            
            st.write("Global Minimum Variance Portfolio Weights:")
            for ticker, weight in zip(tickers, gmv_weights):
                st.write(f"{ticker}: {weight:.2%}")
            
            # Fama-French 3-Factor Model
            st.subheader("Fama-French 3-Factor Model")
            ff3_model = fama_french_3_factor(portfolio_returns, start_date, end_date)
            if ff3_model:
                st.write("Fama-French 3-Factor Model Results:")
                st.write(ff3_model.summary())
            else:
                st.write("Unable to perform Fama-French 3-Factor analysis.")

if __name__ == "__main__":
    main()
