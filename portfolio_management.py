import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy import stats
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate('path/to/your/firebase/credentials.json')
    firebase_admin.initialize_app(cred)
db = firestore.client()

def load_user_data(user_id):
    try:
        user_ref = db.collection('users').document(user_id)
        user_data = user_ref.get().to_dict()
        if user_data is None:
            # Initialize with an empty portfolio if no data exists
            user_data = {"portfolio": {}}
        return user_data
    except Exception as e:
        st.error(f"Error loading user data: {e}")
        return {"portfolio": {}}

def save_user_data(user_id, data):
    try:
        user_ref = db.collection('users').document(user_id)
        # Set the document with the new data, creating it if it does not exist
        user_ref.set(data, merge=True)
    except Exception as e:
        st.error(f"Error saving user data: {e}")


def get_stock_value(ticker, quantity):
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.info.get('regularMarketPrice')
        if current_price is None:
            current_price = stock.history(period="1d")['Close'].iloc[-1]
        return current_price * quantity
    except Exception as e:
        st.error(f"Error fetching stock value for {ticker}: {e}")
        return 0

def perform_portfolio_overview(portfolio_returns, returns, portfolio):
    st.header("Portfolio Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Allocation")
        display_current_portfolio(portfolio)
    
    with col2:
        st.subheader("Key Metrics")
        metrics = calculate_portfolio_metrics(portfolio_returns)
        display_portfolio_metrics(metrics)
    
    st.subheader("Portfolio Returns Over Time")
    plot_portfolio_returns(portfolio_returns.index, portfolio_returns)

def display_current_portfolio(portfolio):
    if portfolio:
        df = pd.DataFrame(list(portfolio.items()), columns=['Stock', 'Quantity'])
        df['Value'] = df.apply(lambda row: get_stock_value(row['Stock'], row['Quantity']), axis=1)
        total_value = df['Value'].sum()
        df['Percentage'] = df['Value'] / total_value * 100 if total_value > 0 else 0
        
        fig = go.Figure(data=[go.Pie(labels=df['Stock'], values=df['Value'], hole=.3)])
        fig.update_layout(title='Portfolio Composition')
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df.style.format({'Quantity': '{:,.0f}', 'Value': '${:,.2f}', 'Percentage': '{:.2f}%'}))
    else:
        st.info("Your portfolio is empty. Add some stocks to get started!")

def add_stock_to_portfolio(user_id, db):
    st.header("Add Stock to Portfolio")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        new_stock = st.text_input("Enter stock ticker:")
    with col2:
        new_quantity = st.number_input("Enter quantity:", min_value=1, step=1)
    with col3:
        if st.button("Add to Portfolio"):
            doc_ref = db.collection("portfolios").document(user_id)
            doc = doc_ref.get()
            if doc.exists:
                portfolio = doc.to_dict()
                if new_stock in portfolio:
                    portfolio[new_stock] += new_quantity
                else:
                    portfolio[new_stock] = new_quantity
            else:
                portfolio = {new_stock: new_quantity}
            
            doc_ref.set(portfolio)
            st.success(f"Added {new_quantity} shares of {new_stock} to your portfolio.")

def perform_portfolio_analysis(portfolio_returns):
    st.header("Portfolio Analysis")
    
    metrics = calculate_portfolio_metrics(portfolio_returns)
    
    col1, col2 = st.columns(2)
    with col1:
        display_portfolio_metrics(metrics)
    with col2:
        plot_portfolio_returns(portfolio_returns.index, portfolio_returns)

def calculate_portfolio_metrics(returns):
    portfolio_return = np.mean(returns) * 252  # Annualized return
    portfolio_volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
    sharpe_ratio = portfolio_return / portfolio_volatility
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    _, p_value = stats.shapiro(returns)
    cumulative_return = (1 + returns).prod() - 1
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    
    return {
        'return': portfolio_return,
        'volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'shapiro_p_value': p_value,
        'cumulative_return': cumulative_return,
        'var_95': var_95,
        'cvar_95': cvar_95
    }

def display_portfolio_metrics(metrics):
    st.subheader("Portfolio Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Annual Return", f"{metrics['return']:.2%}")
        st.metric("Annual Volatility", f"{metrics['volatility']:.2%}")
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        st.metric("Skewness", f"{metrics['skewness']:.2f}")
        st.metric("Kurtosis", f"{metrics['kurtosis']:.2f}")
    with col2:
        st.metric("Shapiro-Wilk p-value", f"{metrics['shapiro_p_value']:.4f}")
        st.metric("Cumulative Return", f"{metrics['cumulative_return']:.2%}")
        st.metric("Value at Risk (95%)", f"{metrics['var_95']:.2%}")
        st.metric("Conditional VaR (95%)", f"{metrics['cvar_95']:.2%}")
    
    with st.expander("Metrics Explanation"):
        st.write("""
        - **Annual Return**: The expected yearly return of the portfolio.
        - **Annual Volatility**: The amount of risk or fluctuation in the portfolio returns.
        - **Sharpe Ratio**: Measures the risk-adjusted return. Higher is better.
        - **Skewness**: Measures the asymmetry of returns. Positive skewness is generally preferred.
        - **Kurtosis**: Measures the tailedness of the return distribution. Higher kurtosis indicates more extreme outcomes.
        - **Shapiro-Wilk p-value**: Tests for normality of returns. A p-value > 0.05 suggests normally distributed returns.
        - **Cumulative Return**: The total return over the period.
        - **Value at Risk (95%)**: The maximum loss expected with 95% confidence over a day.
        - **Conditional VaR (95%)**: The expected loss when the loss exceeds the VaR.
        """)

def plot_portfolio_returns(dates, returns):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=returns, mode='lines', name='Daily Returns'))
    fig.add_trace(go.Scatter(x=dates, y=returns.cumsum(), mode='lines', name='Cumulative Returns'))
    fig.update_layout(title='Portfolio Returns Over Time', xaxis_title='Date', yaxis_title='Returns')
    st.plotly_chart(fig, use_container_width=True)

def portfolio_return(weights, returns):
    return np.sum(returns.mean() * weights) * 252

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

def negative_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
    p_return = portfolio_return(weights, returns)
    p_volatility = portfolio_volatility(weights, cov_matrix)
    return -(p_return - risk_free_rate) / p_volatility

def perform_portfolio_optimization(returns, portfolio):
    st.header("Portfolio Optimization")
    risk_free_rate = st.number_input("Enter risk-free rate:", min_value=0.0, max_value=1.0, value=0.02, step=0.001)
    
    cov_matrix = returns.cov()
    
    # Maximum Sharpe Ratio Portfolio
    msr_weights = maximize_sharpe_ratio(returns, cov_matrix, risk_free_rate)
    
    # Global Minimum Variance Portfolio
    gmv_weights = global_minimum_variance(cov_matrix)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Maximum Sharpe Ratio Portfolio")
        display_portfolio_weights(portfolio.keys(), msr_weights)
        st.write("""
        The Maximum Sharpe Ratio Portfolio aims to find the optimal allocation of assets 
        that maximizes the portfolio's risk-adjusted return (Sharpe ratio).
        """)
    
    with col2:
        st.subheader("Global Minimum Variance Portfolio")
        display_portfolio_weights(portfolio.keys(), gmv_weights)
        st.write("""
        The Global Minimum Variance Portfolio represents the asset allocation with the lowest 
        possible volatility. This portfolio is optimal for highly risk-averse investors.
        """)

def display_portfolio_weights(tickers, weights):
    df = pd.DataFrame({'Stock': tickers, 'Weight': weights})
    fig = go.Figure(data=[go.Bar(x=df['Stock'], y=df['Weight'])])
    fig.update_layout(title='Portfolio Weights', xaxis_title='Stock', yaxis_title='Weight')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df.style.format({'Weight': '{:.2%}'}))

def maximize_sharpe_ratio(returns, cov_matrix, risk_free_rate):
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
    args = (cov_matrix,)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    
    result = minimize(lambda w: portfolio_volatility(w, cov_matrix), num_assets*[1./num_assets],
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

def perform_monte_carlo_var(returns, portfolio, confidence_level=0.95, num_simulations=10000, time_horizon=252):
    st.header("Monte Carlo Value at Risk (VaR) Simulation")
    
    portfolio_value = sum(get_stock_value(ticker, quantity) for ticker, quantity in portfolio.items())
    weights = np.array([get_stock_value(ticker, quantity) / portfolio_value for ticker, quantity in portfolio.items()])
    
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    
    simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, (num_simulations, time_horizon))
    simulated_portfolio_returns = np.dot(simulated_returns, weights)
    simulated_portfolio_values = portfolio_value * (1 + simulated_portfolio_returns).cumprod(axis=1)
    
    final_values = simulated_portfolio_values[:, -1]
    var = np.percentile(final_values, 100 * (1 - confidence_level))
    cvar = final_values[final_values <= var].mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Monte Carlo VaR (95%)", f"${portfolio_value - var:.2f}")
        st.metric("Monte Carlo CVaR (95%)", f"${portfolio_value - cvar:.2f}")
    
    with col2:
        st.write("""
        **Value at Risk (VaR)**: The maximum loss expected with 95% confidence over the simulation period.
        **Conditional VaR (CVaR)**: The expected loss when the loss exceeds the VaR.
        """)
    
    fig = go.Figure()
    for i in range(min(100, num_simulations)):  # Plot first 100 simulations
        fig.add_trace(go.Scatter(y=simulated_portfolio_values[i], mode='lines', name=f'Simulation {i+1}', opacity=0.1))
    fig.update_layout(title='Monte Carlo Simulations of Portfolio Value', xaxis_title='Days', yaxis_title='Portfolio Value')
    st.plotly_chart(fig, use_container_width=True)

def portfolio_management_interface(user_id, db):
    st.title('📊 Advanced Portfolio Management')
    
    # Fetch portfolio data from Firestore
    doc_ref = db.collection("portfolios").document(user_id)
    doc = doc_ref.get()
    if doc.exists:
        portfolio = doc.to_dict()
    else:
        portfolio = {}
    
    display_current_portfolio(portfolio)
    add_stock_to_portfolio(user_id, db)
    
    if portfolio:
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
        
        st.sidebar.title("Navigation")
        analysis_options = ["Portfolio Overview", "Portfolio Analysis", "Portfolio Optimization", "Monte Carlo Simulation"]
        selected_analysis = st.sidebar.radio("Choose Analysis", analysis_options)
        
        if selected_analysis == "Portfolio Overview":
            perform_portfolio_overview(portfolio_returns, returns, portfolio)
        elif selected_analysis == "Portfolio Analysis":
            perform_portfolio_analysis(portfolio_returns)
        elif selected_analysis == "Portfolio Optimization":
            perform_portfolio_optimization(returns, portfolio)
        elif selected_analysis == "Monte Carlo Simulation":
            perform_monte_carlo_var(returns, portfolio)

if __name__ == "__main__":
    # This block will be executed if the script is run directly
    # You can add any initialization code or test calls here
    pass
