import streamlit as st
from real_time import real_time_stock_analysis
from ai_walkthrough import AIWalkthroughAssistant
from debt_management import debt_management_interface
from bs import BudgetSavingsManager
# Ensure st.set_page_config is called before any other Streamlit commands.
if 'page_config_set' not in st.session_state:
    st.set_page_config(page_title="AI-Driven Personal Finance Manager", page_icon="💰", layout="wide")
    st.session_state.page_config_set = True

# Import necessary libraries and initialize Firebase as before
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
import requests
import firebase_admin
from firebase_admin import credentials, firestore

# Constants
MODEL_NAME = "facebook/opt-1.3b"

# Initialize Firebase app
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate("firebase-adminsdk.json")
        firebase_admin.initialize_app(cred)

# Initialize Firestore
def get_firestore_client():
    return firestore.client()

# Call the initialization functions
init_firebase()
db = get_firestore_client()

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

def save_chat_history(user_id, prompt, response):
    doc_ref = db.collection("chat_history").document(user_id)
    chat_data = {
        'prompt': prompt,
        'response': response,
        'timestamp': firestore.SERVER_TIMESTAMP
    }
    doc_ref.collection("messages").add(chat_data)

def load_chat_history(user_id):
    doc_ref = db.collection("chat_history").document(user_id)
    messages = doc_ref.collection("messages").order_by("timestamp").stream()
    history = []
    for message in messages:
        history.append(message.to_dict())
    return history

def save_user_data(user_id, data):
    doc_ref = db.collection("user_data").document(user_id)
    doc_ref.set(data)

def load_user_data(user_id):
    doc_ref = db.collection("user_data").document(user_id)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    else:
        return {}

def sign_in(email, password):
    try:
        response = requests.post(FIREBASE_AUTH_URL, json={
            'email': email,
            'password': password,
            'returnSecureToken': True
        })
        response_data = response.json()
        if 'idToken' in response_data:
            return response_data['idToken']
        else:
            st.error(response_data.get('error', {}).get('message', 'Unknown error'))
            return None
    except Exception as e:
        st.error(f"Error during sign-in: {str(e)}")
        return None

def sign_up(email, password):
    try:
        response = requests.post(FIREBASE_SIGNUP_URL, json={
            'email': email,
            'password': password,
            'returnSecureToken': True
        })
        response_data = response.json()
        if 'idToken' in response_data:
            return response_data['idToken']
        else:
            st.error(response_data.get('error', {}).get('message', 'Unknown error'))
            return None
    except Exception as e:
        st.error(f"Error during sign-up: {str(e)}")
        return None

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_id = None

    if not st.session_state.logged_in:
        st.title("Firebase Authentication")
        mode = st.selectbox("Select Mode", ["Login", "Sign Up"])

        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        if mode == "Login":
            if st.button("Login"):
                token = sign_in(email, password)
                if token:
                    st.session_state.logged_in = True
                    st.session_state.user_id = email
                    st.session_state.current_page = "Financial Advice"
                # No page reload, so manual update needed

        elif mode == "Sign Up":
            if st.button("Sign Up"):
                token = sign_up(email, password)
                if token:
                    st.session_state.logged_in = True
                    st.session_state.user_id = email
                    st.session_state.current_page = "Financial Advice"
                # No page reload, so manual update needed

    if st.session_state.logged_in:
        user_id = st.session_state.user_id
        user_data = load_user_data(user_id)
        
        st.sidebar.title("Navigation")
        selected = st.sidebar.selectbox("Go to", ["Financial Advice", "Real-time Stock Analysis", "AI Finance Manager", "Portfolio Management", "Recommender"])

        st.session_state.current_page = selected

        if selected == "Financial Advice":
            ai_assistant = AIWalkthroughAssistant(user_id, user_data)
            selected = ai_assistant.run() 
            st.title('💬 Financial Advice and Chat')
            user_input = st.text_input("Ask about finances, stocks, or insurance:")

            if st.button("Submit"):
                if user_input:
                    response = generate_ai_response(user_input)
                    st.write(response)
                    save_chat_history(user_id, user_input, response)
                    chat_history = load_chat_history(user_id)
                    for entry in chat_history:
                        st.write(f"**{entry['timestamp']}**\n**You:** {entry['prompt']}\n**AI:** {entry['response']}\n")
                else:
                    st.warning("Please enter a question before submitting.")

        elif selected == "Real-time Stock Analysis":
            real_time_stock_analysis()

        elif selected == "Portfolio Management":
            from Pm import portfolio_management_interface
            portfolio_management_interface(user_id)
        elif selected == "Recommender":
            from recommender import recommend_stocks  # Import your function
            st.title("AI Finance Expected Return Recommendation")
            st.write("Get personalized expected return recommendations based on your risk tolerance, investment timeline, and financial goals.")
            container = st.container()
            with container:
                # User Input Fields
                risk_level_selected = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
                investment_timeline = st.number_input("Investment Timeline (Years)", min_value=1)
                financial_goals_selected = st.selectbox("Financial Goals", [
                    "Wealth Accumulation",
                    "Children's Education",
                    "Buying a House",
                    "Retirement Savings",
                    "Travel Fund",
                    "Emergency Fund"
                ])

            # Button to trigger recommendation
            if st.button("Recommend Expected Return"):
                predicted_return = recommend_stocks(risk_level_selected, investment_timeline, financial_goals_selected)
                st.success(f'Predicted Expected return : {predicted_return}')
    
        elif selected == 'AI Finance Manager':
            from ai_finance_manager import run_advanced_ai_finance_manager
            run_advanced_ai_finance_manager(user_id)  
if __name__ == "__main__":
    main()
    