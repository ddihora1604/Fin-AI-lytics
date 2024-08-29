import yfinance as yf
import requests
import os
from dotenv import load_dotenv
import pandas as pd
import streamlit as st

# Load environment variables
load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

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
        print(f"yfinance: Data for {ticker}: {data.head()}")  # Debugging statement
        return data, stock.info
    except Exception as e:
        st.error(f"Error fetching stock data using yfinance: {e}")
        return pd.DataFrame(), {}

def get_stock_value(ticker, quantity):
    try:
        # Primary method: Use yfinance
        stock = yf.Ticker(ticker)
        current_price = stock.info.get('regularMarketPrice')

        # Debugging statement: print the fetched price
        print(f"yfinance: Current price for {ticker} is {current_price}")

        if current_price is None:
            raise ValueError("yfinance failed to fetch price.")

        return current_price * quantity

    except Exception as e:
        st.error(f"Error fetching stock value for {ticker} using yfinance: {e}")
        return 0
