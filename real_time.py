import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from ml_models import predict_stock_price_lstm



def get_stock_data(ticker, period="3mo", interval="1d"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        return data, stock.info
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame(), {}



def plot_historical_and_predicted_prices(historical_data, predictions, ticker):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['Close'],
        name='Historical Close',
        line=dict(color='blue')
    ))

    
    fig.add_trace(go.Scatter(
        x=predictions['Date'],
        y=predictions['Predicted_Close'],
        name='Predicted Close',
        line=dict(color='red', dash='dash')
    ))

   
    last_historical_date = historical_data.index[-1]
    fig.add_vline(x=last_historical_date, line_dash="dash", line_color="green")

    fig.update_layout(
        title=f"{ticker} Stock Price - 3 Months Historical and 1 Month Prediction",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend_title="Data Type",
        hovermode="x unified"
    )

    return fig
def plot_candlestick_chart(data, ticker):
    if data.empty or 'Open' not in data.columns:
        st.warning("No data available for candlestick chart")
        return go.Figure()

    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])
    fig.update_layout(
        title=f"{ticker} Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price (USD)"
    )
    return fig

def plot_volume_chart(data, ticker):
    if data.empty or 'Volume' not in data.columns:
        st.warning("No data available for volume chart")
        return go.Figure()

    fig = go.Figure(data=[go.Bar(x=data.index, y=data['Volume'])])
    fig.update_layout(
        title=f"{ticker} Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume"
    )
    return fig

def plot_moving_averages(data, ticker):
    if data.empty or 'Close' not in data.columns:
        st.warning("No data available for moving averages")
        return go.Figure()

    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA5'], name='5-day MA'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], name='20-day MA'))
    fig.update_layout(
        title=f"{ticker} Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price (USD)"
    )
    return fig

def calculate_rsi(data, periods=14):
    if data.empty or 'Close' not in data.columns:
        return pd.Series()

    close_delta = data['Close'].diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))
    return rsi

def plot_rsi(data, ticker):
    if data.empty:
        st.warning("No data available for RSI")
        return go.Figure()

    rsi = calculate_rsi(data)
    fig = go.Figure(data=[go.Scatter(x=data.index, y=rsi, name='RSI')])
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(
        title=f"{ticker} Relative Strength Index (RSI)",
        xaxis_title="Date",
        yaxis_title="RSI"
    )
    return fig

def calculate_bollinger_bands(data, window=20, num_std=2):
    if data.empty or 'Close' not in data.columns:
        return pd.DataFrame()

    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return pd.DataFrame({'Upper': upper_band, 'Middle': rolling_mean, 'Lower': lower_band})

def plot_bollinger_bands(data, ticker):
    if data.empty:
        st.warning("No data available for Bollinger Bands")
        return go.Figure()

    bb = calculate_bollinger_bands(data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close'))
    fig.add_trace(go.Scatter(x=data.index, y=bb['Upper'], name='Upper Band'))
    fig.add_trace(go.Scatter(x=data.index, y=bb['Middle'], name='Middle Band'))
    fig.add_trace(go.Scatter(x=data.index, y=bb['Lower'], name='Lower Band'))
    fig.update_layout(
        title=f"{ticker} Bollinger Bands",
        xaxis_title="Date",
        yaxis_title="Price (USD)"
    )
    return fig

def day_trading_analysis(data):
    if data.empty:
        return "Insufficient data for day trading analysis."

    current_price = data['Close'].iloc[-1]
    open_price = data['Open'].iloc[0]
    high = data['High'].max()
    low = data['Low'].min()
    volume = data['Volume'].sum()

    analysis = f"""
    Day Trading Analysis:
    - Opening Price: ${open_price:.2f}
    - Current Price: ${current_price:.2f}
    - Daily High: ${high:.2f}
    - Daily Low: ${low:.2f}
    - Volume: {volume:,}

    Key Levels:
    - Resistance: ${high:.2f}
    - Support: ${low:.2f}

    Potential Strategies:
    1. Breakout Trading: Watch for a break above ${high:.2f} or below ${low:.2f}
    2. Range Trading: Consider buying near ${low:.2f} and selling near ${high:.2f}
    3. Trend Following: The overall trend is {'upward' if current_price > open_price else 'downward'}

    Always use proper risk management and consider using stop-loss orders.
    """
    return analysis

def calculate_atr(data, period=14):
    if data.empty or 'High' not in data.columns or 'Low' not in data.columns or 'Close' not in data.columns:
        return pd.Series()

    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()
    return atr

def risk_management_analysis(data, ticker_info):
    if data.empty:
        return "Insufficient data for risk management analysis."

    current_price = data['Close'].iloc[-1]
    atr = calculate_atr(data).iloc[-1]
    beta = ticker_info.get('beta', None)

    analysis = f"""
    Risk Management Analysis:
    - Current Price: ${current_price:.2f}
    - Average True Range (ATR): ${atr:.2f}
    - Beta: {beta if beta else 'N/A'}

    Suggested Risk Management:
    1. Position Sizing: Consider risking no more than 1-2% of your trading capital per trade.
    2. Stop Loss: Place a stop loss 1-2 ATR (${atr:.2f} - ${(2*atr):.2f}) away from your entry point.
    3. Take Profit: Consider a risk-reward ratio of at least 1:2 or 1:3.

    Volatility Analysis:
    - The stock's ATR of ${atr:.2f} indicates {'high' if atr/current_price > 0.02 else 'moderate to low'} volatility.
    - {'The beta of ' + str(beta) + ' suggests this stock is ' + ('more' if beta > 1 else 'less') + ' volatile than the overall market.' if beta else 'Beta information is not available.'}

    Remember to always adapt your risk management strategy to your personal risk tolerance and overall market conditions.
    """
    return analysis

def plot_predictions(historical_data, predictions, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Close'], name='Historical Close'))
    fig.add_trace(go.Scatter(x=predictions['Date'], y=predictions['Predicted_Close'], name='Predicted Close'))
    fig.update_layout(title=f"{ticker} Stock Price Prediction",
                      xaxis_title="Date",
                      yaxis_title="Price (USD)")
    return fig



def real_time_stock_analysis():

    

    st.title("💲 Real-time Stock Analysis and Prediction")
    
    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        ticker = st.text_input("Enter stock ticker :")
    
    if not ticker:
        st.warning("Please enter a valid stock ticker.")
        return

    placeholder = st.empty()

    while True:
        with placeholder.container():
            data, info = get_stock_data(ticker)
            
            
            if not data.empty:
            
                st.markdown('<p class="sub-header">Key Metrics</p>', unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}", f"{(data['Close'].iloc[-1] - data['Close'].iloc[-2]):.2f} ({((data['Close'].iloc[-1] - data['Close'].iloc[-2])/data['Close'].iloc[-2]*100):.2f}%)")
                
                market_cap = info.get('marketCap', 'N/A')
                if market_cap != 'N/A':
                    market_cap_billion = market_cap / 1_000_000_000
                    col2.metric("Market Cap", f"${market_cap_billion:,.2f}B")
                else:
                    col2.metric("Market Cap", "N/A")
                col3.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}")
                col4.metric("52 Week Low", f"${info.get('fiftyTwoWeekLow', 'N/A'):.2f}")

                st.markdown('<p class="sub-header">Price Prediction</p>', unsafe_allow_html=True)
                predictions = predict_stock_price_lstm(data, days=30)
                with st.expander("View Price Prediction Chart", expanded=True):
                    st.plotly_chart(plot_historical_and_predicted_prices(data, predictions, ticker), use_container_width=True, config={'displayModeBar': False})
                    st.markdown("""
                    <div class="insight-box">
                    <strong>Historical and Future Predicted Price Explanation:</strong><br>
                    • Blue line: Last 3 months of historical closing prices<br>
                    • Red dashed line: Predicted prices for the next month<br>
                    • Vertical green dashed line: Separates historical data from predictions<br>
                    • Predictions are based on an LSTM neural network model<br>
                    • Note: Predictions are estimates and should not be the sole basis for investment decisions
                    </div>
                    """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<p class="sub-header">Price Action</p>', unsafe_allow_html=True)
                    with st.expander("View Candlestick Chart", expanded=True):
                        st.plotly_chart(plot_candlestick_chart(data, ticker), use_container_width=True, config={'displayModeBar': False})
                        st.markdown("""
                **Candlestick Chart Explanation:**
                - Each candle represents a time period (usually 1 day)
                - Green candles indicate price increase, red candles indicate price decrease
                - The body of the candle shows opening and closing prices
                - The wicks show the high and low prices during the period
                """)

                with col2:
                    st.markdown('<p class="sub-header">Volume Analysis</p>', unsafe_allow_html=True)
                    with st.expander("View Volume Chart", expanded=True):
                        st.plotly_chart(plot_volume_chart(data, ticker), use_container_width=True, config={'displayModeBar': False})
                        st.markdown("""
                    **Volume Chart Explanation:**
                    - Volume shows the number of shares traded
                    - High volume often indicates strong trend confirmation
                    - Low volume may suggest weak price movements
                    - Sudden volume spikes can indicate potential breakouts or reversals
                    """)
                col1, col2 = st.columns(2)

                
                st.markdown('<p class="sub-header">Technical Indicators</p>', unsafe_allow_html=True)
                tab1, tab2, tab3 = st.tabs(["Moving Averages", "RSI", "Bolliner Bands"])
                    
                with tab1:
                    st.plotly_chart(plot_moving_averages(data, ticker), use_container_width=True, config={'displayModeBar': False})
                    st.markdown("""
                    **Moving Averages Explanation:**
                    - The 5-day MA responds quickly to price changes
                    - The 20-day MA shows the longer-term trend
                    - When shorter MA crosses above longer MA, it's a bullish signal
                    - When shorter MA crosses below longer MA, it's a bearish signal
                    """)
                

                with tab2:
                    st.plotly_chart(plot_rsi(data, ticker), use_container_width=True, config={'displayModeBar': False})
                    st.markdown("""
                    **RSI Explanation:**
                    - RSI measures the speed and change of price movements
                    - RSI above 70 is considered overbought
                    - RSI below 30 is considered oversold
                    - Divergences between RSI and price can signal potential reversals
                    """)

                with tab3:
                    st.plotly_chart(plot_bollinger_bands(data, ticker), use_container_width=True, config={'displayModeBar': False})
                    st.markdown("""
                    **Bollinger Bands Explanation:**
                    - The middle band is a 20-day moving average
                    - Upper and lower bands are 2 standard deviations away from the middle band
                    - Price touching the upper band may indicate overbought conditions
                    - Price touching the lower band may indicate oversold conditions
                    - Bands contracting indicate low volatility, expanding indicate high volatility
                    """)
            

                with col1:
                    st.markdown('<p class="sub-header">Day Trading Insights</p>', unsafe_allow_html=True)
                    with st.expander("View Day Trading Analysis", expanded=True):
                        st.markdown(f"""
                    
                        {day_trading_analysis(data)}
                        
                        """, unsafe_allow_html=True)

                with col2:
                    st.markdown('<p class="sub-header">Risk Management</p>', unsafe_allow_html=True)
                    with st.expander("View Risk Management Analysis", expanded=True):
                        st.markdown(f"""
                        
                        {risk_management_analysis(data, info)}
                        
                        """, unsafe_allow_html=True)

                

                st.markdown('<p class="sub-header">Additional Real-time Insights</p>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A'):.2f}" if isinstance(info.get('trailingPE'), (int, float)) else "N/A")
                    st.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A")
                with col2:
                    st.metric("Short Ratio", f"{info.get('shortRatio', 'N/A'):.2f}" if isinstance(info.get('shortRatio'), (int, float)) else "N/A")
                    st.metric("Beta", f"{info.get('beta', 'N/A'):.2f}" if isinstance(info.get('beta'), (int, float)) else "N/A")
                with col3:
                    st.metric("Relative Volume", f"{info.get('volume', 0) / info.get('averageVolume', 1):.2f}" if info.get('volume') and info.get('averageVolume') else "N/A")
                    st.metric("52-Week Change", f"{info.get('52WeekChange', 0)*100:.2f}%" if info.get('52WeekChange') is not None else "N/A")

            else:
                st.warning("No data available for the selected ticker.")
        
        time.sleep(180)  

if __name__ == "__main__":
    real_time_stock_analysis()