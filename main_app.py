import streamlit as st
from chat import chat_interface
from stock_analysis import stock_analysis_interface
from expense_tracking import expense_tracking_interface
from portfolio_management import portfolio_management_interface
from user_auth import authenticate_user

def main():
    st.set_page_config(page_title="AI-Driven Personal Finance Manager", page_icon="💰", layout="wide")

    # Sidebar
    with st.sidebar:
        st.title('🤖 AI Finance Assistant')
        selected = st.selectbox(
            "Choose a feature",
            ["Chat", "Stock Analysis", "Expense Tracking", "Portfolio Management"]
        )

    # User Authentication
    user_id = authenticate_user()
    if not user_id:
        return

    # Main content
    if selected == "Chat":
        chat_interface()
    elif selected == "Stock Analysis":
        stock_analysis_interface()
    elif selected == "Expense Tracking":
        expense_tracking_interface(user_id)
    elif selected == "Portfolio Management":
        portfolio_management_interface(user_id)

if __name__ == "__main__":
    main()