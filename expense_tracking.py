import streamlit as st
import plotly.express as px
from user_data import load_user_data, save_user_data
import pandas as pd

def expense_tracking_interface(user_id):
    st.title('💸 Expense Tracking')
    
    user_data = load_user_data(user_id)
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
        st.plotly_chart(plot_expenses(expenses))
        display_expense_summary(expenses)

def plot_expenses(expenses):
    df = pd.DataFrame(expenses)
    fig = px.pie(df, values='amount', names='category', title='Expense Distribution')
    return fig

def display_expense_summary(expenses):
    df = pd.DataFrame(expenses)
    total_expenses = df['amount'].sum()
    st.subheader("Expense Summary")
    st.write(f"Total Expenses: ${total_expenses:.2f}")
    
    category_totals = df.groupby('category')['amount'].sum().sort_values(ascending=False)
    st.write("Top Expense Categories:")
    for category, amount in category_totals.items():
        st.write(f"{category}: ${amount:.2f} ({amount/total_expenses:.1%})")