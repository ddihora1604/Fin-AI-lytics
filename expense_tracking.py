import streamlit as st
import plotly.express as px
import pandas as pd
import firebase_admin
from firebase_admin import firestore, initialize_app, credentials

# Initialize Firebase app
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate("firebase-adminsdk.json")
        initialize_app(cred)

init_firebase()

# Initialize Firestore client
def get_firestore_client():
    return firestore.client()

# Load user data from Firestore
def load_user_data(user_id):
    doc_ref = get_firestore_client().collection("user_data").document(user_id)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    else:
        return {}

# Save user data to Firestore
def save_user_data(user_id, data):
    doc_ref = get_firestore_client().collection("user_data").document(user_id)
    doc_ref.set(data)

# Plot expenses distribution
def plot_expenses(expenses):
    df = pd.DataFrame(expenses)
    fig = px.pie(df, values='amount', names='category', title='Expense Distribution')
    return fig

# Display expense summary
def display_expense_summary(expenses):
    df = pd.DataFrame(expenses)
    total_expenses = df['amount'].sum()
    st.subheader("Expense Summary")
    st.write(f"Total Expenses: ${total_expenses:.2f}")
    
    category_totals = df.groupby('category')['amount'].sum().sort_values(ascending=False)
    st.write("Top Expense Categories:")
    for category, amount in category_totals.items():
        st.write(f"{category}: ${amount:.2f} ({amount/total_expenses:.1%})")

# Main expense tracking interface
def expense_tracking_interface(user_id):
    st.title('💸 Expense Tracking')
    
    # Load user data
    user_data = load_user_data(user_id)
    expenses = user_data.get("expenses", [])
    
    # Input fields for new expense
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount", min_value=0.0)
    with col2:
        category = st.selectbox("Category", ["Food", "Transport", "Entertainment", "Utilities", "Other"])
    
    # Button to add expense
    if st.button("Add Expense"):
        expenses.append({"amount": amount, "category": category})
        user_data["expenses"] = expenses
        save_user_data(user_id, user_data)
        st.success("Expense added successfully!")
    
    # Plot and display expense summary if there are expenses
    if expenses:
        st.plotly_chart(plot_expenses(expenses))
        display_expense_summary(expenses)
