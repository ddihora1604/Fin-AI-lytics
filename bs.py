import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from firebase_admin import firestore

class BudgetSavingsManager:
    def __init__(self):
        self.db = firestore.client()
        self.budgets = self.load_budgets()
        self.expenses = self.load_expenses()

    def load_budgets(self):
        doc_ref = self.db.collection("budget_data").document("main")
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            return data.get('budgets', {})
        return {}

    def load_expenses(self):
        doc_ref = self.db.collection("expense_data").document("main")
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            return data.get('expenses', {})
        return {}

    def save_budgets(self):
        doc_ref = self.db.collection("budget_data").document("main")
        doc_ref.set({
            'budgets': self.budgets,
        }, merge=True)

    def save_expenses(self):
        doc_ref = self.db.collection("expense_data").document("main")
        doc_ref.set({
            'expenses': self.expenses,
        }, merge=True)

    def set_budget(self, category, amount):
        self.budgets[category] = amount
        self.save_budgets()
        return f"Budget set for {category}: ${amount:.2f}"

    def track_expense(self, category, amount, date):
        if category not in self.expenses:
            self.expenses[category] = []
        # Convert date to datetime.datetime
        if isinstance(date, datetime):
            date = datetime(date.year, date.month, date.day)
        self.expenses[category].append({"amount": amount, "date": date})
        self.save_expenses()
        return f"Tracked expense in {category}: ${amount:.2f} on {date.strftime('%Y-%m-%d')}"

    def get_budget_status(self, category):
        budget = self.budgets.get(category, 0)
        expenses = self.expenses.get(category, [])
        spent = sum(expense["amount"] for expense in expenses)
        remaining = budget - spent
        return {
            "budget": budget,
            "spent": spent,
            "remaining": remaining,
            "percentage": (spent / budget * 100) if budget > 0 else 0
        }

    def visualize_budget(self):
        categories = list(self.budgets.keys())
        budgets = [self.budgets[cat] for cat in categories]
        expenses = [sum(expense["amount"] for expense in self.expenses.get(cat, []))
                    for cat in categories]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(categories))
        ax.bar([i - 0.2 for i in x], budgets, 0.4, label='Budget', color='blue')
        ax.bar([i + 0.2 for i in x], expenses, 0.4, label='Expenses', color='red')
        ax.set_ylabel('Amount ($)')
        ax.set_title('Budget vs Expenses')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        return fig
    
    def run(self):
        st.title("Budget & Expense Manager")

        # Initialize session state
        if 'manager' not in st.session_state:
            st.session_state.manager = BudgetSavingsManager()

        # Sidebar for actions
        action = st.sidebar.selectbox(
            "Choose an action",
            ["Set Budget", "Track Expense", "View Budget Status", "Visualize Budget"]
        )

        if action == "Set Budget":
            st.header("Set Budget")
            category = st.text_input("Enter budget category:")
            amount = st.number_input("Enter budget amount:", min_value=0.0, format="%.2f")
            if st.button("Set Budget"):
                result = st.session_state.manager.set_budget(category, amount)
                st.success(result)

        elif action == "Track Expense":
            st.header("Track Expense")
            category = st.text_input("Enter expense category:")
            amount = st.number_input("Enter expense amount:", min_value=0.0, format="%.2f")
            date = st.date_input("Enter expense date:")
            if st.button("Track Expense"):
                result = st.session_state.manager.track_expense(category, amount, datetime(date.year, date.month, date.day))
                st.success(result)

        elif action == "View Budget Status":
            st.header("Budget Status")
            categories = list(st.session_state.manager.budgets.keys())
            if categories:
                category = st.selectbox("Select budget category:", categories)
                status = st.session_state.manager.get_budget_status(category)
                st.write(f"Budget: ${status['budget']:.2f}")
                st.write(f"Spent: ${status['spent']:.2f}")
                st.write(f"Remaining: ${status['remaining']:.2f}")
                st.write(f"Percentage Used: {status['percentage']:.2f}%")
            else:
                st.write("No budgets set yet. Please set a budget first.")

        elif action == "Visualize Budget":
            st.header("Budget Visualization")
            if st.session_state.manager.budgets:
                fig = st.session_state.manager.visualize_budget()
                st.pyplot(fig)
            else:
                st.write("No data to visualize. Please set budgets and track expenses first.")
