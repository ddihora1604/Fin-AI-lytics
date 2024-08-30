import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from firebase_admin import firestore

class DebtManagement:
    def __init__(self, user_id, db):
        self.user_id = user_id
        self.db = db
        self.debts = self.load_debts()

    def load_debts(self):
        doc_ref = self.db.collection("user_data").document(self.user_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict().get("debts", [])
        else:
            return []

    def save_debts(self):
        doc_ref = self.db.collection("user_data").document(self.user_id)
        doc_ref.set({"debts": self.debts}, merge=True)

    def add_debt(self, name, amount, interest_rate, minimum_payment):
        new_debt = {
            "name": name,
            "amount": amount,
            "interest_rate": interest_rate,
            "minimum_payment": minimum_payment,
            "added_date": datetime.now().strftime("%Y-%m-%d")
        }
        self.debts.append(new_debt)
        self.save_debts()

    def update_debt(self, index, name, amount, interest_rate, minimum_payment):
        self.debts[index].update({
            "name": name,
            "amount": amount,
            "interest_rate": interest_rate,
            "minimum_payment": minimum_payment
        })
        self.save_debts()

    def delete_debt(self, index):
        del self.debts[index]
        self.save_debts()

    def display_debt_chart(self):
        if not self.debts:
            st.write("No debt data available.")
            return
        df = pd.DataFrame(self.debts)
        fig = px.pie(df, values='amount', names='name', title='Debt Distribution')
        st.plotly_chart(fig)

    def display_debt_table(self):
        if not self.debts:
            st.write("No debt data available.")
            return
        df = pd.DataFrame(self.debts)
        st.dataframe(df)

    def calculate_total_debt(self):
        return sum(debt['amount'] for debt in self.debts)

    def calculate_payoff_time(self, debt, monthly_payment):
        balance = debt['amount']
        months = 0
        while balance > 0:
            interest = balance * (debt['interest_rate'] / 100 / 12)
            balance += interest
            balance -= monthly_payment
            months += 1
        return months
    def display_payoff_comparison(self):
        if not self.debts:
            st.write("No debt data available.")
            return
        
        strategies = ['Minimum Payments', 'Highest Interest First', 'Snowball Method']
        total_payments = []
        payoff_times = []

        for strategy in strategies:
            total_payment, payoff_time = self.simulate_payoff_strategy(strategy)
            total_payments.append(total_payment)
            payoff_times.append(payoff_time)

        fig = go.Figure(data=[
            go.Bar(name='Total Payments', x=strategies, y=total_payments),
            go.Bar(name='Payoff Time (months)', x=strategies, y=payoff_times)
        ])
        fig.update_layout(barmode='group', title='Debt Payoff Strategy Comparison')
        st.plotly_chart(fig)

    def simulate_payoff_strategy(self, strategy):
        debts = self.debts.copy()
        total_payment = 0
        months = 0
        
        while debts:
            months += 1
            for debt in debts:
                interest = debt['amount'] * (debt['interest_rate'] / 100 / 12)
                debt['amount'] += interest
                
                if strategy == 'Minimum Payments':
                    payment = min(debt['minimum_payment'], debt['amount'])
                elif strategy == 'Highest Interest First':
                    payment = max(debt['minimum_payment'], min(500, debt['amount']))  # Assuming $500 extra payment
                elif strategy == 'Snowball Method':
                    payment = max(debt['minimum_payment'], min(500, debt['amount']))  # Assuming $500 extra payment
                
                debt['amount'] -= payment
                total_payment += payment
            
            debts = [debt for debt in debts if debt['amount'] > 0]
            
            if strategy == 'Highest Interest First':
                debts.sort(key=lambda x: x['interest_rate'], reverse=True)
            elif strategy == 'Snowball Method':
                debts.sort(key=lambda x: x['amount'])

        return total_payment, months

    def run(self):
        st.title('🏦 Debt Management Dashboard')
        st.write("")  # Add space

        # Add new debt
        with st.expander("Add New Debt"):
            debt_name = st.text_input("Debt Name")
            debt_amount = st.number_input("Debt Amount ($)", min_value=0.0, step=100.0)
            interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
            minimum_payment = st.number_input("Minimum Monthly Payment ($)", min_value=0.0, step=10.0)
            if st.button("Add Debt"):
                self.add_debt(debt_name, debt_amount, interest_rate, minimum_payment)
                st.success(f"Added {debt_name} to your debt list.")

        st.write("")  # Add space

        # Display total debt
        total_debt = self.calculate_total_debt()
        st.metric("Total Debt", f"${total_debt:.2f}")

        st.write("")  # Add space

        # Display debt chart
        st.subheader("Debt Distribution")
        self.display_debt_chart()

        st.write("")  # Add space

        # Display debt table
        st.subheader("Debt Details")
        self.display_debt_table()

        st.write("")  # Add space

        # Edit or delete debts
        if self.debts:
            st.subheader("Edit or Delete Debts")
            debt_index = st.selectbox("Select a debt to edit or delete", range(len(self.debts)), format_func=lambda i: self.debts[i]['name'])
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Edit Debt"):
                    debt = self.debts[debt_index]
                    with st.form("edit_debt_form"):
                        edit_name = st.text_input("Debt Name", value=debt['name'])
                        edit_amount = st.number_input("Debt Amount ($)", value=debt['amount'], min_value=0.0, step=100.0)
                        edit_interest = st.number_input("Interest Rate (%)", value=debt['interest_rate'], min_value=0.0, max_value=100.0, step=0.1)
                        edit_minimum = st.number_input("Minimum Monthly Payment ($)", value=debt['minimum_payment'], min_value=0.0, step=10.0)
                        if st.form_submit_button("Update Debt"):
                            self.update_debt(debt_index, edit_name, edit_amount, edit_interest, edit_minimum)
                            st.success(f"Updated {edit_name} in your debt list.")

            with col2:
                if st.button("Delete Debt"):
                    self.delete_debt(debt_index)
                    st.success(f"Deleted {self.debts[debt_index]['name']} from your debt list.")

        st.write("")  # Add space

        # Debt payoff strategy comparison
        st.subheader("Debt Payoff Strategy Comparison")
        self.display_payoff_comparison()

        st.write("")  # Add space

        # Calculate payoff time for each debt
        if self.debts:
            st.subheader("Estimated Payoff Time")
            for debt in self.debts:
                months = self.calculate_payoff_time(debt, debt['minimum_payment'])
                payoff_date = (datetime.now() + timedelta(days=30*months)).strftime("%B %Y")
                st.write(f"{debt['name']}: {months} months (Estimated payoff: {payoff_date})")

        st.write("")  # Add space

        # Basic debt payoff strategy
        st.subheader("Debt Payoff Strategy")
        st.write("Consider paying off high-interest debts first while maintaining minimum payments on others.")
        if self.debts:
            highest_interest_debt = max(self.debts, key=lambda x: x['interest_rate'])
            st.write(f"Focus on paying off {highest_interest_debt['name']} first, as it has the highest interest rate of {highest_interest_debt['interest_rate']}%.")

def debt_management_interface(user_id, db):
    debt_manager = DebtManagement(user_id, db)
    debt_manager.run()