import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import firestore, initialize_app, credentials

def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate("firebase-adminsdk.json")
        initialize_app(cred)

init_firebase()
db = firestore.client()
try:

    class ai_finance_manager:
        def __init__(self, user_id):
            self.user_id = user_id
            self.expenses = []
            self.income = []
            self.savings_goal = 0
            self.debts = []
            self.load_data()

        def load_data(self):
            doc_ref = db.collection("user_data").document(self.user_id)
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                self.debts = data.get("debts", [])
                self.expenses = data.get("expenses", [])
                self.income = data.get("income", [])
                self.date = data.get("added_date", [])
                st.write("Loaded Debts:", self.debts)
                st.write("Loaded Expenses:", self.expenses)
                st.write("Loaded Income:", self.income)
            else:
                st.write("No data found for user.")

        def save_data(self):
            doc_ref = db.collection("user_data").document(self.user_id)
            doc_ref.set({
                "debts": self.debts,
                "expenses": self.expenses,
                "income": self.income
            }, merge=True)

        def add_transaction(self, amount, category, date, transaction_type):
            if isinstance(date, datetime):
                date = date.date()

            transaction = {"amount": amount, "category": category, "date": date.isoformat()}
            if transaction_type == "Expense":
                self.expenses.append(transaction)
            else:
                self.income.append(transaction)
            self.save_data()

        def add_debt(self, amount, interest_rate, description, minimum_payment):
            debt = {
                "amount": amount,
                "interest_rate": interest_rate,
                "minimum_payment": minimum_payment
            }
            self.debts.append(debt)
            self.save_data()

        def set_savings_goal(self, amount):
            self.savings_goal = amount

        def get_transaction_df(self):
            expenses_df = pd.DataFrame(self.expenses)
            income_df = pd.DataFrame(self.income)
            if not expenses_df.empty and not income_df.empty:
                expenses_df['type'] = 'Expense'
                income_df['type'] = 'Income'
                return pd.concat([expenses_df, income_df]).reset_index(drop=True)
            elif not expenses_df.empty:
                expenses_df['type'] = 'Expense'
                return expenses_df
            elif not income_df.empty:
                income_df['type'] = 'Income'
                return income_df
            else:
                return pd.DataFrame()

        def analyze_spending_patterns(self):
            df = pd.DataFrame(self.expenses)
            if df.empty or len(df) < 5:
                st.write("Not enough data to analyze spending patterns.")
                return

            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['week_of_month'] = df['date'].dt.day.apply(lambda x: (x - 1) // 7 + 1)

            X = df[['amount', 'day_of_week', 'week_of_month']]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            kmeans = KMeans(n_clusters=3, random_state=42)
            df['cluster'] = kmeans.fit_predict(X_scaled)

            st.subheader("Spending Patterns")
            for cluster in range(3):
                cluster_data = df[df['cluster'] == cluster]
                avg_amount = cluster_data['amount'].mean()
                common_day = cluster_data['day_of_week'].mode().iloc[0]
                common_week = cluster_data['week_of_month'].mode().iloc[0]
                
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                st.write(f"Cluster {cluster + 1}: Average spend ${avg_amount:.2f}, "
                        f"typically on {days[common_day]}s in week {common_week} of the month")

        def detect_unusual_transactions(self):
            df = pd.DataFrame(self.expenses)
            if df.empty or len(df) < 10:
                return "Not enough data to detect unusual transactions."

            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            df['rolling_mean'] = df['amount'].rolling(window=5, min_periods=1).mean()
            df['rolling_std'] = df['amount'].rolling(window=5, min_periods=1).std()

            df['unusual'] = (df['amount'] - df['rolling_mean']).abs() > 3 * df['rolling_std']

            df['pct_change'] = df['amount'].pct_change()
            df.loc[df['pct_change'] > 1, 'unusual'] = True

            unusual_transactions = df[df['unusual']]
            if unusual_transactions.empty:
                return "No unusual transactions detected."
            else:
                return unusual_transactions[['date', 'category', 'amount']].to_dict('records')

        def predict_monthly_savings(self):
            income_df = pd.DataFrame(self.income)
            expense_df = pd.DataFrame(self.expenses)
            
            if income_df.empty or expense_df.empty:
                return "Not enough data to predict monthly savings."

            income_df['date'] = pd.to_datetime(income_df['date'])
            expense_df['date'] = pd.to_datetime(expense_df['date'])

            last_month = datetime.now().replace(day=1) - timedelta(days=1)
            last_month_start = last_month.replace(day=1)

            monthly_income = income_df[(income_df['date'] >= last_month_start) & (income_df['date'] <= last_month)]['amount'].sum()
            monthly_expenses = expense_df[(expense_df['date'] >= last_month_start) & (expense_df['date'] <= last_month)]['amount'].sum()

            predicted_savings = monthly_income - monthly_expenses
            return predicted_savings

        def recommend_budget_adjustments(self):
            predicted_savings = self.predict_monthly_savings()
            if isinstance(predicted_savings, str):
                return predicted_savings

            if predicted_savings < self.savings_goal:
                shortfall = self.savings_goal - predicted_savings
                expense_df = pd.DataFrame(self.expenses)
                if not expense_df.empty:
                    top_categories = expense_df.groupby('category')['amount'].sum().nlargest(3)
                    recommendations = [f"Consider reducing spending in these categories:"]
                    for category, amount in top_categories.items():
                        suggested_reduction = min(shortfall, amount * 0.1)
                        recommendations.append(f"- {category}: reduce by ${suggested_reduction:.2f}")
                        shortfall -= suggested_reduction
                        if shortfall <= 0:
                            break
                    return "\n".join(recommendations)
                else:
                    return "Not enough expense data to make recommendations."
            else:
                return f"Great job! You're on track to meet or exceed your savings goal by ${predicted_savings - self.savings_goal:.2f}"

        def visualize_cash_flow(self):
            df = self.get_transaction_df()
            if df.empty:
                return None
            if 'date' in df.columns:

                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                df['cumulative'] = df.apply(lambda row: row['amount'] if row['type'] == 'Income' else -row['amount'], axis=1).cumsum()

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['date'], y=df['cumulative'], mode='lines', name='Cash Flow'))
                fig.add_trace(go.Bar(x=df['date'], y=df.apply(lambda row: row['amount'] if row['type'] == 'Income' else 0, axis=1), name='Income'))
                fig.add_trace(go.Bar(x=df['date'], y=df.apply(lambda row: -row['amount'] if row['type'] == 'Expense' else 0, axis=1), name='Expense'))

                fig.update_layout(
                    title='Cash Flow Over Time',
                    xaxis_title='Date',
                    yaxis_title='Amount ($)',
                    barmode='relative',
                    hovermode='x unified'
                )
                return fig

        def apply_extra_payment(self, extra_payment):
            if not self.debts:
                return "No debts to pay off."
            
            # Sort debts by highest interest rate first
            sorted_debts = sorted(self.debts, key=lambda d: d['interest_rate'], reverse=True)
            
            for debt in sorted_debts:
                if extra_payment <= 0:
                    break
                debt_amount = debt['amount']
                if extra_payment >= debt_amount:
                    extra_payment -= debt_amount
                    debt['amount'] = 0
                else:
                    debt['amount'] -= extra_payment
                    extra_payment = 0
            
            self.save_data()

        def manage_debt(self):
            st.subheader("Debt Management")
            
            if not self.debts:
                st.write("No debts recorded.")
                return

            total_debt = sum(debt.get('amount', 0) for debt in self.debts)
            st.write(f"Current total debt: ${total_debt:.2f}")

            st.write("You can reduce your debt by making extra payments. We'll apply payments to the highest interest rate debt first.")
            
            extra_payment = st.number_input("Extra payment amount", min_value=0.0, step=10.0)
            if st.button("Make Extra Payment"):
                result = self.apply_extra_payment(extra_payment)
                st.success(result)
                
                # Recalculate total debt after payment
                new_total_debt = sum(debt.get('amount', 0) for debt in self.debts)
                st.write(f"New total debt: ${new_total_debt:.2f}")
                st.write(f"Total debt reduction: ${total_debt - new_total_debt:.2f}")

            st.subheader("Debt Breakdown")
            for debt in self.debts:
                try:
                    amount = debt.get('amount', 0)
                    interest_rate = debt.get('interest_rate', 0)
                    st.write(f" ${amount:.2f} at {interest_rate}% interest")
                except KeyError as e:
                    st.write(f"Missing key: {e} in debt entry: {debt}")

            
        def visualize_debt_breakdown(self):
            if not self.debts:
                return None

            debt_df = pd.DataFrame(self.debts)

        def calculate_debt_payoff_time(self, extra_payment=0):
            if not self.debts:
                return "No debts recorded."

            payoff_times = []
            for debt in self.debts:
                balance = debt['amount']
                rate = debt['interest_rate'] / 100 / 12  # Monthly interest rate
                payment = debt['minimum_payment'] + extra_payment

                months = 0
                while balance > 0:
                    interest = balance * rate
                    balance += interest - payment
                    months += 1

                payoff_times.append({
                    'months': months,
                    'years': months / 12
                })

            return payoff_times

        def visualize_debt_payoff_timeline(self, extra_payment=0):
            payoff_times = self.calculate_debt_payoff_time(extra_payment)
            if isinstance(payoff_times, str):
                return None

            df = pd.DataFrame(payoff_times)
    def run_advanced_ai_finance_manager(user_id):
        st.title('🧠💰 Premium AI Finance Manager')

        if 'manager' not in st.session_state:
                st.session_state.manager = ai_finance_manager(user_id=user_id)
        
        if 'manager' in st.session_state:
            tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Transactions", "Debt Management", "Insights"])

            with tab1:
                st.header("Financial Dashboard")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Cash Flow")
                    cash_flow_fig = st.session_state.manager.visualize_cash_flow()
                    if cash_flow_fig:
                        st.plotly_chart(cash_flow_fig, use_container_width=True)
                    else:
                        st.write("Not enough data to visualize cash flow.")

                with col2:
                    st.subheader("Debt Breakdown")
                    # Add your code here to visualize the debt breakdown.
                    # Example placeholder:
                    debt_breakdown_fig = st.session_state.manager.visualize_debt_breakdown() if hasattr(st.session_state.manager, 'visualize_debt_breakdown') else None
                    if debt_breakdown_fig:
                        st.plotly_chart(debt_breakdown_fig, use_container_width=True)
                    else:
                        st.write("No debt data available.")

                st.subheader("Past Transactions")
                transactions_df = st.session_state.manager.get_transaction_df()
                if not transactions_df.empty:
                    st.write(transactions_df)
                else:
                    st.write("No transactions available.")

            with tab2:
                st.header("Add Transaction")
                col1, col2 = st.columns(2)

                with col1:
                    amount = st.number_input("Amount", min_value=0.0, step=0.01)
                    category = st.selectbox("Category", ["Salary", "Food", "Transport", "Entertainment", "Utilities", "Other"])

                with col2:
                    date = st.date_input("Date")
                    transaction_type = st.radio("Transaction Type", ["Income", "Expense"])

                if st.button("Add Transaction"):
                    st.session_state.manager.add_transaction(amount, category, date, transaction_type)
                    st.success("Transaction added!")

                st.header("Set Savings Goal")
                savings_goal = st.number_input("Monthly Savings Goal", min_value=0.0, step=10.0)
                if st.button("Set Goal"):
                    st.session_state.manager.set_savings_goal(savings_goal)
                    st.success("Savings goal set!")

            with tab3:
                st.header("Debt Management")
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Add Debt")
                    debt_amount = st.number_input("Debt Amount", min_value=0.0, step=100.0)
                    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1)

                with col2:
                    debt_description = st.text_input("Debt Description")
                    minimum_payment = st.number_input("Minimum Monthly Payment", min_value=0.0, step=10.0)

                if st.button("Add Debt"):
                    st.session_state.manager.add_debt(debt_amount, interest_rate, debt_description, minimum_payment)
                    st.success("Debt added!")

                st.session_state.manager.manage_debt()

                st.subheader("Debt Payoff Timeline")
                extra_payment = st.slider("Extra Monthly Payment", min_value=0, max_value=1000, step=50)
                payoff_timeline_fig = st.session_state.manager.visualize_debt_payoff_timeline(extra_payment) if hasattr(st.session_state.manager, 'visualize_debt_payoff_timeline') else None
                if payoff_timeline_fig:
                    st.plotly_chart(payoff_timeline_fig, use_container_width=True)
                else:
                    st.write("No debt data available.")

            with tab4:
                st.header("AI-Powered Insights")
                st.session_state.manager.analyze_spending_patterns()

                st.subheader("Unusual Transactions")
                unusual_transactions = st.session_state.manager.detect_unusual_transactions()
                if isinstance(unusual_transactions, str):
                    st.write(unusual_transactions)
                else:
                    st.table(pd.DataFrame(unusual_transactions))

                st.subheader("Monthly Savings Prediction")
                predicted_savings = st.session_state.manager.predict_monthly_savings()
                if isinstance(predicted_savings, str):
                    st.write(predicted_savings)
                else:
                    st.write(f"Predicted monthly savings: ${predicted_savings:.2f}")

                st.subheader("Budget Recommendations")
                recommendations = st.session_state.manager.recommend_budget_adjustments()
                st.write(recommendations)

    if __name__ == "__main__":
        pass
except ValueError:
    pass