import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from firebase_admin import firestore
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from deap import creator, base, tools, algorithms

class EnhancedDebtManagement:
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

    def optimize_debt_payoff(self, total_monthly_payment):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", np.random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(self.debts))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evaluate(individual):
            payments = np.array(individual) / sum(individual) * total_monthly_payment
            total_interest = 0
            for debt, payment in zip(self.debts, payments):
                balance = debt['amount']
                while balance > 0:
                    interest = balance * (debt['interest_rate'] / 100 / 12)
                    total_interest += interest
                    balance += interest - payment
            return (total_interest,)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=50)
        algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=50, verbose=False)

        best_individual = tools.selBest(population, k=1)[0]
        optimized_payments = np.array(best_individual) / sum(best_individual) * total_monthly_payment
        return optimized_payments

    def get_personalized_strategy(self):
        if not self.debts:
            return "No debt data available for personalized strategy."

        df = pd.DataFrame(self.debts)
        features = df[['amount', 'interest_rate', 'minimum_payment']].values
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=3, random_state=42)
        df['cluster'] = kmeans.fit_predict(features_scaled)

        strategies = {
            0: "Focus on high-interest debts",
            1: "Balanced approach",
            2: "Snowball method (smallest debts first)"
        }

        dominant_cluster = df['cluster'].mode().values[0]
        return strategies[dominant_cluster]

    def assess_debt_risk(self):
        if not self.debts:
            return "No debt data available for risk assessment."

        df = pd.DataFrame(self.debts)
        features = df[['amount', 'interest_rate', 'minimum_payment']].values
        total_debt = df['amount'].sum()
        df['debt_to_income'] = df['amount'] / total_debt

        X = np.column_stack((features, df['debt_to_income'].values))

        # Simulate historical data for training
        np.random.seed(42)
        y = np.random.choice([0, 1], size=len(X))  # 0: low risk, 1: high risk

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)

        risk_probabilities = clf.predict_proba(X)[:, 1]
        df['risk_score'] = risk_probabilities * 100

        return df[['name', 'risk_score']].sort_values('risk_score', ascending=False)

    def suggest_payment_allocation(self, total_monthly_payment):
        optimized_payments = self.optimize_debt_payoff(total_monthly_payment)
        allocation = []
        for debt, payment in zip(self.debts, optimized_payments):
            allocation.append({
                'name': debt['name'],
                'suggested_payment': payment,
                'percentage': (payment / total_monthly_payment) * 100
            })
        return sorted(allocation, key=lambda x: x['suggested_payment'], reverse=True)

    def run(self):
        st.title('🏦 AI-Enhanced Debt Management Dashboard')
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

        # AI-powered features
        st.subheader("AI-Powered Insights")

        # Personalized debt reduction strategy
        strategy = self.get_personalized_strategy()
        st.write(f"Recommended Strategy: {strategy}")

        # Debt risk assessment
        st.write("Debt Risk Assessment:")
        risk_assessment = self.assess_debt_risk()
        st.dataframe(risk_assessment)

        # Optimized payment allocation
        st.write("Optimized Payment Allocation:")
        total_monthly_payment = st.number_input("Enter your total monthly payment budget ($)", min_value=0.0, step=100.0)
        if st.button("Calculate Optimal Allocation"):
            allocation = self.suggest_payment_allocation(total_monthly_payment)
            for item in allocation:
                st.write(f"{item['name']}: ${item['suggested_payment']:.2f} ({item['percentage']:.2f}%)")

        st.write("")  # Add space

        # Calculate payoff time for each debt
        if self.debts:
            st.subheader("Estimated Payoff Time")
            for debt in self.debts:
                months = self.calculate_payoff_time(debt, debt['minimum_payment'])
                payoff_date = (datetime.now() + timedelta(days=30*months)).strftime("%B %Y")
                st.write(f"{debt['name']}: {months} months (Estimated payoff: {payoff_date})")

def enhanced_debt_management_interface(user_id, db):
    debt_manager = EnhancedDebtManagement(user_id, db)
    debt_manager.run()