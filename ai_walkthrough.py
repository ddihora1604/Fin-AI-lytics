import streamlit as st
from real_time import real_time_stock_analysis
from expense_tracking import expense_tracking_interface
from portfolio_management import portfolio_management_interface
from debt_management import debt_management_interface
from bs import BudgetSavingsManager
from chat_assistance import ChatAssistant

class AIWalkthroughAssistant:
    def __init__(self, user_id, user_data):
        self.user_id = user_id
        self.user_data = user_data
        self.keywords = {
            'stock': 'Real-time Stock Analysis',
            # 'expense': 'Expense Tracking',
            'portfolio': 'Portfolio Management',
            # 'debt': 'Debt Management',
            # 'budget': 'Budget & Savings',
            # 'save': 'Budget & Savings',
            'finance': 'AI Finance Manager',
            'assistant': 'AI Finance Manager',
            'chat': 'Financial Advice',
            'returns': 'Recommender',
            'expected': 'Recommender'
        }

    def process_user_input(self, user_input):
        user_input = user_input.lower()
        for keyword, feature in self.keywords.items():
            if keyword in user_input:
                return feature
        return None
    
    def navigate_to_feature(self, feature):
        if feature == 'Real-time Stock Analysis':
            real_time_stock_analysis()
        # elif feature == 'Expense Tracking':
        #     expense_tracking_interface(self.user_id).run()
        elif feature == 'Portfolio Management':
            portfolio_management_interface(self.user_id)
        # elif feature == 'Debt Management':
        #     debt_management_interface(self.user_id, self.user_data).run()
        # elif feature == 'Budget & Savings':
        #     BudgetSavingsManager().run()
        elif feature == 'Financial Advice':
            ChatAssistant().run()
        elif feature == 'AI Finance Manager':
            from ai_finance_manager import run_advanced_ai_finance_manager
            run_advanced_ai_finance_manager(self.user_id)
        elif feature == 'Recommender':
            from recommender import recommend_stocks
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
        else:
            st.warning("Feature not found.")
        

    def run(self):
        st.subheader("AI Walkthrough Assistant")
        user_input = st.text_input("Tell me what you're looking to do, and I'll guide you to the right feature:")
        
        if user_input:
            suggested_feature = self.process_user_input(user_input)
            if suggested_feature:
                st.success(f"Based on your input, I suggest you try the {suggested_feature} feature.")
                if st.button(f"Explore {suggested_feature}"):
                    self.navigate_to_feature(suggested_feature)
            else:
                st.info("I'm not sure which feature would be best for you. Feel free to explore the options or try rephrasing your request.")
        
        return None

# # Usage
# if __name__ == "__main__":
#     assistant = AIWalkthroughAssistant()
#     result = assistant.run()
#     if result:
#         st.write(f"Navigating to {result}...")