import streamlit as st
import os
import replicate
import requests
from dotenv import load_dotenv

load_dotenv()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

class ChatAssistant:
    def __init__(self):
        self.replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

    def generate_ai_response(self, prompt):
        try:
            output = self.replicate_client.run(
                "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
                input={"prompt": prompt, "max_length": 200, "temperature": 0.7}
            )
            return ''.join(output)
        except Exception as e:
            st.error(f"Error generating AI response: {e}")
            return "Error generating response."

    def get_financial_news(self):
        try:
            url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={ALPHA_VANTAGE_API_KEY}'
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data.get('feed', [])[:5]  # Return the top 5 news items
        except requests.RequestException as e:
            st.error(f"Error fetching financial news: {e}")
            return []

    def run(self):
        st.title('💬 AI Financial Assistant')
        user_input = st.text_input("Ask about finances, stocks, insurance, or request financial news:")
        if user_input:
            if "financial news" in user_input.lower():
                news_items = self.get_financial_news()
                st.write("Here are the latest financial news items:")
                for item in news_items:
                    st.subheader(item['title'])
                    st.write(f"Source: {item['source']}")
                    st.write(f"Summary: {item['summary']}")
                    st.write(f"URL: {item['url']}")
                    st.write("---")
            else:
                response = self.generate_ai_response(user_input)
                st.write(response)