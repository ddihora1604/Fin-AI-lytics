import streamlit as st
from ai_utils import generate_ai_response
from financial_data import get_financial_news

def chat_interface():
    st.title('💬 AI Financial Assistant')
    user_input = st.text_input("Ask about finances, stocks, insurance, or request financial news:")
    if user_input:
        if "financial news" in user_input.lower():
            display_financial_news()
        else:
            response = generate_ai_response(user_input)
            st.write(response)

def display_financial_news():
    news_items = get_financial_news()
    st.write("Here are the latest financial news items:")
    for item in news_items:
        with st.expander(item['title']):
            st.write(f"Source: {item['source']}")
            st.write(f"Summary: {item['summary']}")
            st.write(f"URL: {item['url']}")

if __name__ == "__main__":
    chat_interface()