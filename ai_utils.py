import os
import replicate
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Initialize Replicate client
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

def generate_ai_response(prompt):
    try:
        output = replicate_client.run(
            "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            input={"prompt": prompt, "max_length": 200, "temperature": 0.3}
        )
        return ''.join(output)
    except Exception as e:
        st.error(f"Error generating AI response: {e}")
        return "Error generating response."