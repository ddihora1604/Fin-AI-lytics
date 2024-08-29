import streamlit as st

def authenticate_user():
    st.sidebar.title("User Authentication")
    user_id = st.sidebar.text_input("Enter your user ID:")
    if not user_id:
        st.warning("Please enter a user ID to continue.")
        return None
    return user_id