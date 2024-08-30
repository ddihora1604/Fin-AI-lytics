import streamlit as st
import requests
import os

# Firebase Authentication API endpoint
FIREBASE_AUTH_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=AIzaSyCWQRFrZaVPhD--tZs7_IHfdRrpU5PxZDM"
FIREBASE_SIGNUP_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=AIzaSyCWQRFrZaVPhD--tZs7_IHfdRrpU5PxZDM"

def sign_in(email, password):
    try:
        response = requests.post(FIREBASE_AUTH_URL, json={
            'email': email,
            'password': password,
            'returnSecureToken': True
        })
        response_data = response.json()
        if 'idToken' in response_data:
            return response_data['idToken']
        else:
            st.error(response_data.get('error', {}).get('message', 'Unknown error'))
            return None
    except Exception as e:
        st.error(f"Error during sign-in: {str(e)}")
        return None

def sign_up(email, password):
    try:
        response = requests.post(FIREBASE_SIGNUP_URL, json={
            'email': email,
            'password': password,
            'returnSecureToken': True
        })
        response_data = response.json()
        if 'idToken' in response_data:
            return response_data['idToken']
        else:
            st.error(response_data.get('error', {}).get('message', 'Unknown error'))
            return None
    except Exception as e:
        st.error(f"Error during sign-up: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Login", page_icon="🔑")

    st.title("Firebase Authentication with Email and Password")

    choice = st.sidebar.selectbox("Login/Signup", ["Login", "Sign Up"])

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if choice == "Login":
        if st.button("Login"):
            id_token = sign_in(email, password)
            if id_token:
                st.success("Logged in successfully!")
                # Directly call the main function of the finance_manager module
                import finance_manager
                finance_manager.main()  # Start the finance manager app

    elif choice == "Sign Up":
        if st.button("Sign Up"):
            id_token = sign_up(email, password)
            if id_token:
                st.success("Account created successfully!")
                # Directly call the main function of the finance_manager module
                import finance_manager
                finance_manager.main()  # Start the finance manager app

if __name__ == "__main__":
    main()