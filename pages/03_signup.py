"""
Signup Page for PurePicks.COM
"""

import streamlit as st
from auth import AuthManager, init_session_state

st.set_page_config(page_title="Sign Up - PurePicks.COM", page_icon="✨", layout="centered")

# Custom CSS
st.markdown("""
<style>
    .signup-container {
        max-width: 400px;
        margin: 2rem auto;
        padding: 2rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

init_session_state()
auth_manager = AuthManager()

# Header
st.markdown("### ✨ Join PurePicks.COM")
st.markdown("Create your account and get started")
st.markdown("---")

# Signup Form
with st.form("signup_form"):
    username = st.text_input("Username", placeholder="Choose a username")
    email = st.text_input("Email", placeholder="your@email.com")
    password = st.text_input("Password", type="password", placeholder="Create a secure password")
    password_confirm = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
    
    col1, col2 = st.columns(2)
    with col1:
        submit = st.form_submit_button("Create Account", use_container_width=True, type="primary")
    with col2:
        if st.form_submit_button("Back to Home", use_container_width=True):
            st.switch_page("purepicks_app.py")
    
    if submit:
        # Validation
        if not username or not email or not password:
            st.error("Please fill in all fields")
        elif password != password_confirm:
            st.error("Passwords do not match")
        elif len(password) < 6:
            st.error("Password must be at least 6 characters")
        else:
            success, message = auth_manager.signup(username, email, password)
            if success:
                st.success(message)
                st.info("Please login to continue")
                st.balloons()
                if st.button("Go to Login"):
                    st.switch_page("pages/02_login.py")
            else:
                st.error(message)

st.markdown("---")
st.markdown("Already have an account?")
if st.button("Login Here", use_container_width=True):
    st.switch_page("pages/02_login.py")
