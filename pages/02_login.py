"""
Login Page for PurePicks.COM
"""

import streamlit as st
from auth import AuthManager, init_session_state

st.set_page_config(page_title="Login - PurePicks.COM", page_icon="🔐", layout="centered")

# Custom CSS
st.markdown("""
<style>
    .login-container {
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
st.markdown("### 🔐 Login to PurePicks.COM")
st.markdown("---")

# Login Form
with st.form("login_form"):
    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")
    
    col1, col2 = st.columns(2)
    with col1:
        submit = st.form_submit_button("Login", use_container_width=True, type="primary")
    with col2:
        back_home = st.form_submit_button("Back to Home", use_container_width=True)
    
    if back_home:
        st.switch_page("purepicks_app.py")
    
    if submit:
        if not username or not password:
            st.error("Please enter both username and password")
        else:
            success, user_data = auth_manager.login(username, password)
            if success:
                st.session_state.authenticated = True
                st.session_state.user_data = user_data
                st.success(f"Welcome back, {user_data['username']}!")
                st.info("Redirecting to dashboard...")
                st.switch_page("pages/04_dashboard.py")
            else:
                st.error("Invalid username or password")

st.markdown("---")
st.markdown("Don't have an account?")
if st.button("Sign Up Here", use_container_width=True):
    st.switch_page("pages/03_signup.py")
