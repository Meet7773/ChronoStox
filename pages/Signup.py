import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="ChronoStox | Signup",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="collapsed",
)

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("Create your ChronoStox account")
st.caption("Sign up to start building your simulated portfolio.")

signup_error = st.empty()

with st.form("signup_form", clear_on_submit=False):
    username = st.text_input("Choose a username", placeholder="At least 3 characters")
    password = st.text_input("Password", type="password", placeholder="At least 6 characters")
    confirm_password = st.text_input("Confirm password", type="password")
    submitted = st.form_submit_button("Sign Up", use_container_width=True)

    if submitted:
        username_clean = username.strip()
        if not username_clean or not password or not confirm_password:
            signup_error.error("All fields are required.")
        elif len(username_clean) < 3:
            signup_error.error("Username must be at least 3 characters.")
        elif len(password) < 6:
            signup_error.error("Password must be at least 6 characters.")
        elif password != confirm_password:
            signup_error.error("Passwords do not match.")
        else:
            try:
                res = requests.post(
                    f"{API_URL}/auth/signup",
                    json={"userId": username_clean, "password": password},
                    timeout=10,
                )
                res.raise_for_status()
                st.success("Signup complete! Redirecting to dashboard...")
                st.session_state.auth_user = username_clean
                data = res.json()
                st.session_state.portfolio_cache = data.get("portfolio")
                st.session_state.simulation_cache = data.get("simulation")
                st.cache_data.clear()
                st.switch_page("pages/Dashboard.py")
            except requests.exceptions.RequestException as exc:
                detail = "Unable to create account."
                try:
                    if exc.response is not None:
                        detail = exc.response.json().get("detail", detail)
                except Exception:
                    pass
                signup_error.error(detail)

st.markdown("---")
st.caption("Already have an account?")
st.page_link("Login.py", label="Go to login", icon="🔐")

