import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="ChronoStox | Login",
    page_icon="🔐",
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

st.title("Welcome to ChronoStox")
st.caption("Sign in to access your trading dashboards.")

if st.session_state.get("auth_user"):
    st.success(f"You're already signed in as {st.session_state.auth_user}.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go to Dashboard", use_container_width=True):
            st.query_params["user"] = st.session_state.auth_user
            st.switch_page("pages/Dashboard.py")
    with col2:
        if st.button("Log out", use_container_width=True):
            for key in list(st.session_state.keys()):
                st.session_state.pop(key, None)
            st.query_params.clear()
            st.cache_data.clear()
            st.experimental_rerun()
    st.stop()

login_error = st.empty()

with st.form("login_form", clear_on_submit=False):
    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")
    submitted = st.form_submit_button("Log In", use_container_width=True)

    if submitted:
        if not username or not password:
            login_error.error("Please enter both username and password.")
        else:
            try:
                res = requests.post(
                    f"{API_URL}/auth/login",
                    json={"userId": username.strip(), "password": password},
                    timeout=10,
                )
                res.raise_for_status()
                data = res.json()
                user_id = username.strip()
                st.session_state.auth_user = user_id
                st.session_state.portfolio_cache = data.get("portfolio")
                st.session_state.simulation_cache = data.get("simulation")
                st.cache_data.clear()
                st.success("Login successful! Redirecting...")
                st.query_params["user"] = user_id
                st.switch_page("pages/Dashboard.py")
            except requests.exceptions.RequestException as exc:
                detail = "Unable to connect to server."
                try:
                    if exc.response is not None:
                        detail = exc.response.json().get("detail", detail)
                except Exception:
                    pass
                login_error.error(detail)

st.markdown("---")
st.caption("Don't have an account?")
st.page_link("pages/Signup.py", label="Create an account", icon="📝")

