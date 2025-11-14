import streamlit as st


def require_login():
    """Ensure a user is logged in and return the user id."""
    if "auth_user" not in st.session_state or not st.session_state.auth_user:
        st.switch_page("Login.py")
    return st.session_state.auth_user


def require_login_with_redirect(redirect_page: str = "Login.py") -> str:
    """Variant that allows custom redirect page."""
    if "auth_user" not in st.session_state or not st.session_state.auth_user:
        st.switch_page(redirect_page)
    return st.session_state.auth_user

