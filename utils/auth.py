import streamlit as st


def require_login():
    """Ensure a user is logged in and return the user id."""
    # Check session state first
    if "auth_user" in st.session_state and st.session_state.auth_user:
        st.query_params["user"] = st.session_state.auth_user
        return st.session_state.auth_user
    
    # Try to restore from query params (for page refreshes)
    query_params = st.query_params
    if "user" in query_params:
        user_param = query_params["user"]
        if isinstance(user_param, (list, tuple)):
            user_id = user_param[0]
        else:
            user_id = user_param
        st.session_state.auth_user = user_id
        st.query_params["user"] = user_id
        return user_id
    
    # Not logged in - redirect to login
    st.switch_page("Login.py")


def require_login_with_redirect(redirect_page: str = "Login.py") -> str:
    """Variant that allows custom redirect page."""
    if "auth_user" in st.session_state and st.session_state.auth_user:
        st.query_params["user"] = st.session_state.auth_user
        return st.session_state.auth_user
    
    query_params = st.query_params
    if "user" in query_params:
        user_param = query_params["user"]
        if isinstance(user_param, (list, tuple)):
            user_id = user_param[0]
        else:
            user_id = user_param
        st.session_state.auth_user = user_id
        st.query_params["user"] = user_id
        return user_id
    
    st.switch_page(redirect_page)

