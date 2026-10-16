"""Simple email/password accounts plus a guest mode.

Passwords are salted and hashed with PBKDF2-SHA256; plain-text passwords are never stored.
Accounts live in a local JSON file. On Streamlit Community Cloud that file is reset whenever the
app restarts, so this is suitable for a demo, not for real user data.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import threading
from pathlib import Path

import streamlit as st

import styles

USERS_FILE = Path(__file__).parent / "users.json"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ITERATIONS = 200_000
_lock = threading.Lock()


def _load() -> dict:
    try:
        return json.loads(USERS_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), ITERATIONS).hex()


def register(name: str, email: str, password: str) -> str | None:
    """Create an account. Returns an error message, or None on success."""
    name, email = name.strip(), email.strip().lower()
    if not name:
        return "Please enter your name."
    if not EMAIL_RE.match(email):
        return "Please enter a valid email address."
    if len(password) < 8:
        return "Password must be at least 8 characters."
    with _lock:
        users = _load()
        if email in users:
            return "An account with this email already exists. Sign in instead."
        salt = secrets.token_hex(16)
        users[email] = {"name": name, "salt": salt, "hash": _hash(password, salt)}
        USERS_FILE.write_text(json.dumps(users))
    return None


def authenticate(email: str, password: str) -> dict | None:
    user = _load().get(email.strip().lower())
    if user and hmac.compare_digest(user["hash"], _hash(password, user["salt"])):
        return {"name": user["name"], "email": email.strip().lower()}
    return None


def current_user() -> dict | None:
    return st.session_state.get("user")


def sign_out() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def sign_in_page() -> None:
    left, right = st.columns([1.1, 1], gap="large", vertical_alignment="center")
    with left:
        styles.auth_panel()
    with right:
        tab_in, tab_up = st.tabs(["Sign in", "Create account"])
        with tab_in:
            st.html('<div class="auth-title">Welcome back</div><p class="auth-sub">Sign in to continue to your lab.</p>')
            with st.form("sign_in", border=False):
                email = st.text_input("Email", placeholder="you@example.com")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                if st.form_submit_button("Sign in", type="primary", width="stretch"):
                    user = authenticate(email, password)
                    if user:
                        st.session_state.user = user
                        st.rerun()
                    st.error("Incorrect email or password.")
        with tab_up:
            st.html('<div class="auth-title">Create your account</div><p class="auth-sub">It takes less than a minute.</p>')
            with st.form("sign_up", border=False):
                name = st.text_input("Full name", placeholder="Ada Lovelace")
                email = st.text_input("Email", placeholder="you@example.com", key="su_email")
                password = st.text_input("Password", type="password", placeholder="At least 8 characters", key="su_pw")
                if st.form_submit_button("Create account", type="primary", width="stretch"):
                    error = register(name, email, password)
                    if error:
                        st.error(error)
                    else:
                        st.session_state.user = {"name": name.strip(), "email": email.strip().lower()}
                        st.rerun()

        st.html('<p class="auth-sub" style="text-align:center;margin:14px 0 8px">or</p>')
        if st.button("Continue as guest", width="stretch", icon=":material/person:"):
            st.session_state.user = {"name": "Guest", "email": "Guest session"}
            st.rerun()
