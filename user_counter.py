"""Simple no-database visitor/session counter for Streamlit.

This intentionally uses only Streamlit session state plus a process-level
counter. A fresh browser tab/session is counted once. The displayed value is
shared by sessions handled by the same running Streamlit process.

Important: Streamlit Community Cloud does not guarantee persistence of files or
process memory across app reboots/restarts, so a truly permanent cumulative
counter requires an external persistent store. This module deliberately avoids
a database and external service.
"""

from __future__ import annotations

import threading
from typing import Final

import streamlit as st

_LOCK = threading.Lock()
_STARTING_COUNT: Final[int] = 0

# Shared by Streamlit sessions handled by the same Python process.
_PROCESS_USER_COUNT = _STARTING_COUNT


def register_session() -> int:
    """Count this Streamlit session once and return the process-wide count."""
    global _PROCESS_USER_COUNT

    if st.session_state.get("exam_surveillance_user_counted", False):
        return current_count()

    with _LOCK:
        _PROCESS_USER_COUNT += 1
        current = _PROCESS_USER_COUNT

    st.session_state["exam_surveillance_user_counted"] = True
    st.session_state["exam_surveillance_user_count_at_entry"] = current
    return current


def current_count() -> int:
    """Return the current process-wide visitor/session count."""
    return int(_PROCESS_USER_COUNT)


def render_count(location: str = "main") -> int:
    """Register the current session and render a compact visitor counter."""
    count = register_session()

    if location == "sidebar":
        st.sidebar.metric("👥 App users", count)
        st.sidebar.caption("One count is added when a new Streamlit session enters the app.")
    elif location == "footer":
        st.markdown(
            f"<div class='visitor-counter'>👥 <b>App users:</b> {count} &nbsp;•&nbsp; One count per new app session</div>",
            unsafe_allow_html=True,
        )
    else:
        st.metric("👥 App users", count)

    return count
