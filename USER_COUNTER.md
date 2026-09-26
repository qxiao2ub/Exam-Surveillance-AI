# No-Database App User Counter

The application includes a lightweight visitor/session counter in `user_counter.py`.

## How it works

- A new Streamlit browser session is counted once.
- The count is rendered on the main page, sidebar, and footer so it remains visible throughout the application.
- It uses only `st.session_state` plus an in-process Python counter.
- It does **not** use a SQL/NoSQL database or any external analytics service.
- It does not collect names, emails, IP addresses, cookies, or other identifying information.

## Important Community Cloud limitation

Streamlit Community Cloud does not guarantee persistence of app-generated files or process memory across restarts/reboots. Therefore, a counter that uses **no external persistent storage at all** cannot be guaranteed to remain permanently cumulative across every future reboot.

This implementation is intentionally the simplest no-database option: the count is cumulative while the current app process is alive, and each new Streamlit session increments it exactly once. If permanent cross-reboot counting is required later, a persistent external store or analytics service is necessary.
