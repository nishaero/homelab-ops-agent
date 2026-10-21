import base64
import hashlib
import hmac
import os
from datetime import datetime, timezone

import requests
import streamlit as st

API_BASE = os.environ["OPS_AGENT_API_BASE"]
API_TOKEN = os.environ["OPS_AGENT_API_TOKEN"]
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

CHAT_BASE = os.environ["OPS_AGENT_CHAT_BASE"]
CHAT_TOKEN = os.environ["OPS_AGENT_CHAT_TOKEN"]
CHAT_HEADERS = {"Authorization": f"Bearer {CHAT_TOKEN}"}

st.set_page_config(page_title="Ops Agent Admin", page_icon="🤖", layout="wide")

# Global responsive + chat styling. Scoped to data-testid attributes
# Streamlit itself renders (not guessable class names that change between
# versions) - if this stops working after a Streamlit upgrade, that's why.
st.markdown("""
<style>
@media (max-width: 700px) {
    div[data-testid="stHorizontalBlock"] { flex-direction: column !important; }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        width: 100% !important; flex: 1 1 100% !important;
    }
    div[data-testid="stHorizontalBlock"] .stButton button { width: 100%; }
}
/* Chat: centered, readable-width column, like a real chat app rather than
   a form field stretched across a wide desktop screen. */
div[data-testid="stChatMessage"], div[data-testid="stChatInput"] {
    max-width: 46rem; margin-left: auto; margin-right: auto;
}
div[data-testid="stChatMessage"] { padding: 0.75rem 1rem; }
.chat-empty { text-align: center; padding: 3rem 1rem 1rem; color: #8b949e; }
.chat-empty .icon { font-size: 2.2rem; margin-bottom: 0.5rem; }
.chat-empty .title { font-size: 1.25rem; font-weight: 700; color: inherit; margin-bottom: 0.3rem; }
div[data-testid="stExpander"] { max-width: 46rem; margin-left: auto; margin-right: auto; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------- login gate --
#
# This panel can restart/update containers and (via the chat tab) run
# arbitrary allowed tool calls, so it needs its own login on top of
# whatever network-level restriction (NPM access list, firewall) fronts
# it - this container's port is reachable directly on the LAN, not just
# through the reverse proxy, so network-level restriction alone is not
# sufficient. Two paths, either one sets st.session_state["authed_as"]:
#
# 1. GitHub OAuth (primary) - intentionally no extra OAuth library
#    (requirements.txt is just streamlit+requests): redirect to GitHub's
#    authorize URL, GitHub redirects back with ?code=..., exchange that
#    for a token, fetch the username, check it against the allowed list.
# 2. Username/password (fallback, optional) - for when GitHub itself is
#    unreachable (no internet, GitHub outage) and you still need in on
#    the LAN. Only appears if at least one basic-auth user is configured.
#    Passwords are never stored in plaintext, only a salted PBKDF2 hash.
#    This has no brute-force rate limiting of its own; it relies on this
#    panel already being LAN-only and strong generated passwords.
#
# WHO is allowed in (both lists) lives in ops_agent_api.py's /users
# endpoint (persisted to a file on the Proxmox host, not env vars) so it
# can be managed from the Config tab instead of hand-editing .env and
# rebuilding the container every time. On first boot, if that list is
# empty, it's seeded once from this container's own GITHUB_ALLOWED_LOGIN /
# BASIC_AUTH_USERNAME / BASIC_AUTH_PASSWORD_HASH env vars (still read
# below, now optional) so an existing deployment doesn't get locked out -
# see _seed_users_if_empty(). After that, the env vars are unused; the
# file is authoritative and the UI manages it.
GITHUB_CLIENT_ID = os.environ["GITHUB_CLIENT_ID"]
GITHUB_CLIENT_SECRET = os.environ["GITHUB_CLIENT_SECRET"]
OAUTH_REDIRECT_URI = os.environ["OAUTH_REDIRECT_URI"]

_SEED_GITHUB_LOGIN = os.environ.get("GITHUB_ALLOWED_LOGIN", "").strip()
_SEED_BASIC_USERNAME = os.environ.get("BASIC_AUTH_USERNAME", "").strip()
_SEED_BASIC_PASSWORD_HASH = os.environ.get("BASIC_AUTH_PASSWORD_HASH", "").strip()


def _fetch_users():
    try:
        r = requests.get(f"{API_BASE}/users", headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"github_allowed_logins": [], "basic_auth_users": []}


def _seed_users_if_empty():
    users = _fetch_users()
    if users.get("github_allowed_logins") or users.get("basic_auth_users"):
        return users
    if not _SEED_GITHUB_LOGIN and not (_SEED_BASIC_USERNAME and _SEED_BASIC_PASSWORD_HASH):
        return users
    try:
        if _SEED_GITHUB_LOGIN:
            requests.post(f"{API_BASE}/users/github/add", headers=HEADERS, timeout=10,
                          json={"username": _SEED_GITHUB_LOGIN})
        if _SEED_BASIC_USERNAME and _SEED_BASIC_PASSWORD_HASH:
            requests.post(f"{API_BASE}/users/basic/add", headers=HEADERS, timeout=10,
                          json={"username": _SEED_BASIC_USERNAME, "password_hash": _SEED_BASIC_PASSWORD_HASH})
        return _fetch_users()
    except Exception:
        return users


def _github_login_url():
    return (
        "https://github.com/login/oauth/authorize"
        f"?client_id={GITHUB_CLIENT_ID}&redirect_uri={OAUTH_REDIRECT_URI}"
        "&scope=read:user&allow_signup=false"
    )


def _exchange_code_for_user(code):
    resp = requests.post(
        "https://github.com/login/oauth/access_token",
        headers={"Accept": "application/json"},
        data={
            "client_id": GITHUB_CLIENT_ID,
            "client_secret": GITHUB_CLIENT_SECRET,
            "code": code,
            "redirect_uri": OAUTH_REDIRECT_URI,
        },
        timeout=10,
    )
    resp.raise_for_status()
    access_token = resp.json().get("access_token")
    if not access_token:
        return None
    user_resp = requests.get(
        "https://api.github.com/user",
        headers={"Authorization": f"Bearer {access_token}", "Accept": "application/vnd.github+json"},
        timeout=10,
    )
    user_resp.raise_for_status()
    return user_resp.json().get("login")


def _hash_password(password, iterations=600000):
    salt = os.urandom(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${base64.b64encode(salt).decode()}${base64.b64encode(h).decode()}"


def _verify_password(password, password_hash):
    """password_hash format: pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>.
    Constant-time comparison."""
    try:
        algo, iterations, salt_b64, hash_b64 = password_hash.split("$")
        if algo != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
    except (ValueError, TypeError):
        return False
    computed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, int(iterations))
    return hmac.compare_digest(computed, expected)


_GITHUB_MARK = (
    '<svg height="18" width="18" viewBox="0 0 16 16" fill="currentColor">'
    '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 '
    '0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 '
    '1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 '
    '0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 '
    '1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 '
    '3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/>'
    "</svg>"
)


def _render_login_page(basic_auth_users, error=None):
    st.markdown("""
    <style>
    .block-container { max-width: 460px !important; padding-top: 10vh !important; }
    .login-title { font-size: 1.7rem; font-weight: 700; text-align: center; margin-bottom: 4px; }
    .login-sub { color: #8b949e; text-align: center; margin-bottom: 26px; font-size: 0.95rem; }
    .gh-btn {
        display: flex; align-items: center; justify-content: center; gap: 10px;
        background: #238636; color: #fff !important; text-decoration: none;
        padding: 12px 20px; border-radius: 8px; font-weight: 600; font-size: 1rem;
        width: 100%; box-sizing: border-box; margin-bottom: 6px;
    }
    .gh-btn:hover { background: #2ea043; }
    .auth-divider {
        display: flex; align-items: center; text-align: center; color: #8b949e;
        margin: 22px 0 14px; font-size: 0.82rem;
    }
    .auth-divider::before, .auth-divider::after { content: ''; flex: 1; border-bottom: 1px solid #30363d; }
    .auth-divider::before { margin-right: .75em; }
    .auth-divider::after { margin-left: .75em; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-title">🤖 Ops Agent Admin</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-sub">Sign in to manage your homelab</div>', unsafe_allow_html=True)

    if error:
        st.error(error)

    st.markdown(f'<a class="gh-btn" href="{_github_login_url()}">{_GITHUB_MARK} Log in with GitHub</a>', unsafe_allow_html=True)

    if basic_auth_users:
        st.markdown('<div class="auth-divider">or, if GitHub is unreachable</div>', unsafe_allow_html=True)
        with st.form("basic_auth_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Log in", use_container_width=True):
                match = next((bu for bu in basic_auth_users if bu["username"] == u), None)
                if match and _verify_password(p, match["password_hash"]):
                    st.session_state["authed_as"] = u
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
    st.stop()


def _require_login():
    if st.session_state.get("authed_as"):
        return
    users = _seed_users_if_empty()
    allowed_gh = set(users.get("github_allowed_logins", []))
    basic_auth_users = users.get("basic_auth_users", [])

    code = st.query_params.get("code")
    if code:
        st.query_params.clear()
        try:
            login = _exchange_code_for_user(code)
        except Exception as e:
            _render_login_page(basic_auth_users, error=f"GitHub login failed ({e}) - use the fallback below if it's still down.")
            return
        if login in allowed_gh:
            st.session_state["authed_as"] = login
            st.rerun()
        else:
            _render_login_page(basic_auth_users, error=f"GitHub account '{login}' is not authorized for this panel.")
            return
    _render_login_page(basic_auth_users)


_require_login()

with st.sidebar:
    st.caption(f"Logged in as **{st.session_state['authed_as']}**")
    if st.button("Log out"):
        del st.session_state["authed_as"]
        st.rerun()

st.title("🤖 Ops Agent Admin")

tab_config, tab_logs, tab_ask, tab_containers = st.tabs(
    ["⚙️ Config", "📜 Logs & History", "💬 Chat", "🐳 Containers"]
)

# ------------------------------------------------------------------ config --
with tab_config:
    st.caption(
        "Only these tuning knobs are editable here on purpose - which hosts/containers are "
        "managed, compose paths, and the hard safety exclusions (UniFi, QNAP OS/firmware, "
        "Proxmox+LXC OS-patching) all stay edited by hand over SSH, never through this form, "
        "so a UI slip can't widen what the agent is allowed to touch."
    )
    try:
        resp = requests.get(f"{API_BASE}/config", headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        cfg = data["config"]
        editable = set(data["editable_keys"])
    except Exception as e:
        st.error(f"Could not load config from {API_BASE}: {e}")
        st.stop()

    with st.form("config_form"):
        col1, col2 = st.columns(2)
        with col1:
            model = st.text_input("Ollama model", value=cfg.get("model", ""))
            ollama_url = st.text_input("Ollama URL", value=cfg.get("ollama_url", ""))
            cycle_interval = st.number_input(
                "Cycle interval (minutes)", min_value=5, max_value=180,
                value=int(cfg.get("cycle_interval_minutes", 20)),
                help="Must match the systemd timer's own schedule to be meaningful - changing this here does not reschedule the timer itself.",
            )
        with col2:
            notify_cooldown = st.number_input(
                "Notification cooldown (hours)", min_value=0.0, max_value=48.0,
                value=float(cfg.get("notify_cooldown_hours", 4)), step=0.5,
                help="How long a recurring, still-unresolved issue stays suppressed before re-alerting.",
            )
            health_wait = st.number_input(
                "Health check wait (seconds)", min_value=10, max_value=600,
                value=int(cfg.get("health_wait_seconds", 90)),
            )
            max_attempts = st.number_input(
                "Max update attempts per cycle", min_value=1, max_value=20,
                value=int(cfg.get("max_update_attempts_per_cycle", 3)),
            )
            backup_limit = st.number_input(
                "Data backup size limit (GB)", min_value=1, max_value=50,
                value=int(cfg.get("data_backup_size_limit_gb", 2)),
                help="Bind-mounts larger than this are skipped for pre-update tar backup (relies on the rollback-tag + volume-preserving update instead) - too slow to snapshot inline otherwise.",
            )

        submitted = st.form_submit_button("Save", type="primary")
        if submitted:
            updates = {
                "model": model, "ollama_url": ollama_url,
                "cycle_interval_minutes": int(cycle_interval),
                "notify_cooldown_hours": float(notify_cooldown),
                "health_wait_seconds": int(health_wait),
                "max_update_attempts_per_cycle": int(max_attempts),
                "data_backup_size_limit_gb": int(backup_limit),
            }
            try:
                r = requests.put(f"{API_BASE}/config", headers=HEADERS, json={"config": updates}, timeout=10)
                r.raise_for_status()
                st.success("Saved - takes effect on the next scheduled cycle, not the one currently running (if any).")
            except Exception as e:
                st.error(f"Save failed: {e}")

    with st.expander("Read-only: hosts, containers & compose paths (edit via SSH, not here)"):
        st.json({
            "docker_hosts": cfg.get("docker_hosts"),
            "os_auto_hosts": cfg.get("os_auto_hosts"),
            "data_containers": cfg.get("data_containers"),
            "compose_dirs": cfg.get("compose_dirs"),
            "ntfy": cfg.get("ntfy"),
        })

    st.divider()
    st.subheader("Login access")
    st.caption(
        "Who can log into this panel. Takes effect immediately - existing browser sessions "
        "aren't affected until they log out. If you ever remove every user from both lists, "
        "the next page load re-seeds from this container's original env vars, so you can't "
        "lock yourself out entirely as long as those are still set."
    )
    users_now = _fetch_users()
    col_gh, col_basic = st.columns(2)

    with col_gh:
        st.markdown("**GitHub accounts**")
        for u in users_now.get("github_allowed_logins", []):
            c1, c2 = st.columns([4, 1])
            c1.write(f"@{u}")
            if c2.button("✕", key=f"rmgh_{u}", help=f"Remove @{u}"):
                requests.post(f"{API_BASE}/users/github/remove", headers=HEADERS, timeout=10, json={"username": u})
                st.rerun()
        with st.form("add_gh_user", clear_on_submit=True):
            new_gh = st.text_input("Add GitHub username", label_visibility="collapsed", placeholder="GitHub username")
            if st.form_submit_button("Add") and new_gh.strip():
                requests.post(f"{API_BASE}/users/github/add", headers=HEADERS, timeout=10, json={"username": new_gh.strip()})
                st.rerun()

    with col_basic:
        st.markdown("**Username / password**")
        for bu in users_now.get("basic_auth_users", []):
            c1, c2 = st.columns([4, 1])
            c1.write(bu["username"])
            if c2.button("✕", key=f"rmbasic_{bu['username']}", help=f"Remove {bu['username']}"):
                requests.post(f"{API_BASE}/users/basic/remove", headers=HEADERS, timeout=10, json={"username": bu["username"]})
                st.rerun()
        with st.form("add_basic_user", clear_on_submit=True):
            new_u = st.text_input("Username", placeholder="Username")
            new_p = st.text_input("Password", type="password", placeholder="Password")
            if st.form_submit_button("Add"):
                if new_u.strip() and new_p:
                    requests.post(f"{API_BASE}/users/basic/add", headers=HEADERS, timeout=10, json={
                        "username": new_u.strip(), "password_hash": _hash_password(new_p),
                    })
                    st.rerun()
                else:
                    st.warning("Username and password are both required.")

# -------------------------------------------------------------------- logs --
with tab_logs:
    errors_only = st.checkbox("Errors only")
    refresh = st.button("🔄 Refresh")

    try:
        state = requests.get(f"{API_BASE}/state", headers=HEADERS, timeout=10).json()
        cycles = state.get("cycle_history", [])
    except Exception as e:
        st.error(f"Could not load cycle history: {e}")
        cycles = []

    if not cycles:
        st.info("No cycles recorded yet.")
    else:
        for cycle in reversed(cycles):
            has_error = bool(cycle.get("error")) or cycle.get("status") == "error"
            if errors_only and not has_error:
                continue
            ts = cycle.get("timestamp", "")
            try:
                ts_fmt = datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M UTC")
            except ValueError:
                ts_fmt = ts
            icon = "🔴" if has_error else ("🟡" if cycle.get("unhealthy") else "🟢")
            with st.expander(f"{icon} {ts_fmt} - {cycle.get('status', '?').upper()} - {cycle.get('action_count', 0)} action(s)"):
                if cycle.get("error"):
                    st.error(cycle["error"])
                if cycle.get("unhealthy"):
                    st.warning(f"Unhealthy: {', '.join(cycle['unhealthy'])}")
                if cycle.get("action_counts"):
                    st.write("**Action breakdown:**", cycle["action_counts"])
                for a in cycle.get("actions", []):
                    st.text(f"• {a.get('tool')}: {a.get('summary')}")

    st.divider()
    st.subheader("Raw log tail")
    n_lines = st.slider("Lines", 50, 1000, 300)
    try:
        logs = requests.get(f"{API_BASE}/logs?lines={n_lines}", headers=HEADERS, timeout=10).json()
        text = "\n".join(logs.get("lines", []))
        if errors_only:
            text = "\n".join(l for l in logs.get("lines", []) if "ERROR" in l or "Traceback" in l or "WARNING" in l)
        st.code(text or "(no log lines)", language="log")
    except Exception as e:
        st.error(f"Could not load logs: {e}")


# ------------------------------------------------------------------- chat --
SUGGESTIONS = [
    "Is everything healthy right now?",
    "Any pending updates?",
    "Check vaultwarden's logs",
]

with tab_ask:
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    top_l, top_r = st.columns([5, 1])
    with top_l:
        st.caption(
            "Real-time - checks live state before answering, and can restart/update things right now "
            "if you ask (same safety net as always). No tool to write code or deploy new services, so "
            "it'll explain what that would take instead of pretending to do it."
        )
    with top_r:
        if st.session_state.chat_messages and st.button("🗑️ New chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()

    pending_prompt = None
    chat_box = st.container(height=520)
    with chat_box:
        if not st.session_state.chat_messages:
            st.markdown(
                '<div class="chat-empty"><div class="icon">🤖</div>'
                '<div class="title">Ask about your homelab</div>'
                '<div>Live status, logs, updates - or ask it to fix something.</div></div>',
                unsafe_allow_html=True,
            )
            cols = st.columns(len(SUGGESTIONS))
            for col, s in zip(cols, SUGGESTIONS):
                with col:
                    if st.button(s, key=f"suggest_{s}", use_container_width=True):
                        pending_prompt = s
        for m in st.session_state.chat_messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])
                if m.get("tool_calls"):
                    with st.expander(f"🔧 {len(m['tool_calls'])} tool call(s)"):
                        for tc in m["tool_calls"]:
                            st.text(f"{tc['tool']}({tc['args']}) -> {tc['result_summary']}")

    typed_prompt = st.chat_input("Message the agent...")
    prompt = pending_prompt or typed_prompt

    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.spinner("Checking..."):
            try:
                payload = {"messages": [
                    {"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages
                ]}
                # 300s, not 180: structured-output turns (e.g. drafting a full automation config)
                # measured ~115s per model call against the local model, and a reply can take
                # several turns - see MODEL_CALL_TIMEOUT's comment in ops_agent_chat.py.
                r = requests.post(f"{CHAT_BASE}/chat", headers=CHAT_HEADERS, json=payload, timeout=300)
                r.raise_for_status()
                result = r.json()
                reply = result.get("reply") or "(empty reply)"
                tool_calls = result.get("tool_calls", [])
            except Exception as e:
                reply = f"Could not reach the chat service: {e}"
                tool_calls = []
        st.session_state.chat_messages.append({"role": "assistant", "content": reply, "tool_calls": tool_calls})
        st.rerun()


def _run_action(tool, host, name, verb):
    with st.spinner(f"{verb} {name}..."):
        try:
            r = requests.post(f"{CHAT_BASE}/action", headers=CHAT_HEADERS, timeout=180, json={
                "tool": tool, "host": host, "container": name,
            })
            r.raise_for_status()
            result = r.json().get("result", {})
        except Exception as e:
            st.error(f"{verb} failed: {e}")
            return
    if result.get("error"):
        st.error(f"{verb} failed: {result['error']}")
    elif result.get("success") is False:
        st.error(f"{verb} did not succeed: {result}")
    else:
        st.success(f"Done - {result}")


# ------------------------------------------------------------ containers --
with tab_containers:
    st.caption(
        "Buttons here run restart_container/apply_update directly and immediately, right now - same "
        "safety net as always (backup, health-check, auto-rollback; updates still hold off and explain "
        "instead of applying if the changelog looks breaking), just not queued."
    )
    try:
        state = requests.get(f"{API_BASE}/state", headers=HEADERS, timeout=10).json()
        last_cycle_at = state.get("last_cycle_at")
        managed = state.get("last_managed", [])
    except Exception as e:
        st.error(f"Could not load fleet state: {e}")
        last_cycle_at = None
        managed = []

    if last_cycle_at:
        try:
            age = datetime.now(timezone.utc) - datetime.fromisoformat(last_cycle_at)
            st.caption(f"Container list as of the last autonomous cycle - {int(age.total_seconds() // 60)} min ago.")
        except ValueError:
            pass

    if not managed:
        st.info("No container data yet - wait for the first cycle to complete, or ask in the Chat tab.")

    by_host = {}
    for c in managed:
        by_host.setdefault(c.get("host", "?"), []).append(c)

    for host, containers in sorted(by_host.items()):
        st.subheader(host)
        for c in sorted(containers, key=lambda x: x.get("name", "")):
            name = c.get("name", "?")
            state_ok = c.get("state") == "running" and c.get("health") != "unhealthy"
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                icon = "🟢" if state_ok else "🔴"
                st.write(f"{icon} **{name}** - {c.get('state', '?')}" + (f" ({c.get('health')})" if c.get("health") else ""))
            with col2:
                if st.button("Restart now", key=f"restart_{host}_{name}"):
                    _run_action("restart_container", host, name, "Restarting")
            with col3:
                if st.button("Update now", key=f"update_{host}_{name}"):
                    _run_action("apply_update", host, name, "Updating")

    st.subheader("Plex (native QNAP app, not a container)")
    st.caption("Runs as a QNAP-native app, not Container Station - not covered by the fleet list above. Polled directly here from the same exporter that feeds Grafana and HA.")
    try:
        plex = requests.get(os.environ.get("PLEX_STATUS_URL", "http://plex-host.example:19877/status.json"), timeout=5).json()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", "🟢 Online" if plex.get("up") else "🔴 Offline")
        with col2:
            st.metric("Active streams", plex.get("active_sessions", 0), delta=(f"{plex.get('transcoding_sessions', 0)} transcoding" if plex.get("transcoding_sessions") else None), delta_color="off")
        with col3:
            st.metric("Library items", sum(plex.get("libraries", {}).values()))
        with st.expander("Library breakdown"):
            for section, count in sorted(plex.get("libraries", {}).items()):
                st.write(f"{section}: {count}")
    except Exception as e:
        st.error(f"Could not reach Plex exporter: {e}")
