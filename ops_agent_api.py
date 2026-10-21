#!/usr/bin/env python3
"""
Ops Agent Admin API - tiny sidecar HTTP API, runs directly on Proxmox
(systemd, not docker - same deployment style as ops_agent.py itself)
alongside the agent, since its config/state/log files live on this host's
bare filesystem while the admin UI runs in a container on a different
host (LXC 102). Exposes just enough to power that UI: read/write the
agent's YAML config, read-only access to cycle history + logs, and the
list of who's allowed to log into the UI (GitHub usernames + basic-auth
username/password-hash pairs - see USERS_FILE below).

Interactive/action requests (chat, restart-now/update-now buttons) do NOT
go through this process - see ops_agent_chat.py, a separate sidecar that
runs on this same host under ops_agent.py's own EnvironmentFile and can
execute the agent's tools synchronously. This process intentionally stays
narrower than that: it does NOT import ops_agent.py or hold any of its
credentials (HA token, ntfy topic), on purpose.

Not meant to be reachable from anywhere except the admin UI container -
restricted at the Proxmox hypervisor firewall level to LXC 102's IP only
(see /etc/pve/nodes/<node>/host.fw - and remember to add a rule for EVERY
sidecar port, this one included, not just the first one you set up), plus
a shared-secret bearer token as defense in depth.
"""
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

CONFIG_FILE = Path("/root/ops-agent-config.yaml")
STATE_FILE = Path("/var/lib/ops-agent/state.json")
LOG_DIR = Path("/var/log/ops-agent")
USERS_FILE = Path("/var/lib/ops-agent/ui-users.json")
TOKEN = os.environ["OPS_AGENT_API_TOKEN"]  # required, shared secret with the UI
PORT = int(os.environ.get("OPS_AGENT_API_PORT", "8099"))

# Only these top-level config keys may be edited via the API - mirrors the
# same safety stance as ops_agent.py's config loader: nothing that could
# widen the agent's hard-coded blast-radius limits (UniFi, QNAP OS/
# firmware, Proxmox/LXC OS-patching) is exposed here, because those stay
# Python-level constants, not config, on purpose.
EDITABLE_KEYS = {
    "notify_cooldown_hours", "cycle_interval_minutes", "health_wait_seconds",
    "max_update_attempts_per_cycle", "data_backup_size_limit_gb", "model",
    "ollama_url",
}


# ------------------------------------------------------------------- users --
#
# Who can log into the admin UI. Two independent lists: GitHub usernames
# allowed through OAuth, and username/password-hash pairs for the
# no-internet-needed fallback (see ops-ui/app.py's login gate). This file
# is the source of truth once it has anything in it; on first boot (file
# doesn't exist / is empty), app.py seeds it from its own
# GITHUB_ALLOWED_LOGIN / BASIC_AUTH_USERNAME / BASIC_AUTH_PASSWORD_HASH
# env vars so upgrading an existing deployment doesn't lock anyone out -
# see app.py's _seed_users_if_empty(). This process only ever stores and
# returns a password hash it's given, already computed - it never sees or
# handles a plaintext password.
def _load_users():
    if USERS_FILE.exists():
        try:
            data = json.loads(USERS_FILE.read_text())
            data.setdefault("github_allowed_logins", [])
            data.setdefault("basic_auth_users", [])
            return data
        except json.JSONDecodeError:
            pass
    return {"github_allowed_logins": [], "basic_auth_users": []}


def _save_users(data):
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(data, indent=1))
    USERS_FILE.chmod(0o600)


class Handler(BaseHTTPRequestHandler):
    def _auth_ok(self):
        return self.headers.get("Authorization") == f"Bearer {TOKEN}"

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length))

    def do_GET(self):
        if not self._auth_ok():
            return self._send(401, {"error": "unauthorized"})
        if self.path == "/config":
            cfg = yaml.safe_load(CONFIG_FILE.read_text()) if CONFIG_FILE.exists() else {}
            return self._send(200, {"config": cfg, "editable_keys": sorted(EDITABLE_KEYS)})
        if self.path == "/state":
            state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
            return self._send(200, state)
        if self.path.startswith("/logs"):
            m = re.search(r"lines=(\d+)", self.path)
            n = int(m.group(1)) if m else 300
            log_file = LOG_DIR / "ops-agent.log"
            if not log_file.exists():
                return self._send(200, {"lines": []})
            lines = log_file.read_text(errors="replace").splitlines()[-n:]
            return self._send(200, {"lines": lines})
        if self.path == "/users":
            return self._send(200, _load_users())
        return self._send(404, {"error": "not found"})

    def do_PUT(self):
        if not self._auth_ok():
            return self._send(401, {"error": "unauthorized"})
        if self.path != "/config":
            return self._send(404, {"error": "not found"})
        try:
            body = self._read_json_body()
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid json body"})
        updates = body.get("config", {})
        bad_keys = set(updates) - EDITABLE_KEYS
        if bad_keys:
            return self._send(400, {"error": f"not editable via API: {sorted(bad_keys)}"})
        current = yaml.safe_load(CONFIG_FILE.read_text()) if CONFIG_FILE.exists() else {}
        current.update(updates)
        try:
            new_text = yaml.safe_dump(current, sort_keys=True, default_flow_style=False)
        except yaml.YAMLError as e:
            return self._send(400, {"error": f"could not serialize config: {e}"})
        CONFIG_FILE.write_text(new_text)
        return self._send(200, {"config": current})

    def do_POST(self):
        if not self._auth_ok():
            return self._send(401, {"error": "unauthorized"})

        if self.path == "/users/github/add":
            try:
                body = self._read_json_body()
            except json.JSONDecodeError:
                return self._send(400, {"error": "invalid json body"})
            username = (body.get("username") or "").strip()
            if not username:
                return self._send(400, {"error": "username is required"})
            users = _load_users()
            if username not in users["github_allowed_logins"]:
                users["github_allowed_logins"].append(username)
                _save_users(users)
            return self._send(200, users)

        if self.path == "/users/github/remove":
            try:
                body = self._read_json_body()
            except json.JSONDecodeError:
                return self._send(400, {"error": "invalid json body"})
            username = body.get("username")
            users = _load_users()
            users["github_allowed_logins"] = [u for u in users["github_allowed_logins"] if u != username]
            _save_users(users)
            return self._send(200, users)

        if self.path == "/users/basic/add":
            try:
                body = self._read_json_body()
            except json.JSONDecodeError:
                return self._send(400, {"error": "invalid json body"})
            username = (body.get("username") or "").strip()
            password_hash = body.get("password_hash") or ""
            if not username or not password_hash:
                return self._send(400, {"error": "username and password_hash are required"})
            # Sanity-check the format (pbkdf2_sha256$<iterations>$<salt>$<hash>) so
            # garbage can't get written silently - this process never computes a
            # hash itself, only validates the shape of one it's handed.
            parts = password_hash.split("$")
            if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
                return self._send(400, {"error": "malformed password_hash"})
            users = _load_users()
            users["basic_auth_users"] = [u for u in users["basic_auth_users"] if u["username"] != username]
            users["basic_auth_users"].append({"username": username, "password_hash": password_hash})
            _save_users(users)
            return self._send(200, users)

        if self.path == "/users/basic/remove":
            try:
                body = self._read_json_body()
            except json.JSONDecodeError:
                return self._send(400, {"error": "invalid json body"})
            username = body.get("username")
            users = _load_users()
            users["basic_auth_users"] = [u for u in users["basic_auth_users"] if u["username"] != username]
            _save_users(users)
            return self._send(200, users)

        return self._send(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        pass  # keep the systemd journal quiet - this is a tiny admin sidecar


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"ops-agent-api listening on :{PORT}")
    server.serve_forever()
