#!/usr/bin/env python3
"""
Sync secrets FROM Vaultwarden (the "homelab-secrets" folder) INTO the
actual files/config each service reads. Vault is the source of truth
after this; run this again after changing a value in the vault to push
it out to where it's actually consumed.

Requires: `bw` CLI installed and BW_SESSION exported to an unlocked
session (this script does not unlock anything itself, on purpose - see
the ops-agent README's Secrets section for the unlock flow).

Run this ON PROXMOX - it reads local files directly and reaches the
other hosts (docker-lxc, workstation) the same way ops_agent.py does
(pct exec / ssh), pushing updated files back the same safe way (write
locally, then push - never inline-interpolate a secret into a remote
shell command, several of these values contain characters like $ that
get mangled by shell variable expansion otherwise).
"""
import json
import os
import re
import subprocess
import sys
import tempfile

WORKSTATION_SSH = os.environ.get("OPS_AGENT_WORKSTATION_SSH", "user@workstation.example")
WORKSTATION_HOME = os.environ.get("OPS_AGENT_WORKSTATION_HOME", "/home/user")
from pathlib import Path

import yaml

OPS_AGENT_API_BASE = "http://127.0.0.1:8099"


def run(cmd, check=True):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if check and p.returncode != 0:
        print(f"FAILED: {' '.join(cmd)}\n{p.stderr}", file=sys.stderr)
        sys.exit(1)
    return p.stdout


def bw_item_fields(name):
    data = json.loads(run(["bw", "get", "item", name]))
    return {f["name"]: f.get("value", "") for f in data.get("fields", [])}


def parse_env(text):
    out = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        out[k.strip()] = v.strip()
    return out


def update_env_content(text, updates):
    """Replace KEY=value lines for keys in `updates`, preserving everything
    else (comments, ordering, unrelated keys). Appends any key not already
    present."""
    lines = text.splitlines()
    seen = set()
    out = []
    for line in lines:
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k = s.split("=", 1)[0].strip()
            if k in updates:
                out.append(f"{k}={updates[k]}")
                seen.add(k)
                continue
        out.append(line)
    for k, v in updates.items():
        if k not in seen:
            out.append(f"{k}={v}")
    return "\n".join(out) + "\n"


def push_local_file(local_path, content, mode=0o600):
    Path(local_path).write_text(content)
    Path(local_path).chmod(mode)


def push_lxc_file(content, remote_path, ctid="102", mode="600"):
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(content)
        tmp = f.name
    run(["pct", "push", ctid, tmp, remote_path])
    run(["pct", "exec", ctid, "--", "chmod", mode, remote_path])
    Path(tmp).unlink()


def push_ssh_file(content, host, remote_path, mode="600"):
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(content)
        tmp = f.name
    run(["scp", "-o", "BatchMode=yes", tmp, f"{host}:/tmp/_sync_push"])
    run(["ssh", "-o", "BatchMode=yes", host, f"mv /tmp/_sync_push {remote_path} && chmod {mode} {remote_path}"])
    Path(tmp).unlink()


def read_lxc_file(remote_path, ctid="102"):
    return run(["pct", "exec", ctid, "--", "cat", remote_path])


def read_ssh_file(host, remote_path):
    return run(["ssh", "-o", "BatchMode=yes", host, f"cat {remote_path}"])


def sync_ops_agent():
    fields = bw_item_fields("ops-agent")

    # ops-agent.env (Proxmox, local)
    env_path = "/root/ops-agent.env"
    current = Path(env_path).read_text()
    updated = update_env_content(current, {
        "OPS_AGENT_HA_TOKEN": fields.get("OPS_AGENT_HA_TOKEN", ""),
        "OPS_AGENT_CHAT_TOKEN": fields.get("OPS_AGENT_CHAT_TOKEN", ""),
    })
    push_local_file(env_path, updated)
    print(f"updated {env_path}")

    # ops-agent-api.env (Proxmox, local)
    api_env_path = "/root/ops-agent-api.env"
    current = Path(api_env_path).read_text()
    updated = update_env_content(current, {"OPS_AGENT_API_TOKEN": fields.get("OPS_AGENT_API_TOKEN", "")})
    push_local_file(api_env_path, updated)
    print(f"updated {api_env_path}")

    # ops-agent-config.yaml's ntfy.topic (Proxmox, local) - YAML, not .env
    cfg_path = "/root/ops-agent-config.yaml"
    cfg = yaml.safe_load(Path(cfg_path).read_text()) or {}
    cfg.setdefault("ntfy", {})["topic"] = fields.get("NTFY_TOPIC", cfg.get("ntfy", {}).get("topic", ""))
    Path(cfg_path).write_text(yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False))
    Path(cfg_path).chmod(0o600)
    print(f"updated {cfg_path} (ntfy.topic)")

    # ops-ui/.env (docker-lxc) - tokens + GitHub OAuth app creds. Note:
    # BASIC_AUTH_USERNAME/PASSWORD_HASH here are only a SEED (see
    # app.py's _seed_users_if_empty) - the live credential is managed via
    # /users/basic/add below, not this file, once the managed list is
    # non-empty (which it already is after first boot).
    ui_env_path = "/root/ops-ui/.env"
    current = read_lxc_file(ui_env_path)
    updated = update_env_content(current, {
        "OPS_AGENT_API_TOKEN": fields.get("OPS_AGENT_API_TOKEN", ""),
        "OPS_AGENT_CHAT_TOKEN": fields.get("OPS_AGENT_CHAT_TOKEN", ""),
        "GITHUB_CLIENT_ID": fields.get("GITHUB_CLIENT_ID", ""),
        "GITHUB_CLIENT_SECRET": fields.get("GITHUB_CLIENT_SECRET", ""),
        "GITHUB_ALLOWED_LOGIN": fields.get("GITHUB_ALLOWED_LOGIN", ""),
    })
    push_lxc_file(updated, ui_env_path)
    print(f"updated docker-lxc:{ui_env_path}")

    # Basic-auth password: this is the ACTUAL live credential store (see
    # ops_agent_api.py's /users endpoints) - hash it fresh and upsert,
    # rather than writing a hash into .env where nothing re-reads it.
    password = fields.get("BASIC_AUTH_PASSWORD", "")
    username = fields.get("BASIC_AUTH_USERNAME", "")
    if password and username:
        import base64
        import hashlib
        import os as _os
        salt = _os.urandom(16)
        h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600000)
        password_hash = f"pbkdf2_sha256$600000${base64.b64encode(salt).decode()}${base64.b64encode(h).decode()}"
        token = parse_env(Path("/root/ops-agent-api.env").read_text()).get("OPS_AGENT_API_TOKEN", "")
        r = subprocess.run(
            ["curl", "-s", "-X", "POST", "-H", f"Authorization: Bearer {token}",
             "-H", "Content-Type: application/json",
             "-d", json.dumps({"username": username, "password_hash": password_hash}),
             f"{OPS_AGENT_API_BASE}/users/basic/add"],
            capture_output=True, text=True,
        )
        print(f"basic-auth user '{username}' upserted via /users/basic/add: {r.stdout[:120]}")


def sync_immich():
    fields = bw_item_fields("immich")
    env_path = WORKSTATION_HOME + "/ai-workspace/immich/.env"
    current = read_ssh_file(WORKSTATION_SSH, env_path)
    updated = update_env_content(current, {
        "IMMICH_ADMIN_API_KEY": fields.get("IMMICH_ADMIN_API_KEY", ""),
        "DB_PASSWORD": fields.get("DB_PASSWORD", ""),
        "SMB_PASSWORD": fields.get("SMB_PASSWORD", ""),
    })
    push_ssh_file(updated, WORKSTATION_SSH, env_path)
    print(f"updated workstation:{env_path}")


def sync_nextcloud():
    fields = bw_item_fields("nextcloud")
    env_path = WORKSTATION_HOME + "/ai-workspace/nextcloud/.env"
    current = read_ssh_file(WORKSTATION_SSH, env_path)
    updated = update_env_content(current, {
        "MYSQL_ROOT_PASSWORD": fields.get("MYSQL_ROOT_PASSWORD", ""),
        "MYSQL_PASSWORD": fields.get("MYSQL_PASSWORD", ""),
    })
    push_ssh_file(updated, WORKSTATION_SSH, env_path)
    print(f"updated workstation:{env_path}")


def sync_openwebui():
    fields = bw_item_fields("openwebui")
    env_path = WORKSTATION_HOME + "/ai-workspace/openwebui/.env"
    current = read_ssh_file(WORKSTATION_SSH, env_path)
    updated = update_env_content(current, {
        "OPENWEBUI_API_KEY": fields.get("OPENWEBUI_API_KEY", ""),
        "QDRANT_API_KEY": fields.get("QDRANT_API_KEY", ""),
        "OPENWEBUI_HMAC_KEY": fields.get("OPENWEBUI_HMAC_KEY", ""),
    })
    push_ssh_file(updated, WORKSTATION_SSH, env_path)
    print(f"updated workstation:{env_path}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    funcs = {
        "ops-agent": sync_ops_agent, "immich": sync_immich,
        "nextcloud": sync_nextcloud, "openwebui": sync_openwebui,
    }
    if which == "all":
        for f in funcs.values():
            f()
    elif which in funcs:
        funcs[which]()
    else:
        print(f"unknown target {which!r}, choose from: all, {', '.join(funcs)}", file=sys.stderr)
        sys.exit(1)
    print("\nDone. Restart/recreate any container whose values changed for it to pick them up "
          "(values are usually unchanged from what was already deployed on this first run).")
