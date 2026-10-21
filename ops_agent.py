#!/usr/bin/env python3
"""
Ops agent for a self-hosted homelab.

Runs on the Proxmox host from a systemd timer every 20 minutes. A local Ollama
model (gemma4:31b by default) makes the decisions through tool calls. The model
never gets a shell. It can only call the functions listed in TOOLS.

Two kinds of cycle:
  - Light cycle (every tick, no model calls): finds crashed containers and
    stopped LXCs or VMs and fixes them directly. Anything that needs judgment
    waits for the heavy cycle.
  - Heavy cycle (heavy_cycle_start_hour to heavy_cycle_end_hour in the config, default
    00:00 to 05:00): the model checks for updates, reads changelogs, applies
    updates, patches the OS and reviews automations. It runs at night because
    the workstation GPU is shared with the voice assistant and Immich.

What it manages:
  - Docker containers on the workstation, docker-lxc (LXC 102) and the QNAP.
    Compose projects are updated with compose pull and up -d. Containers
    without compose are recreated from a saved docker inspect.
  - Data containers (Vaultwarden, Nextcloud, Immich) get their bind mounts
    tarred before an update, unless the mount is too large.
  - Every update tags the running image as :ops-agent-rollback first and rolls
    back if the health check fails.
  - Home Assistant through its REST API only: update entities, a backup before
    each update, and a check that it comes back.
  - OS patching with apt on the workstation, npm-lxc and docker-lxc. A reboot
    is scheduled for 03:30 only when the host asks for one.
  - Weekly check for a new Ubuntu LTS on the workstation (notify only).
  - Resource growth and restart-loop reports (notify only).
  - ADS-B Pi checks, switched off by default (adsb_enabled in the config).

What it never touches: UniFi, QNAP firmware, and OS patching of the Proxmox
host. These are code rules, not config, so a config change cannot widen them.

Alerts go to a self-hosted ntfy instance. Settings are read from
ops-agent-config.yaml. Missing keys fall back to defaults, so a missing file
is not a startup failure. See README.md and docs/ai-design.md.
"""

import json
import logging
import os
import re
import shlex
import subprocess
import sys
import tarfile
import socket
import time
import urllib.request
import urllib.error
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import yaml

# ---------------------------------------------------------------- config --
#
# Tunable behavioral settings live in ops-agent-config.yaml (edited either
# directly, or - for the "safe" subset only - via the admin UI through
# ops_agent_api.py's EDITABLE_KEYS allowlist). Defaults below match prior
# hard-coded behavior exactly, so a missing/partial config file is never a
# startup failure. The hard safety exclusions (UniFi never touched, QNAP
# OS/firmware excluded, Proxmox host OS-patching excluded: NEVER_PATCH_HOSTS) are NOT config and
# never will be - those stay Python logic further down this file, on
# purpose, so no config edit or UI form can widen the agent's blast radius
# past what was explicitly approved.

CONFIG_FILE = Path(os.environ.get("OPS_AGENT_CONFIG_FILE", "/root/ops-agent-config.yaml"))


def _load_config():
    if CONFIG_FILE.exists():
        try:
            return yaml.safe_load(CONFIG_FILE.read_text()) or {}
        except yaml.YAMLError as e:
            logging.getLogger("ops-agent").error(f"failed to parse {CONFIG_FILE}, using defaults: {e}")
    return {}


_cfg = _load_config()

LOG_DIR = Path("/var/log/ops-agent")
BACKUP_DIR = Path("/var/backups/ops-agent")
STATE_FILE = Path("/var/lib/ops-agent/state.json")
DECISION_LOG_FILE = Path("/var/lib/ops-agent/decisions.jsonl")


def _log_decision(action, target, reason, outcome):
    """Append one JSON line recording a real action taken (not read-only
    checks) - what, on what target, why, and what happened. Best-effort:
    a logging failure must never break the actual operation, so any
    exception here is caught and silently ignored."""
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "target": target,
            "reason": reason,
            "outcome": outcome,
        }
        DECISION_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with DECISION_LOG_FILE.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def check_restart_loop(host, container, window_hours=6):
    """Report-only: how many times has restart_container already fired for
    this exact target in the last window_hours, per the decision log.
    Mechanically restarting an unhealthy container is the right call once
    - it's not the right call a fourth time in six hours, that's a crash
    loop, and the fix (if there even is one from here) isn't 'restart
    again'. Call this before restart_container whenever a container shows
    up as unhealthy again soon after a previous restart - if the count is
    already high, that's a signal to notify_human with the pattern and
    what get_logs showed, not to just restart it once more and move on."""
    target = f"{host}/{container}"
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    timestamps = []
    try:
        if DECISION_LOG_FILE.exists():
            with DECISION_LOG_FILE.open() as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("action") != "restart_container" or entry.get("target") != target:
                        continue
                    try:
                        ts = datetime.fromisoformat(entry["timestamp"])
                    except (KeyError, ValueError):
                        continue
                    if ts >= cutoff:
                        timestamps.append(entry["timestamp"])
    except Exception as e:
        return {"error": str(e)}
    return {"target": target, "window_hours": window_hours, "restart_count": len(timestamps), "restart_timestamps": sorted(timestamps)}


OLLAMA_URL = os.environ.get("OPS_AGENT_OLLAMA_URL", _cfg.get("ollama_url", "http://localhost:11434/api/chat"))
MODEL = os.environ.get("OPS_AGENT_MODEL", _cfg.get("model", "gemma4:31b"))

HA_BASE = os.environ.get("OPS_AGENT_HA_BASE", "https://ha.example.com")
HA_TOKEN = os.environ["OPS_AGENT_HA_TOKEN"]  # required, no default - see .env.example
# HA is still used for HA's own health/update checks - notifications no
# longer go through it at all (moved to ntfy, see notify_human() below and
# the NTFY_* settings), kept separate on purpose per explicit request.

NTFY_BASE_URL = os.environ.get("OPS_AGENT_NTFY_BASE_URL", (_cfg.get("ntfy") or {}).get("base_url", "https://ntfy.example.com"))
NTFY_TOPIC = os.environ.get("OPS_AGENT_NTFY_TOPIC", (_cfg.get("ntfy") or {}).get("topic"))
if not NTFY_TOPIC:
    raise RuntimeError("ntfy topic not configured - set ntfy.topic in ops-agent-config.yaml or OPS_AGENT_NTFY_TOPIC")

CYCLE_INTERVAL_MINUTES = int(_cfg.get("cycle_interval_minutes", 20))  # must match the systemd timer's OnUnitActiveSec
NOTIFY_COOLDOWN_HOURS = float(_cfg.get("notify_cooldown_hours", 4))  # for topic-tagged notifications only

# Confirmed real 2026-10-04: this agent's own LLM turns can grow to 15-20K+
# tokens of accumulated tool output within one cycle, taking minutes to
# generate on the workstation's single shared GPU - directly colliding with
# anything else on that GPU, including the local voice assistant's
# conversation model. _wait_for_ollama_idle() only defers the START of a
# call, it can't shrink an already-huge in-flight turn. Rather than let the
# full LLM-driven cycle run around the clock, it's confined to a low-usage
# overnight window (default 00:00-05:00 local time) via HEAVY_CYCLE_HOURS;
# every other cycle in the day runs run_lightweight_cycle() instead, which
# reuses the same pure-Python health-check/mechanical-restart helpers but
# never calls the model at all - zero GPU contention, at the cost of
# deferring anything that really needs the model's judgment (update/
# changelog decisions, ambiguous automation-health flags) to the next
# overnight cycle.
CYCLE_BUDGET_SECONDS = int(_cfg.get("cycle_budget_seconds", 1500))
HEAVY_CYCLE_HOURS = range(
    int(_cfg.get("heavy_cycle_start_hour", 0)),
    int(_cfg.get("heavy_cycle_end_hour", 5)),
)

DOCKER_HOSTS = _cfg.get("docker_hosts", ["workstation", "docker-lxc", "qnap"])
# OS-level apt patching is intentionally scoped down to ONLY the workstation
# and the ADS-B Pi. Proxmox itself and both its LXCs (101 nginxproxymanager,
# 102 docker) run on the hypervisor. The hypervisor itself and the NAS are in
# NEVER_PATCH_HOSTS above and can never be OS-patched by this agent. The LXCs
# listed in os_auto_hosts are patched like any other host. This list is
# config-file-editable (by hand, not via the web UI); the never-patch hosts are
# filtered out of it in code even if they are listed.
ADSB_ENABLED = bool(_cfg.get("adsb_enabled", False))  # parked on request; set adsb_enabled: true to bring it back
# Hosts that are never OS-patched by this agent, whatever the config says: the hypervisor and the NAS.
# A bad patch or reboot on either takes the agent's own platform down. This is enforced here in code, twice
# (the config list is filtered, and apply_os_updates refuses), so a hand edit of the config cannot widen it.
NEVER_PATCH_HOSTS = frozenset({"proxmox-host", "qnap"})
_requested_os_hosts = list(_cfg.get("os_auto_hosts", ["workstation", "adsb"] if ADSB_ENABLED else ["workstation"]))
OS_AUTO_HOSTS = [h for h in _requested_os_hosts if h not in NEVER_PATCH_HOSTS]
if len(OS_AUTO_HOSTS) != len(_requested_os_hosts):
    logging.getLogger("ops-agent").warning("ignoring never-patch hosts in os_auto_hosts: %s",
                                           sorted(set(_requested_os_hosts) & NEVER_PATCH_HOSTS))
OS_CHECK_ONLY_HOSTS = []

# Automations that should always be enabled - see check_expected_automations_enabled()
ALWAYS_ENABLED_AUTOMATIONS = _cfg.get("always_enabled_automations", [])

# Proxmox LXCs/VMs that should always be running - see check_proxmox_infra_status().
# Example config: [{vmid: 102, kind: lxc, name: docker}, {vmid: 100, kind: qm, name: haos}]
ALWAYS_RUNNING_PROXMOX = _cfg.get("always_running_proxmox", [])

# Native (non-Docker) QNAP apps to reachability-check - see check_qnap_app_reachable().
# Example config: [{name: Plex, port: 32400}]
QNAP_APPS = _cfg.get("qnap_apps", [])

# Resource-trend monitoring (check_host_resource_trend): report-only, no
# ceiling enforced unless you set one. Example config:
#   host_resource_ceiling_gb: {docker-lxc: 6}
HOST_RESOURCE_CEILING_GB = _cfg.get("host_resource_ceiling_gb", {}) or {}
RESOURCE_GROWTH_ALERT_PCT = float(_cfg.get("resource_growth_alert_pct", 0.5))

# ADS-B Raspberry Pi: not docker-based, just apt + a couple of systemd
# services to watch. Core receiver services (dashboard depends on these);
# the adsbexchange-* ones are secondary (feed-out only) but still checked.
ADSB_CORE_SERVICES = ["readsb.service", "tar1090.service"]
ADSB_SECONDARY_SERVICES = ["adsbexchange-feed.service", "adsbexchange-mlat.service",
                            "adsbexchange-gpsmon.service", "adsbexchange-stats.service"]

# containers holding really irreplaceable user data - get an extra
# data-directory backup attempt before any update, on top of the
# universal image-rollback-tag safety net every container gets.
_data_containers_cfg = _cfg.get("data_containers", {
    "workstation": ["immich_postgres", "immich_server", "nextcloud-app", "nextcloud-db"],
    "docker-lxc": ["vaultwarden"],
    "qnap": [],
})
DATA_CONTAINERS = {host: set(names or []) for host, names in _data_containers_cfg.items()}

# compose project working dirs, so we know where to run `docker compose`
# for each container that belongs to one. Config-file-editable (by hand,
# not via the web UI). bentopdf's path note: it's deployed as a Portainer
# stack whose compose label says /data/compose/3, but that's the path
# INSIDE the portainer container - /data there is actually the
# portainer_data named volume, whose real host-side path is what's in the
# config file. A label's working_dir can't be trusted blindly for
# Portainer-deployed stacks, it needs this translation.
COMPOSE_DIRS = {
    (host, container): path
    for host, containers in (_cfg.get("compose_dirs") or {}).items()
    for container, path in (containers or {}).items()
}

DATA_BACKUP_SIZE_LIMIT_BYTES = int(_cfg.get("data_backup_size_limit_gb", 2)) * 1024 * 1024 * 1024  # skip tar above this, log instead

# BACKUP_DIR (/var/backups/ops-agent) only exists, root-owned, on Proxmox
# itself - fine for docker-lxc (reached via `pct exec` as root) and qnap
# (its `admin` SSH user is uid 0). The workstation's SSH user
# is NOT root, so it needs a location it actually owns and can write to -
# using the global path there silently failed every backup attempt.
BACKUP_DIR_FOR_HOST = {"workstation": os.environ.get("OPS_AGENT_WORKSTATION_BACKUP_DIR", "/home/user/ops-agent-backups")}
WORKSTATION_SSH = os.environ.get("OPS_AGENT_WORKSTATION_SSH", "user@workstation.example")
QNAP_SSH = os.environ.get("OPS_AGENT_QNAP_SSH", "admin@nas.example")
QNAP_HOST = os.environ.get("OPS_AGENT_QNAP_HOST", "nas.example")
IMMICH_GATE = os.environ.get("OPS_AGENT_IMMICH_GATE", "~/bin/immich-gate")

MAX_UPDATE_ATTEMPTS_PER_CYCLE = int(_cfg.get("max_update_attempts_per_cycle", 3))  # cap blast radius of a single bad cycle
HEALTH_WAIT_SECONDS = int(_cfg.get("health_wait_seconds", 90))
HEALTH_POLL_INTERVAL = 5

log = logging.getLogger("ops-agent")


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(LOG_DIR / "ops-agent.log", maxBytes=5 * 1024 * 1024, backupCount=5)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.setLevel(logging.INFO)


# ------------------------------------------------------ host execution --

def run(host, cmd, timeout=120):
    """Run a shell command on the given host. Returns (rc, stdout, stderr)."""
    if host == "docker-lxc":
        full = ["pct", "exec", "102", "--", "bash", "-c", cmd]
    elif host == "npm-lxc":
        full = ["pct", "exec", "101", "--", "bash", "-c", cmd]
    elif host == "proxmox-host":
        full = ["bash", "-c", cmd]
    elif host == "workstation":
        full = ["ssh", "-o", "BatchMode=yes", WORKSTATION_SSH, cmd]
    elif host == "qnap":
        full = ["ssh", "-o", "BatchMode=yes", QNAP_SSH, cmd]
    elif host == "adsb":
        full = ["ssh", "-o", "BatchMode=yes", "adsb", cmd]
    else:
        raise ValueError(f"unknown host {host!r}")
    try:
        p = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"


def docker_cli(host, args_str, timeout=120):
    """Run a docker CLI command on a host that has the docker CLI (not qnap)."""
    return run(host, f"docker {args_str}", timeout=timeout)


def qnap_docker_api(method, path, data=None, timeout=60):
    """Talk to QNAP's docker Engine API over its unix socket via curl, since
    there's no docker CLI installed there. The URL and the body are shell quoted as single
    arguments, so a value with a quote or a semicolon in it cannot break out of the command."""
    if method not in ("GET", "POST", "DELETE"):
        raise ValueError(f"unsupported method {method!r}")
    url = shlex.quote("http://localhost" + path)
    cmd = f"curl -sS -X {method} --unix-socket /var/run/docker.sock {url}"
    if data is not None:
        cmd = f"curl -sS -X {method} -H 'Content-Type: application/json' --unix-socket /var/run/docker.sock -d {shlex.quote(json.dumps(data))} {url}"
    rc, out, err = run("qnap", cmd, timeout=timeout)
    return rc, out, err


# ------------------------------------------------- model supplied arguments --
# Everything the model passes to a tool is untrusted: a prompt injected into a changelog or a log could try to
# craft a container name that escapes a shell command or a URL. Names are checked against strict patterns
# before any tool runs, and the places that build commands also quote their values (defence in depth).
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
IMAGE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,255}")
TAG_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
SERVICE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
ENTITY_RE = re.compile(r"[a-z][a-z0-9_]*\.[a-z0-9_]+")
ARG_PATTERNS = {"container": NAME_RE, "candidate_image_ref": IMAGE_RE, "candidate_tag": TAG_RE,
                "domain": SERVICE_RE, "service": SERVICE_RE, "entity_id": ENTITY_RE}
INT_ARGS = ("lines", "hours", "window_hours", "lookback_hours", "days", "vmid")


def validate_tool_args(args):
    """Raise ValueError if a model supplied argument is not a plain, well formed value."""
    if not isinstance(args, dict):
        raise ValueError("arguments must be an object")
    for key, rx in ARG_PATTERNS.items():
        if key in args and not (isinstance(args[key], str) and rx.fullmatch(args[key])):
            raise ValueError(f"invalid {key}: {str(args[key])[:60]!r}")
    for key in INT_ARGS:
        if key in args and args[key] is not None:       # null means 'use the default'
            try:
                if isinstance(args[key], bool):
                    raise TypeError
                args[key] = int(args[key])
            except (TypeError, ValueError):
                raise ValueError(f"invalid {key}: {str(args[key])[:30]!r}") from None


# ------------------------------------------------------------- tools --

def list_containers(host):
    """Uses `docker inspect` (batched, one round-trip) rather than `docker
    ps`, specifically to get each container's RestartPolicy - that's what
    lets the crash-detection logic (see run_cycle) correctly tell apart
    "meant to run forever but is down" from "ran once and exited 0 on
    purpose" (e.g. hello-world) or "a human stopped this intentionally"
    (restart policy 'no', currently exited). Without this distinction the
    agent will happily "fix" containers that were never broken - confirmed
    live during testing (see NEEDS_MANUAL_SETUP notes at the bottom)."""
    if host == "qnap":
        rc, out, err = qnap_docker_api("GET", "/containers/json?all=true")
        if rc != 0 or not out.strip():
            return {"error": err or "empty response"}
        try:
            containers = json.loads(out)
        except json.JSONDecodeError:
            return {"error": f"bad json: {out[:200]}"}
        result = []
        for c in containers:
            info = _inspect(host, c["Id"]) or {}
            result.append(_container_summary(info, c["Names"][0].lstrip("/"), c["Image"], c["State"], c["Status"]))
        return result
    else:
        rc, out, err = docker_cli(host, "ps -aq")
        if rc != 0:
            return {"error": err}
        ids = [i for i in out.strip().splitlines() if i.strip()]
        if not ids:
            return []
        rc, out, err = docker_cli(host, f"inspect {' '.join(ids)}")
        if rc != 0:
            return {"error": err}
        try:
            infos = json.loads(out)
        except json.JSONDecodeError:
            return {"error": f"bad json from inspect: {out[:200]}"}
        result = []
        for info in infos:
            name = info["Name"].lstrip("/")
            image = info["Config"]["Image"]
            state = info["State"]["Status"]
            status = f"{state} ({info['State'].get('StartedAt', '')})"
            result.append(_container_summary(info, name, image, state, status))
        return result


def _container_summary(info, name, image, state, status):
    restart_policy = (info.get("HostConfig", {}).get("RestartPolicy", {}) or {}).get("Name", "") or "no"
    health = None
    health_block = info.get("State", {}).get("Health")
    if health_block:
        health = health_block.get("Status")
    expected_always_running = restart_policy in ("always", "unless-stopped")
    return {
        "name": name,
        "image": image,
        "state": state,
        "status": status,
        "health": health,
        "restart_policy": restart_policy,
        "expected_always_running": expected_always_running,
    }


def get_logs(host, container, lines=50):
    if host == "qnap":
        rc, out, err = qnap_docker_api("GET", f"/containers/{container}/logs?stdout=true&stderr=true&tail={lines}")
        return out or err
    rc, out, err = run(host, f"docker logs --tail {lines} {shlex.quote(container)} 2>&1")
    return out


def restart_container(host, container):
    log.info(f"[{host}] restarting {container}")
    if host == "qnap":
        rc, out, err = qnap_docker_api("POST", f"/containers/{container}/restart")
        ok = rc == 0
    else:
        rc, out, err = docker_cli(host, f"restart {shlex.quote(container)}")
        ok = rc == 0
    _log_decision("restart_container", f"{host}/{container}", "operator-requested container restart", "success" if ok else "failed")
    return {"success": ok, "output": out, "error": err}


def _inspect(host, container):
    if host == "qnap":
        rc, out, err = qnap_docker_api("GET", f"/containers/{container}/json")
        if rc != 0 or not out.strip():
            return None
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return None
    rc, out, err = docker_cli(host, f"inspect {shlex.quote(container)}")
    if rc != 0:
        return None
    try:
        arr = json.loads(out)
        return arr[0] if arr else None
    except (json.JSONDecodeError, IndexError):
        return None


def check_for_update(host, container):
    """Pull the image the container currently uses (same tag) and compare
    digests. Returns whether a newer image is now available locally."""
    info = _inspect(host, container)
    if not info:
        return {"error": "container not found"}
    image = info["Config"]["Image"]
    old_id = info.get("Image")

    # Locally-built (docker-compose "build:", not "image:") containers get an
    # auto-generated name of "<project>-<service>" or "<project>_<service>" -
    # never a real registry reference. Confirmed real: check_for_update kept
    # failing every cycle on ops-ui/bentopdf/ntfy-webhook-adapter with "pull
    # access denied, repository does not exist" because it was trying to pull
    # e.g. "ops-ui-ops-ui" from Docker Hub. An "update" for these means
    # rebuilding from source, which this tool doesn't do - report that
    # plainly instead of a misleading pull failure.
    labels = info.get("Config", {}).get("Labels") or {}
    project = labels.get("com.docker.compose.project")
    service = labels.get("com.docker.compose.service")
    image_repo = image.rsplit(":", 1)[0] if "/" not in image.rsplit(":", 1)[-1] else image
    if project and service and image_repo in (f"{project}-{service}", f"{project}_{service}"):
        return {"image": image, "locally_built": True,
                "note": "locally built via docker-compose build - no registry image to pull, rebuild from source to update"}

    if host == "qnap":
        rc, out, err = qnap_docker_api("POST", f"/images/create?fromImage={image.split(':')[0]}&tag={image.split(':')[1] if ':' in image else 'latest'}", timeout=180)
        pulled_ok = rc == 0
    else:
        rc, out, err = docker_cli(host, f"pull {shlex.quote(image)}", timeout=180)
        pulled_ok = rc == 0

    if not pulled_ok:
        if "pull access denied" in str(err) or "repository does not exist" in str(err):
            return {"image": image, "locally_built": True,
                    "note": "image exists only locally (docker build / not in a registry) - nothing to pull"}
        return {"error": f"pull failed: {err}"}

    new_info_rc, new_out, _ = docker_cli(host, f"image inspect {shlex.quote(image)} --format '{{{{.Id}}}}'") if host != "qnap" else (0, None, None)
    new_id = new_out.strip() if new_out else None

    return {
        "image": image,
        "old_id": old_id,
        "new_id": new_id,
        "update_available": bool(new_id and new_id != old_id),
    }


def _guess_github_source(image_ref):
    """Fallback for when an image carries no org.opencontainers.image.source
    label (common - confirmed real for portainer/agent, which has rich OCI
    labels but not that one). Two conventions are reliable enough to try:
    ghcr.io images always mirror their GitHub repo by GHCR's own design
    (ghcr.io/OWNER/REPO == github.com/OWNER/REPO), and plenty of plain
    Docker Hub 'namespace/name' images use the same name on GitHub too
    (confirmed real: portainer/agent -> github.com/portainer/agent). Never
    trusted blindly - the caller's GitHub API call verifies it for real; a
    wrong guess just 404s and falls back to unavailable, same as before this
    existed. Bare official images (e.g. 'nginx', 'mariadb') have no owner to
    guess from and aren't covered - those still need the label or an
    explicit mapping."""
    repo_part = image_ref.split(":")[0]
    if repo_part.startswith("ghcr.io/"):
        parts = repo_part[len("ghcr.io/"):].split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return None
    if repo_part.count("/") == 1 and "." not in repo_part.split("/")[0]:
        return repo_part
    return None


def _github_releases_changelog(owner_repo, current_version, new_version):
    """Version-tag-based changelog lookup via GitHub's releases API - the
    original check_changelog approach, factored out so both check_changelog
    and check_upstream_tag_changelog can share it."""
    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{owner_repo}/releases?per_page=30",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "ops-agent"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            releases = json.loads(resp.read())
    except Exception as e:
        return {"available": False, "note": f"GitHub API lookup failed: {e}"}

    relevant, collecting = [], False
    for rel in releases:
        tag = rel.get("tag_name", "")
        if new_version in tag:
            collecting = True
        if collecting:
            relevant.append(f"## {tag}\n{(rel.get('body') or '')[:1500]}")
        if current_version in tag:
            break
    if not relevant:
        return {"available": False, "note": f"couldn't match {current_version} -> {new_version} against {owner_repo}'s GitHub releases (tag naming mismatch)"}
    return {
        "available": True, "source": owner_repo,
        "from_version": current_version, "to_version": new_version,
        "changelog_excerpt": "\n\n".join(relevant)[:6000],
    }


def _github_commit_changelog(owner_repo, old_sha, new_sha):
    """Fallback changelog signal for images that carry a git revision label
    (org.opencontainers.image.revision, or the git_commit label some images
    use instead) but no version label to bound a release-tag range against -
    confirmed real for portainer/agent, which has neither .source nor
    .version but does have .revision. Uses GitHub's compare API to get the
    real commit list between the two exact commits the running and
    candidate images were built from - less curated than release notes, but
    a real, verifiable signal rather than nothing."""
    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{owner_repo}/compare/{old_sha}...{new_sha}",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "ops-agent"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            comparison = json.loads(resp.read())
    except Exception as e:
        return {"available": False, "note": f"GitHub compare API lookup failed: {e}"}

    commits = comparison.get("commits", [])
    if not commits:
        return {"available": False, "note": f"GitHub compare of {old_sha[:7]}...{new_sha[:7]} on {owner_repo} returned no commits"}
    messages = [f"- {c['commit']['message'].splitlines()[0]}" for c in commits[-50:]]
    return {
        "available": True, "source": owner_repo,
        "from_version": old_sha[:7], "to_version": new_sha[:7],
        "changelog_excerpt": f"{len(commits)} commit(s) between {old_sha[:7]} and {new_sha[:7]} (commit "
                              f"messages, not curated release notes - no version/release-tag metadata was "
                              f"available on this image):\n" + "\n".join(messages[:50]),
    }


def check_changelog(host, container):
    """Best-effort: reads OCI image labels (org.opencontainers.image.source /
    .version) off the container's currently-running image and the freshly
    pulled candidate image (call check_for_update first), and if the source
    is a GitHub repo, fetches release notes spanning those two versions via
    GitHub's public API. Returns an explicit 'not available' note when the
    image doesn't carry this metadata (many don't) or the lookup otherwise
    fails - that's not an error, it just means apply_update's own backup +
    health-check + rollback safety net is the only signal for that
    container, same as before this tool existed. Purely read-only - never
    edits anything. The system prompt decides what to do with the result:
    a changelog that reads as breaking (major version jump, migration
    language, removed config) means skip apply_update and notify_human with
    the excerpt instead of applying; anything else proceeds as normal."""
    if host == "qnap":
        return {"available": False, "note": "changelog lookup not implemented for qnap (no docker CLI there)"}
    info = _inspect(host, container)
    if not info:
        return {"available": False, "note": "container not found"}
    current_labels = info.get("Config", {}).get("Labels") or {}
    current_version = current_labels.get("org.opencontainers.image.version")
    current_revision = current_labels.get("org.opencontainers.image.revision") or current_labels.get("git_commit")
    source = current_labels.get("org.opencontainers.image.source")

    image_ref = info.get("Config", {}).get("Image")
    rc, out, err = docker_cli(host, f"inspect --format '{{{{json .Config.Labels}}}}' {shlex.quote(image_ref)}")
    try:
        new_labels = json.loads(out.strip()) or {}
    except (json.JSONDecodeError, AttributeError):
        new_labels = {}
    new_version = new_labels.get("org.opencontainers.image.version")
    new_revision = new_labels.get("org.opencontainers.image.revision") or new_labels.get("git_commit")
    source = source or new_labels.get("org.opencontainers.image.source")

    owner_repo = source.split("github.com/")[-1].rstrip("/").removesuffix(".git") if source and "github.com" in source else None
    if not owner_repo:
        owner_repo = _guess_github_source(image_ref)

    if not owner_repo:
        return {"available": False, "note": "no GitHub source label on this image, and no reliable source to guess from - no changelog signal available"}

    if current_version and new_version:
        if current_version == new_version:
            return {"available": False, "note": "version label unchanged from running container - no version-level changelog to check"}
        return _github_releases_changelog(owner_repo, current_version, new_version)

    if current_revision and new_revision:
        if current_revision == new_revision:
            return {"available": False, "note": "revision label unchanged from running container - no changelog to check"}
        return _github_commit_changelog(owner_repo, current_revision, new_revision)

    return {"available": False, "note": f"source found ({owner_repo}) but neither version nor revision labels present - can't bound a changelog range"}


# --- Upstream tag-aware updates -------------------------------------------
# check_for_update/check_changelog above only ever compare a container's
# PINNED tag against itself (a fresh pull of the same tag) - by design,
# since that's what keeps a tag like "nextcloud:32" from silently jumping
# major versions the way ":latest" did (see project_nextcloud_reliability_fix
# memory). But that also means the agent can never notice "34" exists at
# all. The functions below close that gap: list what tags actually exist
# upstream, find the highest one that's a newer version of the SAME image
# variant (same v-prefix and suffix as the current tag, e.g. never
# suggesting a plain "32" jump to a "-rocm" build), get a changelog for
# that specific jump, and - if the calling model judges the changelog safe,
# not by keyword-matching but by actually reading it - apply it via the
# exact same backup+recreate+health-check+rollback path as apply_update.


def _parse_image_ref(image):
    """Split a full image reference into (registry, repo, tag, digest).
    registry is None for implicit Docker Hub - only treated as an explicit
    registry host when the first path segment contains a '.' or ':' or is
    'localhost', the same heuristic docker itself uses. digest is the
    sha256:... part when the image is pinned by digest (in which case tag
    is irrelevant for update purposes - see check_upstream_tag_update)."""
    digest = None
    if "@" in image:
        image, digest = image.split("@", 1)

    if "/" in image:
        first_segment, rest = image.split("/", 1)
        if "." in first_segment or ":" in first_segment or first_segment == "localhost":
            registry = first_segment
        else:
            registry, rest = None, image
    else:
        registry, rest = None, image

    if ":" in rest.rsplit("/", 1)[-1]:
        repo, tag = rest.rsplit(":", 1)
    else:
        repo, tag = rest, None

    return registry, repo, tag, digest


_GHCR_TAG_PAGE_LIMIT = 30  # actively-tagged repos (per-commit dev builds etc.) can
                            # have thousands of tags with no guaranteed sort order -
                            # confirmed on immich-app/immich-server, where 300 tags
                            # of unsorted history never even reached the current
                            # v3.x releases at all. Only hit as a fallback now that
                            # check_upstream_tag_update tries GitHub releases first
                            # for anything with a GitHub source label.
_DOCKERHUB_TAG_PAGE_LIMIT = 5  # Hub's own ordering=last_updated already surfaces
                                # recent tags in the first page or two - more pages
                                # buys nothing and risks anonymous rate-limiting
                                # (confirmed: 30 pages 403'd on a repo 3 pages
                                # handled fine moments earlier).


def _list_registry_tags(registry, repo):
    """List tags from ghcr.io (OCI API, Link-header pagination) or docker.io
    (Hub v2, 'next'-URL pagination). See _GHCR_TAG_PAGE_LIMIT /
    _DOCKERHUB_TAG_PAGE_LIMIT for why the two use different page caps
    100 each. Registries make no promise tags are returned newest-first (or
    sorted at all) - the page limit needs real headroom, not just enough for
    a small/quiet repo, or a busy one's actual latest tags never surface."""
    if registry == "ghcr.io":
        token_url = f"https://ghcr.io/token?scope=repository:{repo}:pull&service=ghcr.io"
        req = urllib.request.Request(token_url, headers={"User-Agent": "ops-agent"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise Exception(f"ghcr.io token request failed: {exc}")
        if not isinstance(data, dict) or "token" not in data:
            raise Exception("ghcr.io token response missing 'token' field")
        token = data["token"]

        tags = []
        url = f"https://ghcr.io/v2/{repo}/tags/list?n=100"
        for page in range(_GHCR_TAG_PAGE_LIMIT):
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ops-agent", "Authorization": f"Bearer {token}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    link_header = resp.headers.get("Link")
            except Exception as exc:
                raise Exception(f"ghcr.io tags request failed: {exc}")
            if not isinstance(body, dict) or "tags" not in body or not isinstance(body["tags"], list):
                raise Exception("ghcr.io tags response missing valid 'tags' list")
            tags.extend(body["tags"])
            if link_header and page < _GHCR_TAG_PAGE_LIMIT - 1:
                next_url = None
                for part in link_header.split(","):
                    part = part.strip()
                    if 'rel="next"' in part:
                        s, e = part.find("<"), part.find(">")
                        if s != -1 and e != -1:
                            next_url = part[s + 1 : e]
                        break
                if next_url:
                    # ghcr.io's Link header gives a path-only URL (e.g.
                    # "/v2/owner/repo/tags/list?..."), not absolute - urllib
                    # can't fetch that directly.
                    url = urllib.parse.urljoin("https://ghcr.io", next_url)
                    continue
            break
        return tags

    elif registry in ("docker.io", None):
        api_repo = repo if "/" in repo else f"library/{repo}"
        tags = []
        url = f"https://hub.docker.com/v2/repositories/{api_repo}/tags?page_size=100&ordering=last_updated"
        for page in range(_DOCKERHUB_TAG_PAGE_LIMIT):
            req = urllib.request.Request(url, headers={"User-Agent": "ops-agent"})
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
            except Exception as exc:
                raise Exception(f"docker.io tags request failed: {exc}")
            if not isinstance(body, dict) or "results" not in body or not isinstance(body["results"], list):
                raise Exception("docker.io tags response missing valid 'results' list")
            for item in body["results"]:
                if not isinstance(item, dict) or "name" not in item:
                    raise Exception("docker.io tags result entry missing 'name' field")
                tags.append(item["name"])
            if page < _DOCKERHUB_TAG_PAGE_LIMIT - 1 and body.get("next"):
                url = body["next"]
                continue
            break
        return tags

    else:
        raise Exception(f"Unsupported registry: {registry!r}")


def _parse_tag_shape(tag):
    """Split a tag into (prefix, version_tuple, suffix), e.g. 'v3.1.0-rocm'
    -> ('v', (3, 1, 0), '-rocm'). Returns None if the tag doesn't start
    with an optional 'v' followed by a dotted integer version."""
    m = re.match(r"^(v?)(\d+(?:\.\d+){0,3})(.*)$", tag)
    if not m:
        return None
    prefix, version_str, suffix = m.groups()
    version_tuple = tuple(int(p) for p in version_str.split("."))
    return prefix, version_tuple, suffix


def _find_newer_tag(current_tag, all_tags):
    """Among all_tags, find the single highest one with the EXACT SAME
    prefix ('v' or not), suffix (e.g. '-rocm', '-alpine'), AND version
    granularity (number of dotted segments) as current_tag - varying
    only the numeric value - that's strictly newer. Both the shape and
    the granularity match matter: a bare major-version pin like
    Nextcloud's '32' (which auto-tracks patches within that major on
    every same-tag pull) must only ever be compared against other bare
    major tags like '33', never against a fully-qualified patch tag like
    '34.0.4' - those are different pinning philosophies for the same
    repo, and conflating them would silently change how updates behave
    going forward, not just bump a version. Intentionally conservative:
    a different suffix is a different image variant, not a newer
    version of this one, and never proposed either."""
    current_shape = _parse_tag_shape(current_tag)
    if current_shape is None:
        return None
    cur_prefix, cur_version, cur_suffix = current_shape

    best_tag, best_version = None, cur_version
    for tag in all_tags:
        shape = _parse_tag_shape(tag)
        if shape is None:
            continue
        prefix, version, suffix = shape
        if prefix != cur_prefix or suffix != cur_suffix:
            continue
        if len(version) != len(cur_version):
            continue
        if version > best_version:
            best_tag, best_version = tag, version
    return best_tag


def _list_github_release_tags(owner_repo):
    """Published release tag_names only (via GitHub's Releases API, 3 pages
    of 100 - real release counts are nowhere near registry tag counts, so
    this is plenty). Confirmed necessary on immich-app/immich-server: its
    GHCR repo has thousands of per-commit/per-PR CI tags with no useful
    sort order, where even 3000 raw registry tags never reached a current
    release tag at all - actual releases are a needle in that haystack.
    GitHub's own release list is the clean, targeted source of truth for
    "what versions did this project actually cut", so try this first for
    anything with a GitHub source label, and only fall back to raw
    registry tag listing when there's no such label to work from."""
    tags = []
    url = f"https://api.github.com/repos/{owner_repo}/releases?per_page=100"
    for page in range(3):
        req = urllib.request.Request(
            url, headers={"Accept": "application/vnd.github+json", "User-Agent": "ops-agent"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            releases = json.loads(resp.read())
            link_header = resp.headers.get("Link")
        if not isinstance(releases, list):
            raise Exception("GitHub releases response was not a list")
        tags.extend(rel.get("tag_name", "") for rel in releases if rel.get("tag_name"))
        next_url = None
        if link_header:
            for part in link_header.split(","):
                part = part.strip()
                if 'rel="next"' in part:
                    s, e = part.find("<"), part.find(">")
                    if s != -1 and e != -1:
                        next_url = part[s + 1 : e]
                    break
        if not next_url:
            break
        url = next_url
    return tags


def check_upstream_tag_update(host, container):
    """Unlike check_for_update (same pinned tag, new build only), this
    finds the highest upstream tag that's a newer version of the same
    image variant. Prefers GitHub's Releases API when the image carries
    an org.opencontainers.image.source label pointing at a GitHub repo -
    the clean, targeted source of truth for actual releases - falling
    back to raw registry tag listing (Docker Hub / GHCR) only when
    there's no such label. Read-only - never pulls or edits anything;
    call check_upstream_tag_changelog next on any candidate before ever
    considering bump_compose_tag_and_apply."""
    if host == "qnap":
        return {"available": False, "note": "not implemented for qnap (no compose files there)"}
    info = _inspect(host, container)
    if not info:
        return {"error": "container not found"}
    image = info["Config"]["Image"]
    registry, repo, tag, digest = _parse_image_ref(image)
    if digest:
        return {"available": False, "note": f"{image} is pinned by digest, not a movable tag - skipping"}
    if not tag:
        return {"available": False, "note": f"{image} has no tag to compare against"}

    source = (info.get("Config", {}).get("Labels") or {}).get("org.opencontainers.image.source")
    newer_tag, method = None, None
    if source and "github.com" in source:
        owner_repo = source.split("github.com/")[-1].rstrip("/").removesuffix(".git")
        try:
            release_tags = _list_github_release_tags(owner_repo)
            newer_tag = _find_newer_tag(tag, release_tags)
            method = "github_releases"
        except Exception as e:
            log.warning(f"[{host}] {container}: GitHub releases lookup failed ({e}), falling back to registry tags")

    if not newer_tag:
        try:
            all_tags = _list_registry_tags(registry, repo)
        except Exception as e:
            return {"available": False, "note": f"tag lookup failed: {e}"}
        newer_tag = _find_newer_tag(tag, all_tags)
        method = "registry_tags" if newer_tag else method

    if not newer_tag:
        return {"available": False, "current_tag": tag, "note": "no newer matching-shape tag found upstream (checked GitHub releases and registry tags)"}

    new_image_ref = f"{registry + '/' if registry else ''}{repo}:{newer_tag}"
    return {"available": True, "current_tag": tag, "candidate_tag": newer_tag, "candidate_image": new_image_ref, "found_via": method}


def scan_container_updates():
    """Check every managed container for (a) a newer build under its pinned tag and (b) a newer upstream tag,
    in parallel, in plain Python. Returns ONLY the containers that have something available, so the model
    spends turns on decisions instead of 40+ one-container-per-turn checks."""
    snapshot = gather_fleet_snapshot()
    targets = []
    for host, containers in snapshot.items():
        if isinstance(containers, dict):
            continue
        for c in containers:
            if c.get("expected_always_running"):
                targets.append((host, c["name"]))

    def one(t):
        host, name = t
        out = {"host": host, "container": name}
        try:
            r = check_for_update(host, name)
            out["same_tag"] = r
            if not r.get("locally_built") and not r.get("error"):
                out["upstream_tag"] = check_upstream_tag_update(host, name)
        except Exception as e:
            out["error"] = str(e)
        return out

    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(one, targets))
    updates, errors, local = [], [], 0
    for r in results:
        st = r.get("same_tag", {})
        up = r.get("upstream_tag", {})
        if r.get("error") or st.get("error"):
            errors.append({"host": r["host"], "container": r["container"], "error": r.get("error") or st.get("error")})
        elif st.get("locally_built"):
            local += 1
        if st.get("update_available") or up.get("available"):
            updates.append({"host": r["host"], "container": r["container"],
                            "new_build_same_tag": bool(st.get("update_available")),
                            "newer_tag": up if up.get("available") else None})
    return {"checked": len(targets), "locally_built_skipped": local, "with_updates": updates,
            "check_errors": errors[:10]}


def check_upstream_tag_changelog(host, container, candidate_image_ref):
    """Like check_changelog but compares the running image against an EXPLICIT
    candidate image:tag (e.g. "nextcloud:33") that may differ from the
    container's current image tag, rather than a same-tag re-pull."""
    if host == "qnap":
        return {"available": False, "note": "changelog lookup not implemented for qnap (no docker CLI there)"}
    info = _inspect(host, container)
    if not info:
        return {"available": False, "note": "container not found"}
    current_labels = info.get("Config", {}).get("Labels") or {}
    current_version = current_labels.get("org.opencontainers.image.version")
    current_revision = current_labels.get("org.opencontainers.image.revision") or current_labels.get("git_commit")
    source = current_labels.get("org.opencontainers.image.source")

    rc, out, err = docker_cli(host, f"pull {shlex.quote(candidate_image_ref)}", timeout=180)
    if rc != 0:
        return {"available": False, "note": f"could not pull candidate image {candidate_image_ref}: {err}"}

    rc, out, err = docker_cli(host, f"inspect --format '{{{{json .Config.Labels}}}}' {shlex.quote(candidate_image_ref)}")
    try:
        new_labels = json.loads(out.strip()) or {}
    except (json.JSONDecodeError, AttributeError):
        new_labels = {}
    new_version = new_labels.get("org.opencontainers.image.version")
    new_revision = new_labels.get("org.opencontainers.image.revision") or new_labels.get("git_commit")
    source = source or new_labels.get("org.opencontainers.image.source")

    owner_repo = source.split("github.com/")[-1].rstrip("/").removesuffix(".git") if source and "github.com" in source else None
    if not owner_repo:
        owner_repo = _guess_github_source(candidate_image_ref)

    if not owner_repo:
        return {"available": False, "note": f"no GitHub source label on image or {candidate_image_ref}, and no reliable source to guess from - no changelog signal available"}

    if current_version and new_version:
        if current_version == new_version:
            return {"available": False, "note": f"version label unchanged between running container and {candidate_image_ref} - no version-level changelog to check"}
        result = _github_releases_changelog(owner_repo, current_version, new_version)
        if result["available"]:
            result["candidate_image"] = candidate_image_ref
        return result

    if current_revision and new_revision:
        if current_revision == new_revision:
            return {"available": False, "note": f"revision label unchanged between running container and {candidate_image_ref} - no changelog to check"}
        result = _github_commit_changelog(owner_repo, current_revision, new_revision)
        if result["available"]:
            result["candidate_image"] = candidate_image_ref
        return result

    return {"available": False, "note": f"source found ({owner_repo}) but neither version nor revision labels present on running image or {candidate_image_ref} - can't bound a changelog range"}


def _find_compose_file(host, compose_dir):
    for name in ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"):
        rc, out, _ = run(host, f"test -f {shlex.quote(compose_dir)}/{name} && echo {name}")
        if rc == 0 and out.strip() == name:
            return f"{compose_dir}/{name}"
    return None


def _set_compose_image_tag(host, container, old_image_ref, new_image_ref):
    """Literal string substitution of the exact image reference in the
    service's compose file - intentionally not a YAML parse/rewrite, to
    avoid reformatting comments or unrelated structure. Safe because
    old_image_ref is the EXACT string docker resolved for this
    container, so a plain sed match is unambiguous. Uses | as the sed
    delimiter since image refs always contain / but never |."""
    compose_dir = COMPOSE_DIRS.get((host, container))
    if not compose_dir:
        return {"success": False, "error": "no compose_dir configured for this container"}
    compose_file = _find_compose_file(host, compose_dir)
    if not compose_file:
        return {"success": False, "error": f"no compose file found in {compose_dir}"}
    sed_expr = f"s|{old_image_ref}|{new_image_ref}|"
    rc, out, err = run(host, f"sed -i {shlex.quote(sed_expr)} {shlex.quote(compose_file)}")
    if rc != 0:
        return {"success": False, "error": f"sed failed: {err}"}
    return {"success": True, "compose_file": compose_file}


def bump_compose_tag_and_apply(host, container, candidate_tag):
    """For a compose-managed container: edits the pinned tag in its compose
    file to candidate_tag (from check_upstream_tag_update), then runs the
    exact same backup+recreate+health-check+rollback flow as apply_update.
    If that doesn't end in success=True, reverts the compose file edit back
    to the original tag - a failed/rolled-back attempt must never leave
    the compose file pointing at a tag that isn't actually what's
    running, or the next manual `docker compose up` / host reboot would
    silently reapply the bad tag.

    For a non-compose container (no compose_dir configured - e.g.
    portainer_agent, confirmed real: it was started via plain `docker run`,
    not compose, on the workstation), there's no compose file to edit, so
    this instead passes the new image straight to apply_update's
    target_image param, which recreates via _generic_recreate the same way
    a compose container's bump does via its compose file - same backup/
    health-check/rollback safety net either way, just no file to revert on
    failure since none was ever edited.

    Only ever call this after reading an actual changelog and judging it
    safe - never as a blind next step after check_upstream_tag_update
    alone."""
    info = _inspect(host, container)
    if not info:
        return {"success": False, "error": "container not found"}
    old_image = info["Config"]["Image"]
    registry, repo, old_tag, digest = _parse_image_ref(old_image)
    if digest:
        return {"success": False, "error": f"{old_image} is pinned by digest - can't tag-bump"}
    new_image = f"{registry + '/' if registry else ''}{repo}:{candidate_tag}"

    compose_dir = COMPOSE_DIRS.get((host, container))
    if not compose_dir:
        result = apply_update(host, container, target_image=new_image)
        outcome = "success" if result.get("success") else f"failed (rolled_back={result.get('rolled_back')})"
        _log_decision("bump_compose_tag_and_apply", f"{host}/{container}", f"tag bump {old_tag} -> {candidate_tag} (non-compose)", outcome)
        result["old_tag"] = old_tag
        result["new_tag"] = candidate_tag
        return result

    edit_result = _set_compose_image_tag(host, container, old_image, new_image)
    if not edit_result.get("success"):
        return {"success": False, "stage": "compose_edit", "error": edit_result.get("error")}

    result = apply_update(host, container)

    if not result.get("success"):
        revert_result = _set_compose_image_tag(host, container, new_image, old_image)
        result["compose_reverted"] = revert_result.get("success", False)
        _log_decision("bump_compose_tag_and_apply", f"{host}/{container}", f"tag bump {old_tag} -> {candidate_tag} failed, compose reverted", "failed")
    else:
        _log_decision("bump_compose_tag_and_apply", f"{host}/{container}", f"tag bump {old_tag} -> {candidate_tag}", "success")

    result["old_tag"] = old_tag
    result["new_tag"] = candidate_tag
    return result


def _parse_size_to_bytes(s):
    units = {"TiB": 1024**4, "GiB": 1024**3, "MiB": 1024**2, "KiB": 1024, "B": 1,
             "TB": 1000**4, "GB": 1000**3, "MB": 1000**2, "KB": 1000}
    s = s.strip()
    for unit in sorted(units, key=len, reverse=True):
        if s.endswith(unit):
            try:
                return int(float(s[:-len(unit)]) * units[unit])
            except ValueError:
                return None
    return None


def _host_container_mem_snapshot(host):
    """{container_name: mem_bytes_used} for every running container on host."""
    if host == "qnap":
        result = {}
        listing = list_containers(host)
        if isinstance(listing, dict) and "error" in listing:
            return {}
        for c in listing:
            name = c.get("name")
            rc, out, err = qnap_docker_api("GET", f"/containers/{name}/stats?stream=false")
            try:
                result[name] = json.loads(out)["memory_stats"].get("usage")
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        return result
    rc, out, err = docker_cli(host, "stats --no-stream --format '{{.Name}}\t{{.MemUsage}}'")
    if rc != 0:
        return {}
    result = {}
    for line in out.strip().splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        name, mem = parts
        b = _parse_size_to_bytes(mem.split("/")[0])
        if b is not None:
            result[name] = b
    return result


def check_host_resource_trend(host):
    """Deterministic and notify-only, same philosophy as check_resource_limits:
    compares each container's current memory footprint against a snapshot from
    ~7+ days ago (persisted in state.json) and flags containers that grew past
    RESOURCE_GROWTH_ALERT_PCT, plus whether the host's total crossed its
    optional configured ceiling (host_resource_ceiling_gb in config - unset by
    default, so no ceiling is enforced unless you configure one). This never
    moves, resizes, or restarts anything. A flagged container is a migration
    CANDIDATE for the model to reason about and recommend via notify_human -
    there is no tool that lets it act on this itself, migrating a service is a
    human-executed action (stop, copy volume data, restart elsewhere, retarget
    the NPM proxy_host entry) on purpose, see system prompt."""
    snapshot = _host_container_mem_snapshot(host)
    if not snapshot:
        return {"host": host, "flags": [], "note": "no data"}

    state = load_state()
    history = state.setdefault("resource_history", {}).setdefault(host, {})
    now = datetime.now(timezone.utc)
    flags = []

    for name, mem_bytes in snapshot.items():
        past = history.get(name)
        if past and past.get("mem_bytes"):
            days = (now - datetime.fromisoformat(past["date"])).days
            if days >= 6:
                growth = (mem_bytes - past["mem_bytes"]) / past["mem_bytes"]
                if growth > RESOURCE_GROWTH_ALERT_PCT:
                    flags.append({
                        "container": name, "mem_bytes_now": mem_bytes,
                        "mem_bytes_previous": past["mem_bytes"],
                        "growth_pct": round(growth * 100, 1), "days_span": days,
                    })
                history[name] = {"date": now.isoformat(), "mem_bytes": mem_bytes}
        else:
            history[name] = {"date": now.isoformat(), "mem_bytes": mem_bytes}

    save_state(state)

    total_bytes = sum(snapshot.values())
    result = {"host": host, "total_mem_bytes": total_bytes, "flags": flags}
    ceiling_gb = HOST_RESOURCE_CEILING_GB.get(host)
    if ceiling_gb and total_bytes > ceiling_gb * 1024**3:
        result["ceiling_exceeded"] = True
        result["ceiling_bytes"] = ceiling_gb * 1024**3
    return result


def _dir_size_bytes(host, path):
    rc, out, _ = run(host, f"du -sb {shlex.quote(path)} 2>/dev/null | cut -f1")
    try:
        return int(out.strip())
    except (ValueError, AttributeError):
        return None


def backup_container_data(host, container):
    """Tar up the container's bind-mount data directories to a local
    timestamped backup, skipping any mount too large to snapshot quickly.
    Returns a manifest of what was (and wasn't) backed up."""
    info = _inspect(host, container)
    if not info:
        return {"error": "container not found"}

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backed_up, skipped = [], []

    for mount in info.get("Mounts", []):
        if mount.get("Type") != "bind":
            continue
        src = mount["Source"]
        size = _dir_size_bytes(host, src)
        backup_root = BACKUP_DIR_FOR_HOST.get(host, str(BACKUP_DIR))
        dest_dir = f"{backup_root}/{host}/{container}"
        dest_file = f"{dest_dir}/{ts}_{os.path.basename(src.rstrip('/'))}.tar.gz"

        if size is not None and size > DATA_BACKUP_SIZE_LIMIT_BYTES:
            skipped.append({"path": src, "size_bytes": size, "reason": "too large to snapshot inline"})
            log.warning(f"[{host}] {container}: skipping backup of {src} ({size/1e9:.1f}GB, over limit) - relying on rollback-tag + volume-preserving update instead")
            continue

        mkdir_cmd = f"mkdir -p {shlex.quote(dest_dir)}"
        tar_cmd = f"tar czf {shlex.quote(dest_file)} -C {shlex.quote(os.path.dirname(src.rstrip('/')))} {shlex.quote(os.path.basename(src.rstrip('/')))}"
        if host == "docker-lxc" or host == "npm-lxc":
            rc1, _, e1 = run(host, mkdir_cmd)
            rc2, _, e2 = run(host, tar_cmd, timeout=300)
        else:
            rc1, _, e1 = run(host, mkdir_cmd)
            rc2, _, e2 = run(host, tar_cmd, timeout=300)

        if rc1 == 0 and rc2 == 0:
            backed_up.append({"path": src, "backup_file": dest_file})
            log.info(f"[{host}] {container}: backed up {src} -> {dest_file}")
        else:
            skipped.append({"path": src, "reason": f"tar failed: {e1 or ''} {e2 or ''}"})

    return {"timestamp": ts, "backed_up": backed_up, "skipped": skipped}


def _tag_rollback_image(host, image):
    """Tag the currently-running image as :ops-agent-rollback so we always
    have a stable local reference to revert to."""
    repo = image.split(":")[0].split("@")[0]
    rollback_tag = f"{repo}:ops-agent-rollback"
    if host == "qnap":
        qnap_docker_api("POST", f"/images/{image}/tag?repo={repo}&tag=ops-agent-rollback")
    else:
        docker_cli(host, f"tag {shlex.quote(image)} {shlex.quote(rollback_tag)}")
    return rollback_tag


def _container_healthy(host, container, wait_seconds=HEALTH_WAIT_SECONDS):
    """Poll a container's health for up to wait_seconds. If it has a
    HEALTHCHECK, wait for 'healthy'. Otherwise just require it to stay
    'running' for the whole window (a crash-loop will show as restarting)."""
    deadline = time.time() + wait_seconds
    has_healthcheck = None
    while time.time() < deadline:
        info = _inspect(host, container)
        if not info:
            time.sleep(HEALTH_POLL_INTERVAL)
            continue
        state = info.get("State", {})
        if has_healthcheck is None:
            has_healthcheck = "Health" in state
        if has_healthcheck:
            status = state["Health"]["Status"]
            if status == "healthy":
                return True
            if status == "unhealthy":
                return False
        else:
            if not state.get("Running") or state.get("Restarting"):
                return False
            restart_count = info.get("RestartCount", 0)
            if restart_count and restart_count > 0:
                return False
        time.sleep(HEALTH_POLL_INTERVAL)
    if has_healthcheck:
        return False
    info = _inspect(host, container)
    st = (info or {}).get("State", {})
    return bool(info and st.get("Running") and not st.get("Restarting"))


def apply_update(host, container, target_image=None):
    """The safety-critical path: back up data (if applicable), pull +
    recreate via compose (preferred) or generic inspect-based recreate,
    verify health, and automatically roll back to the pre-update image
    (plus restored data backup) if it doesn't come up healthy.

    target_image (optional): recreate with this exact image:tag instead of
    re-pulling the container's own currently-pinned tag. Only meaningful
    for non-compose containers (the compose path always gets its target
    image from the compose file itself, edited by bump_compose_tag_and_apply
    before calling this) - lets a non-compose container's pinned tag be
    bumped via _generic_recreate the same way a compose one is bumped via
    editing its compose file, instead of that path being compose-only."""
    info = _inspect(host, container)
    if not info:
        return {"success": False, "error": "container not found"}

    current_image = info["Config"]["Image"]
    rollback_tag = _tag_rollback_image(host, current_image)

    backup_result = None
    if container in DATA_CONTAINERS.get(host, set()):
        backup_result = backup_container_data(host, container)

    target = _compose_target(host, container, info)
    if target:
        compose_dir, prefix, service_name = target
        rc, out, err = run(host, f"cd {shlex.quote(compose_dir)} && {prefix} pull {shlex.quote(service_name)} 2>&1 && {prefix} up -d {shlex.quote(service_name)} 2>&1", timeout=300)
        recreate_ok = rc == 0
        recreate_log = out + err
    else:
        recreate_ok, recreate_log = _generic_recreate(host, container, info, target_image=target_image)

    if not recreate_ok:
        log.error(f"[{host}] {container}: recreate failed: {recreate_log}")
        return {"success": False, "stage": "recreate", "log": recreate_log, "backup": backup_result}

    healthy = _container_healthy(host, container)
    if healthy:
        _discard_preserved(host, container)
        log.info(f"[{host}] {container}: update applied and healthy")
        _log_decision("apply_update", f"{host}/{container}", "image update applied, health check passed", "success")
        return {"success": True, "backup": backup_result, "rolled_back": False}

    log.warning(f"[{host}] {container}: unhealthy after update, rolling back to {rollback_tag}")
    rollback_ok, rollback_log = _rollback(host, container, rollback_tag, info, None)
    if rollback_ok:
        # a rollback only counts if the restored container is actually healthy, not merely started
        rollback_ok = _container_healthy(host, container, wait_seconds=45)
        if not rollback_ok:
            rollback_log += " | restored container is NOT healthy"
    if rollback_ok:
        _discard_preserved(host, container)
    else:
        notify_human(f"ROLLBACK FAILED: {container} on {host}",
                     f"{container} failed its update health check and the automatic rollback did not produce a healthy container. "
                     f"The previous container (if any) is preserved as '{container}.ops-old'. Manual attention needed. Details: {rollback_log[-400:]}",
                     urgent=True, topic=f"rollback-failed:{host}/{container}")
    _log_decision("apply_update", f"{host}/{container}", f"unhealthy after update, rolled back to {rollback_tag}", f"failed (rolled_back={rollback_ok})")
    return {
        "success": False,
        "stage": "health_check",
        "backup": backup_result,
        "rolled_back": rollback_ok,
        "rollback_log": rollback_log,
    }


def _generic_recreate(host, container, info, target_image=None):
    """For the handful of non-compose containers (portainer_agent), pull
    the new image and recreate with the same mounts/ports/env/restart
    policy extracted from the running container's own inspect data.
    target_image lets this pull/recreate a DIFFERENT tag than the one
    currently running (a tag bump) instead of just refreshing the same
    tag - see apply_update's target_image docstring."""
    image = target_image or info["Config"]["Image"]
    if host == "qnap":
        rc, _, err = qnap_docker_api("POST", f"/images/create?fromImage={image.split(':')[0]}&tag={image.split(':')[1] if ':' in image else 'latest'}", timeout=180)
    else:
        rc, _, err = docker_cli(host, f"pull {shlex.quote(image)}", timeout=180)
    if rc != 0:
        return False, f"pull failed: {err}"
    return _recreate_from_inspect(host, info, image)


def _compose_target(host, container, info):
    """(workdir, 'docker compose ...' prefix, service) if this container is managed by docker
    compose, else None. The ops-agent-config.yaml compose_dirs entry wins (needed for Portainer
    stacks whose label paths are container-internal); otherwise the container's own compose
    labels are used, so a compose container can never silently fall into the generic recreate
    path (which rebuilt wyoming-kokoro without its command and network on 2026-10-16)."""
    if host == "qnap":
        return None
    labels = (info.get("Config", {}) or {}).get("Labels", {}) or {}
    service = labels.get("com.docker.compose.service")
    cfg_dir = COMPOSE_DIRS.get((host, container))
    if cfg_dir:
        return cfg_dir, "docker compose", service or container.split("/")[-1]
    workdir = labels.get("com.docker.compose.project.working_dir")
    files = [f for f in (labels.get("com.docker.compose.project.config_files") or "").split(",") if f]
    project = labels.get("com.docker.compose.project")
    if not (service and workdir and files and project):
        return None
    checks = " && ".join(f"test -f {shlex.quote(f)}" for f in files)
    rc, _, _ = run(host, f"test -d {shlex.quote(workdir)} && {checks}")
    if rc != 0:
        return None
    flags = " ".join(f"-f {shlex.quote(f)}" for f in files)
    return workdir, f"docker compose -p {shlex.quote(project)} {flags}", service


def _discard_preserved(host, container):
    """Remove the pre-update container that _recreate_from_inspect keeps as '<name>.ops-old'."""
    if host == "qnap":
        return
    docker_cli(host, f"rm -f {shlex.quote(container + '.ops-old')} 2>/dev/null")


def _image_defaults(host, image_id):
    rc, out, _ = docker_cli(host, f"image inspect {shlex.quote(image_id)} --format '{{{{json .Config}}}}'")
    try:
        cfg = json.loads(out)
        return cfg.get("Entrypoint"), cfg.get("Cmd")
    except Exception:
        return None, None


def _docker_create_args(host, info, image):
    """Build a faithful `docker create` argument list from `docker inspect` data: command,
    entrypoint (only when they differ from the old image's defaults, so a new image's own
    defaults still apply), networks + aliases, ports, mounts, env, labels and the runtime
    options that matter (devices, GPUs, caps, privileged, extra hosts, limits, healthcheck...)."""
    cfg = info.get("Config", {}) or {}
    hc = info.get("HostConfig", {}) or {}
    name = info["Name"].lstrip("/")
    a = ["--name", name]
    rp = hc.get("RestartPolicy", {}) or {}
    pol = rp.get("Name") or "unless-stopped"
    if pol == "on-failure" and rp.get("MaximumRetryCount"):
        pol += f":{rp['MaximumRetryCount']}"
    a += ["--restart", pol]
    if hc.get("Privileged"): a.append("--privileged")
    if cfg.get("User"): a += ["--user", cfg["User"]]
    if cfg.get("WorkingDir"): a += ["--workdir", cfg["WorkingDir"]]
    netmode = hc.get("NetworkMode") or "default"
    host_like = netmode == "host" or netmode == "none" or netmode.startswith("container:")
    if cfg.get("Hostname") and not host_like and cfg["Hostname"] != info.get("Id", "")[:12]:
        a += ["--hostname", cfg["Hostname"]]
    for k, v in (cfg.get("Labels") or {}).items():
        a += ["--label", f"{k}={v}"]
    for e in cfg.get("Env") or []:
        a += ["-e", e]
    for m in info.get("Mounts", []) or []:
        mode = "rw" if m.get("RW") else "ro"
        if m.get("Type") == "bind":
            a += ["-v", f"{m['Source']}:{m['Destination']}:{mode}"]
        elif m.get("Type") == "volume":
            a += ["-v", f"{m['Name']}:{m['Destination']}:{mode}"]
    for dst, opts in (hc.get("Tmpfs") or {}).items():
        a += ["--tmpfs", f"{dst}:{opts}" if opts else dst]
    # networking
    nets = (info.get("NetworkSettings", {}) or {}).get("Networks", {}) or {}
    if host_like:
        a += ["--network", netmode]
    else:
        primary = "bridge" if netmode == "default" else netmode
        a += ["--network", primary]
        pn = nets.get(primary, {})
        short = info.get("Id", "")[:12]
        for al in pn.get("Aliases") or []:
            if al != short:
                a += ["--network-alias", al]
        ip = (pn.get("IPAMConfig") or {}).get("IPv4Address")
        if ip: a += ["--ip", ip]
        for container_port, bindings in (hc.get("PortBindings") or {}).items():
            for b in bindings or []:
                hip = b.get("HostIp") or ""
                hp = b.get("HostPort") or ""
                a += ["-p", ":".join(x for x in (hip, hp, container_port) if x != "") if hp else container_port]
    for h in hc.get("ExtraHosts") or []: a += ["--add-host", h]
    for d in hc.get("Dns") or []: a += ["--dns", d]
    for d in hc.get("Devices") or []:
        a += ["--device", f"{d['PathOnHost']}:{d['PathInContainer']}:{d.get('CgroupPermissions', 'rwm')}"]
    for r in hc.get("DeviceRequests") or []:
        if "gpu" in (r.get("Capabilities") or [[]])[0] or r.get("Driver") in ("nvidia", ""):
            a += ["--gpus", "all" if r.get("Count", 0) == -1 else f"count={r.get('Count', 1)}"]
    for c in hc.get("CapAdd") or []: a += ["--cap-add", c]
    for c in hc.get("CapDrop") or []: a += ["--cap-drop", c]
    for s in hc.get("SecurityOpt") or []: a += ["--security-opt", s]
    for g in hc.get("GroupAdd") or []: a += ["--group-add", g]
    for k, v in (hc.get("Sysctls") or {}).items(): a += ["--sysctl", f"{k}={v}"]
    for u in hc.get("Ulimits") or []: a += ["--ulimit", f"{u['Name']}={u['Soft']}:{u['Hard']}"]
    if hc.get("ShmSize") and hc["ShmSize"] != 67108864: a += ["--shm-size", str(hc["ShmSize"])]
    if hc.get("Memory"): a += ["--memory", str(hc["Memory"])]
    if hc.get("NanoCpus"): a += ["--cpus", str(hc["NanoCpus"] / 1e9)]
    if hc.get("PidsLimit"): a += ["--pids-limit", str(hc["PidsLimit"])]
    if hc.get("PidMode"): a += ["--pid", hc["PidMode"]]
    if hc.get("IpcMode") and hc["IpcMode"] not in ("private", "shareable", ""): a += ["--ipc", hc["IpcMode"]]
    if hc.get("ReadonlyRootfs"): a.append("--read-only")
    if hc.get("Init"): a.append("--init")
    lc = hc.get("LogConfig") or {}
    if lc.get("Type") and lc["Type"] != "json-file": a += ["--log-driver", lc["Type"]]
    for k, v in (lc.get("Config") or {}).items(): a += ["--log-opt", f"{k}={v}"]
    if cfg.get("StopSignal"): a += ["--stop-signal", cfg["StopSignal"]]
    h = cfg.get("Healthcheck") or {}
    if h.get("Test"):
        t = h["Test"]
        if t == ["NONE"]:
            a.append("--no-healthcheck")
        else:
            a += ["--health-cmd", t[1] if t[0] == "CMD-SHELL" and len(t) > 1 else " ".join(t[1:])]
            for flag, key in (("--health-interval", "Interval"), ("--health-timeout", "Timeout"), ("--health-start-period", "StartPeriod")):
                if h.get(key): a += [flag, f"{h[key]}ns"]
            if h.get("Retries"): a += ["--health-retries", str(h["Retries"])]
    # command / entrypoint: only pin what differs from the old image's defaults
    img_ep, img_cmd = _image_defaults(host, info.get("Image", ""))
    ep, cmd = cfg.get("Entrypoint"), cfg.get("Cmd")
    tail = []
    if ep != img_ep and ep is not None:
        a += ["--entrypoint", ep[0] if ep else ""]
        tail += list(ep[1:])
    if cmd != img_cmd and cmd is not None:
        tail += list(cmd)
    elif ep != img_ep and ep is not None and cmd:
        tail += list(cmd)
    return a, tail, nets, host_like


def _parity_problems(old, new):
    """Compare the fields whose loss breaks a container; returns a list of differences."""
    problems = []
    oc, nc = old.get("Config", {}) or {}, new.get("Config", {}) or {}
    oh, nh = old.get("HostConfig", {}) or {}, new.get("HostConfig", {}) or {}
    def eff(i):
        return (i.get("Config", {}).get("Entrypoint") or []) + (i.get("Config", {}).get("Cmd") or [])
    # command is only compared when the image is unchanged (a new image may properly change defaults)
    if old.get("Image") == new.get("Image") and eff(old) != eff(new):
        problems.append(f"command {eff(old)} -> {eff(new)}")
    if (oh.get("NetworkMode") or "default") != (nh.get("NetworkMode") or "default"):
        problems.append(f"network mode {oh.get('NetworkMode')} -> {nh.get('NetworkMode')}")
    onets = set(((old.get("NetworkSettings") or {}).get("Networks") or {}).keys())
    nnets = set(((new.get("NetworkSettings") or {}).get("Networks") or {}).keys())
    if onets != nnets:
        problems.append(f"networks {sorted(onets)} -> {sorted(nnets)}")
    if sorted((oh.get("PortBindings") or {}).keys()) != sorted((nh.get("PortBindings") or {}).keys()):
        problems.append("published ports differ")
    om = sorted((m.get("Destination"), m.get("Source") or m.get("Name")) for m in old.get("Mounts", []) or [] if m.get("Type") in ("bind", "volume"))
    nm = sorted((m.get("Destination"), m.get("Source") or m.get("Name")) for m in new.get("Mounts", []) or [] if m.get("Type") in ("bind", "volume"))
    if om != nm:
        problems.append("mounts differ")
    if bool(oh.get("Privileged")) != bool(nh.get("Privileged")):
        problems.append("privileged differs")
    if sorted(oh.get("CapAdd") or []) != sorted(nh.get("CapAdd") or []):
        problems.append("cap_add differs")
    if sorted(d.get("PathOnHost") for d in oh.get("Devices") or []) != sorted(d.get("PathOnHost") for d in nh.get("Devices") or []):
        problems.append("devices differ")
    if len(oh.get("DeviceRequests") or []) != len(nh.get("DeviceRequests") or []):
        problems.append("GPU/device requests differ")
    if sorted(oh.get("ExtraHosts") or []) != sorted(nh.get("ExtraHosts") or []):
        problems.append("extra hosts differ")
    if (oc.get("User") or "") != (nc.get("User") or ""):
        problems.append("user differs")
    return problems


def _recreate_from_inspect(host, info, image):
    """Re-create a container from a prior `docker inspect`, targeting the given image (must exist
    locally). The old container is stopped and RENAMED to '<name>.ops-old' (not deleted), the new
    one is created with a faithful copy of its configuration, and a parity check compares the
    result with the original. On any failure the old container is put back untouched, so a bad
    recreate can no longer leave the service broken (it used to drop command/network/devices).
    _rollback restores the preserved container directly; apply_update removes it after a healthy
    update."""
    name = info["Name"].lstrip("/")
    restart_policy = info.get("HostConfig", {}).get("RestartPolicy", {}).get("Name") or "unless-stopped"

    if host == "qnap":
        binds = []
        for m in info.get("Mounts", []):
            if m.get("Type") == "bind":
                binds.append(f"{m['Source']}:{m['Destination']}:{'rw' if m.get('RW') else 'ro'}")
            elif m.get("Type") == "volume":
                binds.append(f"{m['Name']}:{m['Destination']}:{'rw' if m.get('RW') else 'ro'}")
        port_bindings = info.get("HostConfig", {}).get("PortBindings") or {}
        exposed_ports = {p: {} for p in port_bindings}
        qnap_docker_api("POST", f"/containers/{name}/stop")
        qnap_docker_api("DELETE", f"/containers/{name}")
        body = {
            "Image": image,
            "Env": info["Config"].get("Env", []),
            "ExposedPorts": exposed_ports,
            "Labels": info["Config"].get("Labels") or {},
            "HostConfig": {
                "Binds": binds,
                "PortBindings": port_bindings,
                "RestartPolicy": {"Name": restart_policy},
                "NetworkMode": info.get("HostConfig", {}).get("NetworkMode") or "default",
            },
        }
        if info["Config"].get("Cmd"): body["Cmd"] = info["Config"]["Cmd"]
        if info["Config"].get("Entrypoint"): body["Entrypoint"] = info["Config"]["Entrypoint"]
        rc, out, err = qnap_docker_api("POST", f"/containers/create?name={name}", data=body)
        if rc != 0:
            return False, f"create failed: {out} {err}"
        try:
            new_id = json.loads(out)["Id"]
        except (json.JSONDecodeError, KeyError):
            return False, f"create returned unexpected response: {out}"
        rc, out2, err2 = qnap_docker_api("POST", f"/containers/{new_id}/start")
        return rc == 0, out + out2 + err2

    old_name = f"{name}.ops-old"
    args, tail, nets, host_like = _docker_create_args(host, info, image)
    primary = (info.get("HostConfig", {}).get("NetworkMode") or "default")
    primary = "bridge" if primary == "default" else primary
    extra_nets = [] if host_like else [n for n in nets if n != primary]

    steps = [
        f"docker rm -f {shlex.quote(old_name)} >/dev/null 2>&1",
        f"docker stop {shlex.quote(name)} >/dev/null",
        f"docker rename {shlex.quote(name)} {shlex.quote(old_name)}",
        "docker create " + " ".join(shlex.quote(x) for x in args) + f" {shlex.quote(image)} " + " ".join(shlex.quote(x) for x in tail),
    ]
    for n in extra_nets:
        al = [x for x in (nets[n].get("Aliases") or []) if x != info.get("Id", "")[:12]]
        steps.append(f"docker network connect {' '.join('--alias ' + shlex.quote(x) for x in al)} {shlex.quote(n)} {shlex.quote(name)}")
    steps.append(f"docker start {shlex.quote(name)}")
    script = "set -e; " + "; ".join(steps)
    rc, out, err = run(host, f"bash -c {shlex.quote(script)}", timeout=180)
    restore = (f"docker rm -f {shlex.quote(name)} >/dev/null 2>&1; docker rename {shlex.quote(old_name)} {shlex.quote(name)} && docker start {shlex.quote(name)}")
    if rc != 0:
        run(host, restore, timeout=120)
        return False, f"recreate failed, original container restored: {out} {err}"
    new_info = _inspect(host, name)
    problems = _parity_problems(info, new_info) if new_info else ["new container not found"]
    if problems:
        run(host, restore, timeout=120)
        return False, "recreated container differs from the original (" + "; ".join(problems) + "); original restored"
    return True, out + err


def _rollback(host, container, rollback_tag, original_info, compose_dir):
    """Restore the pre-update state. Order of preference:
    1. the preserved original container '<name>.ops-old' (exact original configuration);
    2. compose: re-tag the saved image and force-recreate the service from its compose file
       (only when the compose file still resolves to the original image name);
    3. faithful recreate from the captured inspect data.
    Re-tags the rollback image onto the container's real image name first so the next cycle's
    check_for_update still resolves against the correct upstream repo:tag."""
    original_image = original_info["Config"]["Image"]
    if host == "qnap":
        repo = original_image.split(":")[0].split("@")[0]
        tag = original_image.split(":")[1] if ":" in original_image else "latest"
        qnap_docker_api("POST", f"/images/{rollback_tag}/tag?repo={repo}&tag={tag}")
        return _recreate_from_inspect(host, original_info, original_image)
    docker_cli(host, f"tag {shlex.quote(rollback_tag)} {shlex.quote(original_image)}")
    old_name = f"{container}.ops-old"
    rc, _, _ = docker_cli(host, f"inspect {shlex.quote(old_name)} >/dev/null 2>&1")
    if rc == 0:
        cmd = (f"docker rm -f {shlex.quote(container)} >/dev/null 2>&1; docker rename {shlex.quote(old_name)} {shlex.quote(container)} "
               f"&& docker start {shlex.quote(container)}")
        rc, out, err = run(host, cmd, timeout=120)
        if rc == 0:
            return True, "restored the preserved original container " + out + err
    target = _compose_target(host, container, original_info)
    if target:
        wd, prefix, svc = target
        rc, out, err = run(host, f"cd {shlex.quote(wd)} && {prefix} config --images 2>/dev/null", timeout=60)
        if rc == 0 and original_image in out.split():
            rc, out, err = run(host, f"cd {shlex.quote(wd)} && {prefix} up -d --no-deps --force-recreate --pull never {shlex.quote(svc)} 2>&1", timeout=240)
            if rc == 0:
                return True, out + err
    return _recreate_from_inspect(host, original_info, original_image)


# hosts where the ssh/pct-exec user isn't already root and needs sudo
# for apt. Proxmox's own root ssh key was only just added to these
# accounts this deploy, so passwordless sudo for apt is a real
# prerequisite, not guaranteed - see NEEDS_MANUAL_SETUP at the bottom.
SUDO_FOR_APT_HOSTS = {"workstation", "adsb"}


def _apt_prefix(host):
    return "sudo -n " if host in SUDO_FOR_APT_HOSTS else ""


def check_os_updates(host):
    if host not in OS_AUTO_HOSTS + OS_CHECK_ONLY_HOSTS:
        return {"error": "unsupported host"}
    sudo = _apt_prefix(host)
    rc, out, err = run(host, f"{sudo}apt-get update -qq 2>&1 && apt list --upgradable 2>/dev/null | grep -v '^Listing'", timeout=120)
    packages = [line.split("/")[0] for line in out.strip().splitlines() if line.strip()]
    reboot_rc, reboot_out, _ = run(host, "test -f /var/run/reboot-required && echo yes || echo no")
    return {"upgradable_count": len(packages), "packages": packages[:30], "reboot_required": reboot_out.strip() == "yes"}


def apply_os_updates(host):
    if host in NEVER_PATCH_HOSTS:
        return {"success": False, "error": f"{host} is never OS-patched by this agent (hard exclusion in code)"}
    if host not in OS_AUTO_HOSTS:
        return {"success": False, "error": f"{host} is check-only, not auto-apply (see module docstring)"}
    log.info(f"[{host}] applying OS updates")
    sudo = _apt_prefix(host)
    rc, out, err = run(host, f"{sudo}env DEBIAN_FRONTEND=noninteractive apt-get update -qq && {sudo}env DEBIAN_FRONTEND=noninteractive apt-get -y upgrade 2>&1", timeout=900)
    if rc != 0 and "password is required" in (out + err).lower():
        notify_human(
            f"⚠️ {host} OS updates blocked - no sudo access",
            f"The ops-agent account on {host} doesn't have passwordless sudo for apt yet. "
            f"One-time fix needed: add a sudoers.d rule granting NOPASSWD for apt-get/apt to that account.",
        )
        return {"success": False, "error": "sudo not available - see notification"}
    reboot_rc, reboot_out, _ = run(host, "test -f /var/run/reboot-required && echo yes || echo no")
    reboot_required = reboot_out.strip() == "yes"
    if reboot_required:
        notify_human(
            f"⚠️ {host} needs a reboot",
            f"OS updates were applied to {host} and a reboot is required to finish. "
            f"Scheduling it for 03:30 local time tonight.",
        )
        run(host, f"{sudo}shutdown -r 03:30 'ops-agent: scheduled reboot to finish OS updates' 2>&1 || true")
    _log_decision("apply_os_updates", host, "OS package update (apt/yum) run", "success" if rc == 0 else "failed")
    return {"success": rc == 0, "output": out[-2000:], "reboot_scheduled": reboot_required}


def check_ubuntu_lts_available(host):
    """Read-only: compares the host's current Ubuntu codename/version against
    the latest LTS release Canonical has published. NEVER triggers a release
    upgrade (do-release-upgrade rewrites the kernel, drivers, and - on the
    workstation specifically - the hand-tuned GPU stack; that's a far bigger
    blast radius than an apt-get upgrade and stays a human-only action,
    permanently). This only checks and notifies via check_only_os_hosts()'s
    sibling call in main() - not exposed as an LLM tool since there's no
    judgment call involved, purely mechanical like check_only_os_hosts()."""
    if host != "workstation":
        return {"error": "unsupported host - only workstation runs Ubuntu"}
    rc, out, err = run(host, "lsb_release -cs; lsb_release -rs")
    lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return {"error": f"couldn't read current release: {err or out}"}
    current_codename, current_version = lines[0], lines[1]
    try:
        with urllib.request.urlopen("https://changelogs.ubuntu.com/meta-release-lts", timeout=15) as resp:
            meta = resp.read().decode("utf-8", "replace")
    except Exception as e:
        return {"error": f"couldn't reach changelogs.ubuntu.com: {e}"}
    latest_codename = latest_version = None
    for block in meta.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if fields.get("Dist"):
            latest_codename, latest_version = fields["Dist"], fields.get("Version", "unknown")
    if not latest_codename:
        return {"error": "couldn't parse meta-release-lts feed"}
    return {
        "current_codename": current_codename,
        "current_version": current_version,
        "latest_lts_codename": latest_codename,
        "latest_lts_version": latest_version,
        "new_lts_available": latest_codename != current_codename,
    }


def _adsb_reachable():
    rc, out, _ = run("adsb", "echo ok", timeout=15)
    return rc == 0 and out.strip() == "ok"


def check_adsb_service_health():
    """Check the ADS-B Pi's core receiver services (dashboard depends on
    these) and secondary feed-out services. If the Pi itself can't be
    reached it says so once, instead of listing every service as down
    (which made the model retry restarts in a loop)."""
    if not ADSB_ENABLED:
        return {"disabled": True, "all_ok": True, "down": []}
    if not _adsb_reachable():
        return {"unreachable": True, "all_ok": False, "down": [],
                "note": "ADS-B Pi is not reachable over SSH; nothing can be restarted. Do not retry, just report it."}
    all_services = ADSB_CORE_SERVICES + ADSB_SECONDARY_SERVICES
    down = []
    for svc in all_services:
        rc, out, _ = run("adsb", f"systemctl is-active {shlex.quote(svc)}")
        if out.strip() != "active":
            down.append({"service": svc, "state": out.strip(), "core": svc in ADSB_CORE_SERVICES})
    return {"down": down, "all_ok": len(down) == 0}


def restart_adsb_service(service):
    if not ADSB_ENABLED:
        return {"success": False, "error": "ADS-B management is disabled"}
    if not _adsb_reachable():
        return {"success": False, "error": "ADS-B Pi unreachable over SSH - restart not possible, do not retry"}
    rc, out, err = run("adsb", f"sudo systemctl restart {shlex.quote(service)}")
    return {"success": rc == 0, "output": out, "error": err}


# ---------------------------------------------------- Home Assistant --
# HA is managed exclusively through its own REST/WebSocket API - never
# SSH into the HAOS VM itself, it's an appliance-style OS not meant for
# that. Its ha-backup-manager.timer/backup_manager.py cron (already
# deployed on Proxmox) is a SEPARATE daily job; the functions below
# trigger their own backup inline before an update rather than depending
# on that cron's timing; the same hassio.backup_full/poll pattern.

def _ha_request(method, path, body=None, timeout=30):
    req = urllib.request.Request(
        f"{HA_BASE}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else None


def check_ha_health():
    """Basic reachability check - if this fails, HA is down."""
    try:
        result = _ha_request("GET", "/api/", timeout=10)
        return {"reachable": True}
    except Exception as e:
        return {"reachable": False, "error": str(e)}


def check_ha_updates():
    """Enumerate update.* entities (Core, Supervisor, OS, per-add-on) and
    report which have a pending update (state 'on')."""
    try:
        states = _ha_request("GET", "/api/states", timeout=15)
    except Exception as e:
        return {"error": str(e)}
    pending = []
    for s in states:
        if s["entity_id"].startswith("update."):
            if s["state"] == "on":
                pending.append({
                    "entity_id": s["entity_id"],
                    "name": s["attributes"].get("friendly_name"),
                    "installed_version": s["attributes"].get("installed_version"),
                    "latest_version": s["attributes"].get("latest_version"),
                })
    return {"pending": pending}


# --- General HA visibility/control, beyond HA's own updates ---
#
# Read access (state/history/logbook) is unrestricted - any entity, no
# blast radius, this is exactly what got hand-rolled as a throwaway
# websocket script for every HA automation debugging session before this
# existed (lock sensors, a doorbell, lighting automations). ha_call_service is different: it can make something in
# the physical house actually happen, so it's restricted to
# HA_SAFE_SERVICE_DOMAINS - lights, switches, climate, media, scenes,
# scripts, helpers. Intentionally NOT included: locks, covers (garage
# doors), alarm_control_panel, or anything else security/safety-critical
# - those need a separate, explicit decision, not a default inclusion
# here just because the HA token technically has the scope to call them.

HA_SAFE_SERVICE_DOMAINS = {
    "light", "switch", "climate", "fan", "media_player",
    "input_boolean", "input_number", "input_select", "input_text",
    "timer", "notify",
}
# A script or a scene can do anything its author put in it, including unlocking a door, so these two domains
# are NOT in the safe set. They are callable only for entities listed under ha_allowed_entities in the config
# (empty by default), which you have read and know do nothing dangerous.
HA_RESTRICTED_DOMAINS = {"script", "scene"}
HA_ALLOWED_ENTITIES = set(_cfg.get("ha_allowed_entities", []))
# Keys that would aim a service call at something other than the one named entity.
HA_TARGET_KEYS = {"entity_id", "device_id", "area_id", "floor_id", "label_id"}


def ha_search_entities(query):
    """Find entity_ids by a case-insensitive substring match against the
    entity_id or friendly name - use this FIRST when unsure of an exact
    entity_id rather than guessing variations. Read-only, returns at most
    25 matches (id, current state, friendly name)."""
    try:
        states = _ha_request("GET", "/api/states", timeout=15)
    except Exception as e:
        return {"error": str(e)}
    q = query.lower()
    matches = []
    for s in states:
        name = s.get("attributes", {}).get("friendly_name", "") or ""
        if q in s["entity_id"].lower() or q in name.lower():
            matches.append({"entity_id": s["entity_id"], "state": s["state"], "name": name})
    return {"matches": matches[:25], "total_found": len(matches)}


def ha_get_state(entity_id):
    """Read one Home Assistant entity's current state and attributes.
    Read-only, any entity - no restriction, this can never change anything."""
    try:
        return _ha_request("GET", f"/api/states/{entity_id}", timeout=15)
    except Exception as e:
        return {"error": str(e)}


def ha_get_history(entity_id, hours=24):
    """Read an entity's state-change history over the last N hours (max
    168 = 1 week). Read-only. Use this to check whether something actually
    changed state when it should have, e.g. debugging an automation that
    may or may not have fired."""
    hours = min(hours, 168)
    start = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    try:
        result = _ha_request(
            "GET",
            f"/api/history/period/{start}?filter_entity_id={entity_id}&no_attributes=true",
            timeout=30,
        )
        return result[0] if result else []
    except Exception as e:
        return {"error": str(e)}


def ha_get_logbook(entity_id, hours=24):
    """Read Home Assistant logbook entries for an entity over the last N
    hours (max 168 = 1 week) - shows what actually triggered/happened,
    e.g. whether an automation fired. Read-only."""
    hours = min(hours, 168)
    start = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    try:
        return _ha_request("GET", f"/api/logbook/{start}?entity={entity_id}", timeout=30)
    except Exception as e:
        return {"error": str(e)}


def ha_call_service(domain, service, entity_id, data=None):
    """Call a Home Assistant service to actually control something - e.g.
    domain=light, service=turn_on, entity_id=light.kitchen. Restricted to
    HA_SAFE_SERVICE_DOMAINS - locks, covers/garage doors, and alarm panels
    are intentionally not callable this way."""
    if domain not in HA_SAFE_SERVICE_DOMAINS and domain not in HA_RESTRICTED_DOMAINS:
        return {"error": f"domain {domain!r} is not in the allowed set: {sorted(HA_SAFE_SERVICE_DOMAINS)}"}
    if not (isinstance(entity_id, str) and ENTITY_RE.fullmatch(entity_id) and entity_id.startswith(domain + ".")):
        return {"error": "entity_id must be one entity of the same domain as the service, for example light.kitchen"}
    if domain in HA_RESTRICTED_DOMAINS and entity_id not in HA_ALLOWED_ENTITIES:
        return {"error": f"{entity_id} is not listed in ha_allowed_entities. Scripts and scenes can reach anything, including locks, so each one must be allowed by name in the config"}
    if data and HA_TARGET_KEYS & set(data):
        return {"error": "data may not contain a target (entity_id, device_id, area_id, floor_id, label_id)"}
    body = {"entity_id": entity_id}
    if data:
        body.update(data)
    try:
        result = _ha_request("POST", f"/api/services/{domain}/{service}", body=body, timeout=15)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def check_automation_health(lookback_hours=2):
    """Report-only: for each ENABLED automation with state or numeric_state
    triggers, check whether any trigger entity actually crossed into
    trigger-relevant territory more recently than the automation's own
    last_triggered - if so, it may not be firing when it should. This
    can't tell a real miss apart from a condition properly blocking
    it (that looks identical from this signal alone) - it's a flag for a
    human/the model to weigh, not proof of a bug. This is exactly the
    pattern that let the doorbell popup automation silently miss most of
    a day's unlock events before anyone noticed - see
    a doorbell automation incident.
    numeric_state support (below/above threshold crossing, not just any
    value change) was added after a lighting automation turned
    out to be silently unwatched by this check entirely - it uses
    numeric_state, which the original state-only extraction skipped.
    One flag per automation (first miss found), so a chatty automation
    doesn't drown out everything else in one report.

    Also proactively runs check_sensor_value_stuck on every numeric_state
    trigger entity seen, independent of whether it crossed a threshold -
    a threshold-crossing check can never see a sensor frozen INSIDE the
    dead zone between below/above (no crossing to detect), which is
    exactly the failure mode that let a lighting automation go
    unwatched twice (2026-09-24 stuck at 237.0, 2026-09-27 stuck at 139.0,
    both inside its 80-150 dead zone). This used to be prompt-only
    guidance asking the model to remember to call check_sensor_value_stuck
    reactively when something else looked wrong - but a frozen mid-range
    sensor produces no other symptom to react to, so it was silently
    never called. Flagged here unconditionally instead."""
    try:
        states = _ha_request("GET", "/api/states", timeout=20)
    except Exception as e:
        return {"error": str(e)}

    automations = [s for s in states if s["entity_id"].startswith("automation.") and s["state"] == "on"]
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=lookback_hours)
    flags = []
    stuck_checked = {}

    for auto in automations:
        auto_id = auto["attributes"].get("id")
        last_triggered_str = auto["attributes"].get("last_triggered")
        if not auto_id:
            continue
        try:
            cfg = _ha_request("GET", f"/api/config/automation/config/{auto_id}", timeout=15)
        except Exception:
            continue

        # numeric_state carries the same entity_id shape as state (e.g. the
        # illuminance-threshold trigger on a lighting automation) -
        # confirmed missing here let that automation go completely unwatched,
        # regardless of whether it was actually working. Unlike a plain state
        # trigger, a numeric_state entity (illuminance) changes value
        # constantly without ever crossing below/above - flagging on ANY
        # change would be near-constant noise, so its below/above threshold
        # is tracked per entity and only a real crossing counts.
        specs_by_entity = {}
        for trig in cfg.get("triggers", []):
            ttype = trig.get("trigger")
            if ttype not in ("state", "numeric_state"):
                continue
            eids = trig.get("entity_id")
            if isinstance(eids, str):
                eids = [eids]
            for eid in (eids or []):
                specs_by_entity.setdefault(eid, []).append(
                    {"type": ttype, "below": trig.get("below"), "above": trig.get("above")}
                )
        if not specs_by_entity:
            continue

        for eid, specs in specs_by_entity.items():
            if not any(s["type"] == "numeric_state" for s in specs):
                continue
            if eid not in stuck_checked:
                stuck_checked[eid] = check_sensor_value_stuck(eid, hours=3)
            result = stuck_checked[eid]
            if result.get("stuck"):
                flags.append({
                    "automation": auto["entity_id"],
                    "name": auto["attributes"].get("friendly_name"),
                    "trigger_entity": eid,
                    "issue": "sensor_stuck",
                    "value": result.get("value"),
                    "unchanged_since": result.get("unchanged_since"),
                    "window_hours": 3,
                })

        last_triggered = None
        if last_triggered_str:
            try:
                last_triggered = datetime.fromisoformat(last_triggered_str)
            except ValueError:
                pass

        found = None
        for eid, specs in specs_by_entity.items():
            hist = ha_get_history(eid, hours=lookback_hours)
            if not isinstance(hist, list):
                continue
            for point in hist:
                changed_str = point.get("last_changed")
                if not changed_str:
                    continue
                try:
                    changed_at = datetime.fromisoformat(changed_str.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if changed_at < cutoff:
                    continue
                if last_triggered is not None and changed_at <= last_triggered:
                    continue

                relevant = False
                for spec in specs:
                    if spec["type"] == "state":
                        relevant = True
                        break
                    try:
                        value = float(point.get("state"))
                    except (TypeError, ValueError):
                        continue
                    if spec["below"] is not None and value < spec["below"]:
                        relevant = True
                        break
                    if spec["above"] is not None and value > spec["above"]:
                        relevant = True
                        break
                if not relevant:
                    continue

                found = {
                    "automation": auto["entity_id"],
                    "name": auto["attributes"].get("friendly_name"),
                    "trigger_entity": eid,
                    "issue": "missed_trigger",
                    "trigger_changed_at": changed_at.isoformat(),
                    "automation_last_triggered": last_triggered_str,
                }
                break
            if found:
                break
        if found:
            flags.append(found)

    return {"flags": flags, "automations_checked": len(automations)}


def check_sensor_value_stuck(entity_id, hours=3):
    """Report-only: is this sensor's value bit-for-bit IDENTICAL across the
    entire lookback window - a stuck/frozen device, not just an available
    one. Confirmed real gap on 2026-09-24: check_automation_health (and
    every cycle's own follow-up investigation) only ever checked whether a
    trigger sensor was 'available' (not unavailable/unknown), which a
    frozen sensor still reports as - it just never gets a NEW value.
    an illuminance sensor sat at exactly 237.0 lux for 3+ hours
    into the evening (should have been dropping toward darkness) and every
    cycle concluded 'sensor healthy, no outage' without ever checking
    this. Call this on a flagged automation's trigger sensor IN ADDITION
    to ha_get_history's availability check, not instead of it - a
    really stable reading (e.g. a thermostat setpoint, or a truly dark
    room overnight) can properly hold the same value for a while, so
    judge the result in context of what the sensor measures, same as any
    other flag here. Not a replacement for check_automation_health, a
    second signal to combine with it."""
    history = ha_get_history(entity_id, hours=hours)
    if isinstance(history, dict) and "error" in history:
        return history
    if not history:
        return {"stuck": False, "note": "no history available in this window"}
    values = {h.get("state") for h in history}
    if len(values) == 1:
        return {
            "stuck": True,
            "value": history[0].get("state"),
            "unchanged_since": history[0].get("last_changed"),
            "window_hours": hours,
            "note": f"identical value for the entire {hours}h window checked - may have been stuck even longer, "
                    f"widen hours to find out. Judge against what this sensor measures: suspicious for anything "
                    f"that normally fluctuates (illuminance, power draw, temperature), plausible for a really "
                    f"stable one (a setpoint, a binary state that properly hasn't changed).",
        }
    return {"stuck": False, "distinct_values_seen": len(values)}


def check_expected_automations_enabled():
    """Report-only: verify a configured set of automations are actually ON
    right now. Catches the silently-disabled case (HA restart that failed
    to reload, accidental toggle-off) which check_automation_health can't
    see because it only inspects automations already in the "on" state."""
    try:
        states = _ha_request("GET", "/api/states", timeout=20)
    except Exception as e:
        return {"error": str(e)}
    state_map = {s["entity_id"]: s for s in states}
    flags = []
    for entity_id in ALWAYS_ENABLED_AUTOMATIONS:
        s = state_map.get(entity_id)
        if s is None:
            flags.append({"entity_id": entity_id, "friendly_name": None, "state": "missing"})
        elif s["state"] != "on":
            flags.append({"entity_id": entity_id, "friendly_name": s.get("attributes", {}).get("friendly_name"), "state": s["state"]})
    return {"flags": flags, "checked": len(ALWAYS_ENABLED_AUTOMATIONS)}


def check_proxmox_infra_status():
    """Report-only: is every LXC/VM in ALWAYS_RUNNING_PROXMOX actually running
    right now? This is the hypervisor layer, one level below everything else
    this agent checks - a stopped LXC/VM is invisible to the container/HA
    checks, which just see that host as unreachable and can't tell why.
    Confirmed real gap from the 2026-09-18 power outage: LXC 102 (docker-lxc,
    hosting Prometheus/Grafana/ntfy/vaultwarden/etc.) had onboot=0 and stayed
    stopped until a human noticed and started it by hand - taking the whole
    monitoring stack and several services down silently in the meantime."""
    if not ALWAYS_RUNNING_PROXMOX:
        return {"flags": [], "checked": 0}
    rc, out, err = run("proxmox-host", "pct list; echo ---; qm list", timeout=30)
    if rc != 0:
        return {"error": err or out}
    # pct list columns: VMID Status Lock Name (status at index 1)
    # qm list columns:  VMID NAME Status Mem(MB) Bootdisk(GB) PID (status at index 2)
    # different tools, different column layouts - do not assume they match.
    running = {"lxc": set(), "qm": set()}
    section = "lxc"
    status_index = {"lxc": 1, "qm": 2}
    for line in out.splitlines():
        line = line.strip()
        if line == "---":
            section = "qm"
            continue
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        idx = status_index[section]
        if len(parts) > idx and parts[idx] == "running":
            running[section].add(int(parts[0]))
    flags = [e for e in ALWAYS_RUNNING_PROXMOX if e["vmid"] not in running[e["kind"]]]
    return {"flags": flags, "checked": len(ALWAYS_RUNNING_PROXMOX)}


def start_proxmox_lxc_or_vm(vmid, kind):
    """Start a stopped LXC ('lxc') or VM ('qm'). Low-risk - only brings up
    something already configured to run continuously, doesn't touch its
    config or data. Use right after check_proxmox_infra_status flags it."""
    cmd = f"pct start {int(vmid)}" if kind == "lxc" else f"qm start {int(vmid)}"
    rc, out, err = run("proxmox-host", cmd, timeout=60)
    ok = rc == 0
    _log_decision("start_proxmox_lxc_or_vm", f"{kind}/{vmid}", "flagged by check_proxmox_infra_status as unexpectedly stopped", "success" if ok else "failed")
    return {"success": ok, "output": out, "error": err}


def check_qnap_app_reachable():
    """Report-only: are the configured native QNAP apps (QPKGs, not Docker
    containers - those go through the normal container checks) reachable on
    their port. Intentionally no auto-fix: QNAP's native-app restart commands
    aren't ones this agent has a well-tested path for, and a really busy
    NAS (e.g. mid-reindex right after a power outage, as observed on
    2026-09-18) can look unreachable for a while without being broken -
    notify_human and let a human judge, rather than guessing at a restart."""
    flags = []
    for app in QNAP_APPS:
        try:
            with socket.create_connection((QNAP_HOST, app["port"]), timeout=5):
                pass
        except OSError as e:
            flags.append({"name": app["name"], "port": app["port"], "error": str(e)})
    return {"flags": flags, "checked": len(QNAP_APPS)}


def _recent_backup_for(name_suffix, window_minutes=90):
    """Was a backup already successfully triggered for this exact target
    recently, per the decision log? Confirmed real gap on 2026-09-19: three
    separate retries of the same HA update (each failing at the *install*
    step, after the backup itself had already succeeded) each blindly
    created a brand new full backup - 7 near-duplicates piled up in one
    day and re-triggered the same disk-space crisis this was already fixed
    once for. A backup from this same retry window is still perfectly
    valid to reuse; nothing about the target state changed between
    attempts."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
    try:
        if not DECISION_LOG_FILE.exists():
            return None
        with DECISION_LOG_FILE.open() as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("action") != "trigger_ha_backup" or entry.get("target") != name_suffix:
                    continue
                if entry.get("outcome", "").startswith("success") and datetime.fromisoformat(entry["timestamp"]) > cutoff:
                    return entry["outcome"].split(":", 1)[1] if ":" in entry["outcome"] else None
    except Exception:
        pass
    return None


def _trigger_ha_backup(name_suffix, timeout_s=1200):
    """Trigger a hassio.backup_full and poll sensor.backup_backup_manager_state
    until idle - same pattern as /root/backup_manager.py, called inline
    here so an HA update always has a fresh pre-update backup regardless
    of that cron's own schedule. Reuses a recent one for the same target
    instead of creating a duplicate - see _recent_backup_for()."""
    reusable = _recent_backup_for(name_suffix)
    if reusable:
        return {"success": True, "backup_name": reusable, "reused": True}

    name = f"Pre-update backup {name_suffix} {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"
    # Same fix as apply_ha_update's install call and for the same reason,
    # confirmed by the actual failure on 2026-09-19: hassio.backup_full's
    # own REST call blocks until the backup meaningfully progresses, not
    # just until it's queued - the previous 30s default was the real
    # bottleneck ("backup outlasting the read timeout" per that cycle's
    # own summary), not the install call this was first mistaken for.
    _ha_request("POST", "/api/services/hassio/backup_full", body={"name": name, "compressed": True}, timeout=1200)
    time.sleep(10)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = _ha_request("GET", "/api/states/sensor.backup_backup_manager_state")
        if state and state["state"] == "idle":
            _log_decision("trigger_ha_backup", name_suffix, "pre-update backup for apply_ha_update", f"success:{name}")
            return {"success": True, "backup_name": name}
        time.sleep(15)
    _log_decision("trigger_ha_backup", name_suffix, "pre-update backup for apply_ha_update", "failed:timeout")
    return {"success": False, "backup_name": name, "error": "backup did not complete within timeout"}


def apply_ha_update(entity_id):
    """Back up first, then install one HA update entity (Core/Supervisor/
    OS/an add-on). IMPORTANT asymmetry vs. every other host in this
    fleet: if HA doesn't come back healthy after an update, we do NOT
    attempt an automated restore. The agent's only control channel to HA
    IS the HA API - if HA is down, the agent has no way to drive a
    restore either, and blindly retrying against a system that might be
    unreachable is more likely to make things worse than better. The
    safe failure path here is a single urgent notification naming the
    pre-update backup so a human can restore it manually."""
    backup = _trigger_ha_backup(entity_id.replace("update.", ""))
    if not backup["success"]:
        notify_human(
            "🔴 HA update aborted - backup failed",
            f"Could not complete a pre-update backup before updating {entity_id}, so the update was not attempted.",
            urgent=True,
        )
        return {"success": False, "stage": "backup", "backup": backup}

    try:
        # _ha_request's default 30s timeout is fine for every other call this
        # agent makes, but HA's update/install service call blocks on the REST
        # API until the whole backup+download+install finishes - confirmed by
        # real failures on 2026-09-19 (repeated "read operation timed out" with
        # the update never actually starting per update.*'s own in_progress
        # attribute). 1200s matches _trigger_ha_backup's own budget above.
        _ha_request("POST", "/api/services/update/install", body={"entity_id": entity_id}, timeout=1200)
    except Exception as e:
        # Core/Supervisor/OS updates restart HA mid-request, so a dropped connection is the expected
        # outcome, not a failure - fall through to the health/state verification below, which decides.
        dropped = any(k in str(e) for k in ("Remote end closed", "Connection reset", "RemoteDisconnected",
                                            "Connection aborted", "timed out", "Broken pipe"))
        if not dropped:
            notify_human("🔴 HA update install call failed", f"{entity_id}: {e}. Pre-update backup: {backup['backup_name']}", urgent=True)
            return {"success": False, "stage": "install_call", "error": str(e), "backup": backup}
        log.info("HA dropped the connection during %s install (expected during restarts) - verifying instead: %s", entity_id, e)

    # HA/Supervisor/OS updates can involve a restart; give it real time,
    # and don't hammer it while it's rebooting.
    time.sleep(60)
    deadline = time.time() + 600
    healthy = False
    while time.time() < deadline:
        health = check_ha_health()
        if health.get("reachable"):
            state = _ha_request("GET", f"/api/states/{entity_id}")
            if state and state["state"] == "off":  # 'off' = no pending update = install succeeded
                healthy = True
                break
        time.sleep(20)

    if healthy:
        notify_human(f"✅ HA updated: {entity_id}", f"Installed successfully. Pre-update backup kept: {backup['backup_name']}")
        return {"success": True, "backup": backup}

    notify_human(
        f"🔴 HA update may have failed: {entity_id}",
        f"HA did not confirm a healthy post-update state within 10 minutes. Do NOT assume automated recovery - "
        f"check it manually. A pre-update backup exists for restore if needed: '{backup['backup_name']}'.",
        urgent=True,
    )
    return {"success": False, "stage": "post_update_health", "backup": backup}


def docker_cleanup(host):
    """Housekeeping: remove dangling (untagged) images, stopped containers
    that have been stopped for over 24h, and unused volumes. Docker's own
    `volume prune` (without -a) already refuses to remove any volume still
    referenced by a container (running or stopped) - it only ever removes
    orphaned/anonymous volumes with zero references, so this can never
    touch a live data volume. Same idea for dangling images: a tag like
    our own :ops-agent-rollback keeps an image non-dangling until it's
    superseded by a newer rollback tag, at which point the old one
    naturally becomes prunable here."""
    if host == "qnap":
        rc1, out1, err1 = qnap_docker_api("POST", "/images/prune?filters=%7B%22dangling%22%3A%5B%22true%22%5D%7D")
        rc2, out2, err2 = qnap_docker_api("POST", "/volumes/prune")
        rc3, out3, err3 = qnap_docker_api("POST", "/containers/prune?filters=%7B%22until%22%3A%5B%2224h%22%5D%7D")
        return {"images": out1, "volumes": out2, "containers": out3}
    rc1, out1, err1 = docker_cli(host, "image prune -f", timeout=120)
    rc2, out2, err2 = docker_cli(host, "volume prune -f", timeout=120)
    rc3, out3, err3 = docker_cli(host, "container prune -f --filter until=24h", timeout=60)
    log.info(f"[{host}] cleanup: images={out1.strip()!r} volumes={out2.strip()!r} containers={out3.strip()!r}")
    _log_decision("docker_cleanup", host, "prune dangling images, unused volumes, and stopped containers", "success")
    return {"images": out1, "volumes": out2, "containers": out3}


def check_resource_limits(host, container):
    """Flag (never auto-change) containers with obviously mismatched
    resource limits: no memory limit configured at all, or actual usage
    sitting near/over a configured limit (undersized), or barely using a
    huge configured limit (oversized). Right-sizing requires a recreate
    with different flags, which carries its own risk - this tool reports
    only, a human decides."""
    info = _inspect(host, container)
    if not info:
        return {"error": "container not found"}
    mem_limit = info.get("HostConfig", {}).get("Memory", 0)

    if host == "qnap":
        rc, out, err = qnap_docker_api("GET", f"/containers/{container}/stats?stream=false")
        try:
            stats = json.loads(out)
            usage = stats["memory_stats"].get("usage", 0)
        except (json.JSONDecodeError, KeyError):
            usage = None
    else:
        rc, out, err = docker_cli(host, f"stats --no-stream --format '{{{{.MemUsage}}}}' {shlex.quote(container)}")
        usage = out.strip() if rc == 0 else None

    flags = []
    if mem_limit == 0:
        flags.append("no memory limit configured")
    return {"memory_limit_bytes": mem_limit, "memory_usage": usage, "flags": flags}


MEMORY_URL = os.environ.get("OPS_AGENT_MEMORY_URL", "").rstrip("/")
MEMORY_TOKEN = os.environ.get("OPS_AGENT_MEMORY_TOKEN", "")


def _mem(method, path, body=None):
    """Shared household/agent memory (best effort: memory trouble must never break a cycle)."""
    if not MEMORY_URL: return None
    try:
        req = urllib.request.Request(MEMORY_URL + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"X-Token": MEMORY_TOKEN, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except Exception as e:
        log.warning(f"memory call {path} failed: {e}")
        return None


def _mem_log(kind, text, meta=None):
    _mem("POST", "/events", {"kind": kind, "text": text[:1500], "meta": meta or {}})


def recall_past(query, days=120):
    """Search past ops events/notes (similar incidents, earlier fixes) in the shared memory."""
    res = _mem("POST", "/events/search", {"query": query, "k": 6, "days": int(days)})
    if res is None: return {"error": "memory unavailable"}
    return {"matches": [{"when": r["ts"][:16], "kind": r["kind"], "text": r["text"], "distance": r["dist"]} for r in res]}


def remember_note(text):
    """Store a durable lesson/observation for future cycles."""
    r = _mem("POST", "/events", {"kind": "ops_note", "text": text[:1500], "meta": {}})
    return {"saved": bool(r)}


def notify_human(title, message, urgent=False, topic=None):
    """topic (optional): a short stable key like 'update-fail:immich_slideshow'
    for a RECURRING/persistent issue - if the same topic was already
    notified within NOTIFY_COOLDOWN_HOURS, this call is suppressed rather
    than re-alerting every single cycle for something already reported and
    still unresolved. Leave topic unset for one-off/new-information
    notifications, which always send."""
    if topic:
        state = load_state()
        recent = state.setdefault("recent_notify_topics", {})
        last = recent.get(topic)
        if last and datetime.now(timezone.utc) - datetime.fromisoformat(last) < timedelta(hours=NOTIFY_COOLDOWN_HOURS):
            log.info(f"NOTIFY suppressed (topic '{topic}' within {NOTIFY_COOLDOWN_HOURS}h cooldown): {title} - {message}")
            return {"sent": False, "suppressed": True, "reason": f"topic already notified within {NOTIFY_COOLDOWN_HOURS}h"}
        recent[topic] = datetime.now(timezone.utc).isoformat()
        save_state(state)

    log.info(f"NOTIFY ({'urgent' if urgent else 'normal'}): {title} - {message}")
    _mem_log("ops_notify", f"{title} - {message}", {"urgent": bool(urgent), "topic": topic})

    # Intentionally separate from Home Assistant's notifications entirely
    # (explicit request - HA is still used elsewhere for its own health/
    # update checks, just not for delivering alerts). ntfy's priority
    # tiers are what make the "iOS summary notification" behavior work:
    # urgent=True -> priority 5 ("urgent"), which iOS delivers immediately
    # and lets through Focus modes / Notification Summary. urgent=False ->
    # priority 3 ("default"), which iOS is free to fold into a scheduled
    # Notification Summary digest instead of interrupting right away - but
    # that folding only happens if Scheduled Summary is turned on for the
    # ntfy app in iOS Settings -> Notifications -> ntfy, a one-time manual
    # step on the phone that this code can't do remotely.
    req = urllib.request.Request(
        f"{NTFY_BASE_URL}/{NTFY_TOPIC}",
        data=message.encode("utf-8"),
        method="POST",
        headers={
            "Title": title.encode("utf-8"),
            "Priority": "urgent" if urgent else "default",
            "Content-Type": "text/plain; charset=utf-8",
        },
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        return {"sent": True}
    except Exception as e:
        log.error(f"notify_human: failed to reach ntfy at {NTFY_BASE_URL}: {e}")
        return {"sent": False, "error": str(e)}


TOOLS = [
    {"type": "function", "function": {
        "name": "list_containers",
        "description": "List all containers on a docker host with their current state/status/health.",
        "parameters": {"type": "object", "properties": {
            "host": {"type": "string", "enum": DOCKER_HOSTS}},
            "required": ["host"]},
    }},
    {"type": "function", "function": {
        "name": "get_logs",
        "description": "Get recent log lines from a container, for diagnosing why it's unhealthy or crashing.",
        "parameters": {"type": "object", "properties": {
            "host": {"type": "string", "enum": DOCKER_HOSTS},
            "container": {"type": "string"},
            "lines": {"type": "integer"}},
            "required": ["host", "container"]},
    }},
    {"type": "function", "function": {
        "name": "restart_container",
        "description": "Restart a crashed or unhealthy container.",
        "parameters": {"type": "object", "properties": {
            "host": {"type": "string", "enum": DOCKER_HOSTS},
            "container": {"type": "string"}},
            "required": ["host", "container"]},
    }},
    {"type": "function", "function": {
        "name": "check_restart_loop",
        "description": (
            "How many times has restart_container already been called for this exact container in the "
            "last window_hours (default 6), per the decision log. Call this before restarting a container "
            "that's ALSO shown up unhealthy in a recent prior cycle - a high count means restarting again "
            "won't fix it, this is a crash loop and worth a properly investigated notify_human instead of "
            "another blind restart. Reports only, never changes anything."
        ),
        "parameters": {"type": "object", "properties": {
            "host": {"type": "string", "enum": DOCKER_HOSTS},
            "container": {"type": "string"},
            "window_hours": {"type": "integer", "description": "Default 6"}},
            "required": ["host", "container"]},
    }},
    {"type": "function", "function": {
        "name": "check_proxmox_infra_status",
        "description": (
            "Is every Proxmox LXC/VM configured to always run actually running right now? This is the "
            "hypervisor layer - a stopped LXC/VM is invisible to every other check (a container/HA check "
            "against a down LXC just looks like a network error). Call this FIRST, before anything else "
            "this cycle: if e.g. docker-lxc is stopped, that also takes ntfy down with it, which would "
            "silently swallow any notify_human call later in the same cycle. Report-only, never changes "
            "anything."
        ),
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "start_proxmox_lxc_or_vm",
        "description": "Start a stopped LXC or VM flagged by check_proxmox_infra_status. Low-risk - only brings up something already meant to run continuously.",
        "parameters": {"type": "object", "properties": {
            "vmid": {"type": "integer"},
            "kind": {"type": "string", "enum": ["lxc", "qm"]}},
            "required": ["vmid", "kind"]},
    }},
    {"type": "function", "function": {
        "name": "check_qnap_app_reachable",
        "description": (
            "Are the configured native QNAP apps (not Docker containers, e.g. Plex) reachable on their "
            "port. Report-only, no auto-fix - QNAP's native-app restart commands aren't well-tested here, "
            "and a busy NAS (e.g. mid-reindex after a power outage) can look down without being broken. "
            "Use judgment before notifying: check how long it's plausibly been down, don't alert on a "
            "single reading right after a host reboot."
        ),
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "scan_container_updates",
        "description": "Check ALL managed containers for updates in one call (new build under the same tag, and newer upstream tag). Returns only the containers that have something available. Use this instead of calling check_for_update / check_upstream_tag_update per container.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "check_for_update",
        "description": "Check whether a newer image is available for a container (pulls the same tag and compares digests).",
        "parameters": {"type": "object", "properties": {
            "host": {"type": "string", "enum": DOCKER_HOSTS},
            "container": {"type": "string"}},
            "required": ["host", "container"]},
    }},
    {"type": "function", "function": {
        "name": "apply_update",
        "description": (
            "Safely update a container to the latest pulled image: automatically backs up its data first "
            "if it holds irreplaceable user data, recreates it (via docker compose when available, which "
            "never deletes volumes), verifies health, and automatically rolls back to the exact previous "
            "image if the health check fails. Call check_for_update first to confirm an update is actually "
            "available, and check_changelog before this if you haven't already this cycle for this "
            "container. Only refuse and notify_human instead if check_changelog actually FOUND text that "
            "reads as breaking - most images carry no version-label/GitHub-source metadata at all, so "
            "'available: false' (no changelog signal) is the common, expected case and is NOT a reason to "
            "hold back. This is a same-tag rebuild (never a version change) with an automatic health-check "
            "and rollback already protecting it - proceed by default."
        ),
        "parameters": {"type": "object", "properties": {
            "host": {"type": "string", "enum": DOCKER_HOSTS},
            "container": {"type": "string"}},
            "required": ["host", "container"]},
    }},
    {"type": "function", "function": {
        "name": "check_changelog",
        "description": (
            "Best-effort changelog/release-notes lookup between the running and newly-pulled image version "
            "(via GitHub releases, when the image carries OCI source/version labels - many don't, in which "
            "case this reports 'not available' and apply_update's own backup+rollback net is the only safety "
            "mechanism for that container, which is fine). Call this after check_for_update confirms an "
            "update is available and before apply_update, so you can catch likely-breaking changes (major "
            "version jump, migration steps, removed/renamed config) before applying rather than after."
        ),
        "parameters": {"type": "object", "properties": {
            "host": {"type": "string", "enum": DOCKER_HOSTS},
            "container": {"type": "string"}},
            "required": ["host", "container"]},
    }},
    {"type": "function", "function": {
        "name": "check_upstream_tag_update",
        "description": (
            "Unlike check_for_update (which only ever detects a new build under the SAME pinned tag - "
            "necessary because container tags are intentionally pinned off :latest for safety), this checks "
            "what tags actually exist upstream (Docker Hub or GHCR) and proposes the highest one that's a "
            "newer version of the same image variant (same v-prefix and suffix as the current tag - e.g. "
            "never proposes jumping a plain '32' to a '-rocm' build). Read-only, never pulls or changes "
            "anything. Worth calling periodically for containers pinned to a specific version tag, since "
            "check_for_update alone will silently never notice a new major/minor version exists."
        ),
        "parameters": {"type": "object", "properties": {
            "host": {"type": "string", "enum": DOCKER_HOSTS},
            "container": {"type": "string"}},
            "required": ["host", "container"]},
    }},
    {"type": "function", "function": {
        "name": "check_upstream_tag_changelog",
        "description": (
            "Like check_changelog, but for a specific candidate tag from check_upstream_tag_update rather "
            "than a same-tag rebuild - pulls that candidate image and looks up its GitHub release notes "
            "against the currently running version. Call this on any candidate_image before ever calling "
            "bump_compose_tag_and_apply. When it returns actual changelog text, read it and judge for "
            "yourself whether it reads as breaking (migration steps, removed/renamed config, data format "
            "changes) - do not pattern-match on keywords, actually reason about what it says. When it "
            "returns 'available: false' (no metadata to look up - common, most images don't carry it), "
            "that is NOT a signal to hold back on its own: for a patch/minor bump (same major version), "
            "proceed anyway - bump_compose_tag_and_apply's own rollback is the safety net. Reserve holding "
            "back on missing changelog data for an actual MAJOR version jump, where an undocumented change "
            "is more likely to need a real migration step a rollback can't cleanly undo."
        ),
        "parameters": {"type": "object", "properties": {
            "host": {"type": "string", "enum": DOCKER_HOSTS},
            "container": {"type": "string"},
            "candidate_image_ref": {"type": "string", "description": "The candidate_image string from check_upstream_tag_update, e.g. 'nextcloud:33'"}},
            "required": ["host", "container", "candidate_image_ref"]},
    }},
    {"type": "function", "function": {
        "name": "bump_compose_tag_and_apply",
        "description": (
            "Edits the pinned tag in this container's compose file to candidate_tag, then runs the exact "
            "same backup+recreate+health-check+rollback flow as apply_update - if that fails, the compose "
            "file edit is automatically reverted too, so a failed attempt never leaves the compose file "
            "out of sync with what's actually running. Only call this after check_upstream_tag_changelog. "
            "If it found real changelog text that reads as breaking, notify_human instead and never call "
            "this. If it found no changelog data at all, that alone should only block a MAJOR version jump - "
            "for a patch/minor bump, proceed and rely on this tool's own automatic rollback."
        ),
        "parameters": {"type": "object", "properties": {
            "host": {"type": "string", "enum": DOCKER_HOSTS},
            "container": {"type": "string"},
            "candidate_tag": {"type": "string", "description": "The candidate_tag string from check_upstream_tag_update, e.g. '33'"}},
            "required": ["host", "container", "candidate_tag"]},
    }},
    {"type": "function", "function": {
        "name": "check_os_updates",
        "description": "Check available OS package updates on a host. Only hosts listed in os_auto_hosts are valid. The Proxmox host and the NAS are never OS-patched by this agent, whatever the config says.",
        "parameters": {"type": "object", "properties": {
            "host": {"type": "string", "enum": OS_AUTO_HOSTS}},
            "required": ["host"]},
    }},
    {"type": "function", "function": {
        "name": "apply_os_updates",
        "description": (
            "Apply pending OS package updates on a host. Only works on workstation and adsb - Proxmox "
            "itself and both its LXCs are permanently excluded from OS-level patching (too high a blast "
            "radius, they host everything else including this agent). Their containers are still fully "
            "managed via the docker tools, this exclusion is OS-level only."
        ),
        "parameters": {"type": "object", "properties": {
            "host": {"type": "string"}},
            "required": ["host"]},
    }},
    {"type": "function", "function": {
        "name": "check_adsb_service_health",
        "description": "Check the ADS-B Raspberry Pi's core receiver services (readsb, tar1090) and secondary feed services.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "restart_adsb_service",
        "description": "Restart a systemd service on the ADS-B Pi that check_adsb_service_health reported as down.",
        "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]},
    }},
    {"type": "function", "function": {
        "name": "check_ha_health",
        "description": "Check whether Home Assistant is reachable at all.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "check_ha_updates",
        "description": "List pending Home Assistant updates (Core, Supervisor, OS, add-ons) via its update.* entities.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "ha_search_entities",
        "description": "Find Home Assistant entity_ids by a case-insensitive substring match against the entity_id or friendly name. Use this FIRST when unsure of an exact entity_id rather than guessing variations. Read-only.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}},
            "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "ha_get_state",
        "description": "Read one Home Assistant entity's current state and attributes. Read-only, any entity, never changes anything.",
        "parameters": {"type": "object", "properties": {
            "entity_id": {"type": "string"}},
            "required": ["entity_id"]},
    }},
    {"type": "function", "function": {
        "name": "ha_get_history",
        "description": "Read a Home Assistant entity's state-change history over the last N hours (max 168). Read-only - use to check whether something actually changed state when expected.",
        "parameters": {"type": "object", "properties": {
            "entity_id": {"type": "string"},
            "hours": {"type": "integer", "description": "Default 24, max 168"}},
            "required": ["entity_id"]},
    }},
    {"type": "function", "function": {
        "name": "ha_get_logbook",
        "description": "Read Home Assistant logbook entries for an entity over the last N hours (max 168) - shows what actually triggered/happened, e.g. whether an automation fired. Read-only.",
        "parameters": {"type": "object", "properties": {
            "entity_id": {"type": "string"},
            "hours": {"type": "integer", "description": "Default 24, max 168"}},
            "required": ["entity_id"]},
    }},
    {"type": "function", "function": {
        "name": "ha_call_service",
        "description": (
            "Call a Home Assistant service to control something - e.g. domain=light, service=turn_on, "
            "entity_id=light.kitchen. Restricted to a safe-domain allowlist (lights, switches, climate, "
            "fans, media players, input_* helpers, timers, notify) - locks, covers/garage doors, "
            "and alarm panels are NOT callable this way, and scripts and scenes only if named in the config."
        ),
        "parameters": {"type": "object", "properties": {
            "domain": {"type": "string"},
            "service": {"type": "string"},
            "entity_id": {"type": "string"},
            "data": {"type": "object", "description": "Extra service call fields, e.g. {\"brightness\": 128} for light.turn_on"}},
            "required": ["domain", "service", "entity_id"]},
    }},
    {"type": "function", "function": {
        "name": "apply_ha_update",
        "description": (
            "Update one Home Assistant component. Always takes a fresh full backup first, then installs, "
            "then verifies HA came back healthy. If it does NOT come back healthy, this does NOT attempt an "
            "automated restore (HA being down would break the agent's own control channel to it) - it just "
            "sends an urgent notification naming the pre-update backup for manual recovery."
        ),
        "parameters": {"type": "object", "properties": {"entity_id": {"type": "string"}}, "required": ["entity_id"]},
    }},
    {"type": "function", "function": {
        "name": "docker_cleanup",
        "description": "Housekeeping: prune dangling images, orphaned volumes, and stopped containers older than 24h. Never removes a volume still attached to any container.",
        "parameters": {"type": "object", "properties": {
            "host": {"type": "string", "enum": DOCKER_HOSTS}},
            "required": ["host"]},
    }},
    {"type": "function", "function": {
        "name": "check_resource_limits",
        "description": "Check a container's memory limit vs actual usage and flag obvious mismatches (no limit set, near-limit usage, way-oversized limit). Reports only - never changes anything.",
        "parameters": {"type": "object", "properties": {
            "host": {"type": "string", "enum": DOCKER_HOSTS},
            "container": {"type": "string"}},
            "required": ["host", "container"]},
    }},
    {"type": "function", "function": {
        "name": "check_host_resource_trend",
        "description": (
            "Compare each container's current memory footprint on a host against its footprint from ~7+ "
            "days ago, flagging any that grew significantly, plus whether the host crossed its optional "
            "configured resource ceiling. Reports only - there is no tool to migrate, resize, or restart "
            "anything based on this. If you see a strong migration candidate, notify_human with a specific "
            "reasoned recommendation instead (see system prompt)."
        ),
        "parameters": {"type": "object", "properties": {
            "host": {"type": "string", "enum": DOCKER_HOSTS}},
            "required": ["host"]},
    }},
    {"type": "function", "function": {
        "name": "check_automation_health",
        "description": (
            "Check enabled Home Assistant automations for ones whose trigger entity changed state more "
            "recently than the automation's own last_triggered - meaning it may have silently not fired "
            "(or a condition properly blocked it, which looks identical from this signal alone - use "
            "judgment, this is a flag not proof). Also automatically runs the stuck-sensor check on every "
            "numeric_state trigger entity it sees and includes any frozen sensor as its own flag (issue: "
            "'sensor_stuck') - this catches a sensor frozen INSIDE its automation's dead zone (never crossing "
            "a threshold, so invisible to the crossing-check above) which is what let a lighting "
            "automation go unwatched twice. Flags carry an 'issue' field ('missed_trigger' or 'sensor_stuck') "
            "so you can tell the two apart. Reports only, never changes anything."
        ),
        "parameters": {"type": "object", "properties": {
            "lookback_hours": {"type": "integer", "description": "Default 2"}},
            "required": []},
    }},
    {"type": "function", "function": {
        "name": "check_sensor_value_stuck",
        "description": (
            "Is this sensor's value bit-for-bit identical across the whole lookback window - a frozen/stuck "
            "device, which still reports as 'available' (not unavailable/unknown) so ha_get_history's usual "
            "availability check alone won't catch it. check_automation_health already calls this automatically "
            "on every numeric_state trigger entity it sees (surfaced as issue='sensor_stuck' flags), so you "
            "usually don't need to call this yourself for that path. Call it directly when investigating some "
            "other numeric sensor outside that flow - e.g. a resource-trend or manual freshness question - "
            "where ha_get_history's availability check alone isn't enough. Judge the result against what the "
            "sensor measures: suspicious for anything that should naturally fluctuate (illuminance, power "
            "draw, temperature), plausible for a really stable one (a setpoint, a binary state with no "
            "reason to have changed)."
        ),
        "parameters": {"type": "object", "properties": {
            "entity_id": {"type": "string"},
            "hours": {"type": "integer", "description": "Lookback window, default 3"}},
            "required": ["entity_id"]},
    }},
    {"type": "function", "function": {
        "name": "check_expected_automations_enabled",
        "description": (
            "Check that automations listed in the always_enabled_automations config are actually turned on "
            "right now. Catches the case check_automation_health can't - an automation that got silently "
            "disabled (failed reload after an HA restart, accidental toggle-off) and so is invisible to any "
            "trigger-based check. Reports only, never changes anything."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "notify_human",
        "description": (
            "Send a push notification to both phones. Use urgent=true only for failures that could not be "
            "auto-resolved. For a RECURRING issue you're reporting again this cycle because it's still "
            "unresolved from a previous cycle (e.g. the same container still failing to update), always pass "
            "a stable 'topic' string identifying it (e.g. 'update-fail:<container>') - this prevents "
            "re-alerting the human every single cycle about something already reported and still pending; "
            "repeat notifications with the same topic within a few hours are automatically suppressed. Leave "
            "topic unset only for really new/one-off information."
        ),
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"},
            "message": {"type": "string"},
            "urgent": {"type": "boolean"},
            "topic": {"type": "string", "description": "Stable key for a recurring issue, e.g. 'update-fail:immich_slideshow'. Omit for new/one-off notifications."}},
            "required": ["title", "message"]},
    }},
    {"type": "function", "function": {
        "name": "recall_past",
        "description": "Search the shared long-term memory for similar past incidents, fixes, notifications or notes (semantic search). Use it BEFORE deciding what to do about a recurring or unfamiliar problem, e.g. 'nextcloud returns 502' or 'update failed for immich'.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "days": {"type": "integer", "description": "How far back to look, default 120"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "remember_note",
        "description": "Save a short durable lesson for future cycles (root cause found, a fix that worked, something to avoid). Keep it factual and under 300 characters. Do not store secrets.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    }},
]

if not ADSB_ENABLED:
    TOOLS = [t for t in TOOLS if "adsb" not in t["function"]["name"]]

TOOL_IMPL = {
    "list_containers": list_containers,
    "recall_past": recall_past,
    "remember_note": remember_note,
    "get_logs": get_logs,
    "restart_container": restart_container,
    "check_restart_loop": check_restart_loop,
    "check_for_update": check_for_update,
    "scan_container_updates": scan_container_updates,
    "check_changelog": check_changelog,
    "check_upstream_tag_update": check_upstream_tag_update,
    "check_upstream_tag_changelog": check_upstream_tag_changelog,
    "bump_compose_tag_and_apply": bump_compose_tag_and_apply,
    "apply_update": apply_update,
    "check_os_updates": check_os_updates,
    "apply_os_updates": apply_os_updates,
    "docker_cleanup": docker_cleanup,
    "check_resource_limits": check_resource_limits,
    "check_host_resource_trend": check_host_resource_trend,
    "check_automation_health": check_automation_health,
    "check_sensor_value_stuck": check_sensor_value_stuck,
    "check_expected_automations_enabled": check_expected_automations_enabled,
    "check_proxmox_infra_status": check_proxmox_infra_status,
    "start_proxmox_lxc_or_vm": start_proxmox_lxc_or_vm,
    "check_qnap_app_reachable": check_qnap_app_reachable,
    "check_adsb_service_health": check_adsb_service_health,
    "restart_adsb_service": restart_adsb_service,
    "check_ha_health": check_ha_health,
    "check_ha_updates": check_ha_updates,
    "ha_search_entities": ha_search_entities,
    "ha_get_state": ha_get_state,
    "ha_get_history": ha_get_history,
    "ha_get_logbook": ha_get_logbook,
    "ha_call_service": ha_call_service,
    "apply_ha_update": apply_ha_update,
    "notify_human": notify_human,
}


# ----------------------------------------------------------- state --

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"last_update_check": None, "cycle_history": []}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state))


# ------------------------------------------------------- HA status publish --

CYCLE_HISTORY_LIMIT = 30

ACTION_ICONS = {
    "restart_container": "mdi:restart",
    "apply_update": "mdi:package-up",
    "apply_os_updates": "mdi:cellphone-arrow-down",
    "apply_ha_update": "mdi:home-assistant",
    "restart_adsb_service": "mdi:airplane",
    "notify_human": "mdi:bell-alert",
    "docker_cleanup": "mdi:broom",
}


def _ha_set_state(entity_id, state, attributes):
    try:
        _ha_request("POST", f"/api/states/{entity_id}", body={"state": state, "attributes": attributes})
    except Exception as e:
        log.warning(f"failed to publish {entity_id} to HA: {e}")


def publish_status_to_ha(status, state, snapshot=None, managed=None, unmanaged=None,
                          unhealthy=None, ha_health=None, adsb_health=None,
                          actions=None, error=None):
    """Push the agent's own status/metrics into HA as sensors so it can be
    monitored on its own dashboard tab, same pattern as the existing
    stats/adsb/news publishers. Called at cycle start (status='running'),
    cycle end (status='idle'), and on crash (status='error')."""
    now = datetime.now(timezone.utc)

    _ha_set_state("sensor.ops_agent_status", status, {
        "friendly_name": "Ops Agent Status",
        "icon": "mdi:robot-industrial" if status != "error" else "mdi:robot-dead",
    })
    _ha_set_state("sensor.ops_agent_model", MODEL, {
        "friendly_name": "Ops Agent Model", "icon": "mdi:brain",
    })

    if status == "running":
        return  # nothing else meaningful to report until the cycle finishes

    per_host_counts = {}
    total = len(managed or []) + len(unmanaged or [])
    if snapshot:
        for host, containers in snapshot.items():
            per_host_counts[host] = len(containers) if isinstance(containers, list) else 0

    _ha_set_state("sensor.ops_agent_containers_total", total, {
        "friendly_name": "Ops Agent Containers Managed", "icon": "mdi:docker",
        "unit_of_measurement": "containers", "per_host": per_host_counts,
        "managed": len(managed or []), "unmanaged_one_shot": len(unmanaged or []),
    })

    unhealthy_names = [f"{c['host']}/{c['name']}" for c in (unhealthy or [])]
    _ha_set_state("sensor.ops_agent_containers_unhealthy", len(unhealthy or []), {
        "friendly_name": "Ops Agent Unhealthy Containers", "icon": "mdi:alert-circle",
        "unit_of_measurement": "containers", "names": unhealthy_names,
    })

    _ha_set_state("binary_sensor.ops_agent_ha_reachable",
                   "on" if (ha_health or {}).get("reachable") else "off",
                   {"friendly_name": "Ops Agent: HA Reachable", "device_class": "connectivity"})

    adsb_ok = bool(adsb_health) and not adsb_health.get("error") and all(
        v == "active" for k, v in (adsb_health or {}).items() if k in ADSB_CORE_SERVICES)
    _ha_set_state("sensor.ops_agent_adsb_health", ("ok" if adsb_ok else "degraded") if ADSB_ENABLED else "disabled", {
        "friendly_name": "Ops Agent: ADS-B Pi Health", "icon": "mdi:airplane-check",
        "services": adsb_health or {},
    })

    actions = actions or []
    action_counts = {}
    for a in actions:
        action_counts[a["tool"]] = action_counts.get(a["tool"], 0) + 1

    _ha_set_state("sensor.ops_agent_last_cycle", now.isoformat(), {
        "friendly_name": "Ops Agent Last Cycle", "device_class": "timestamp",
        "icon": "mdi:clock-check", "action_count": len(actions),
        "action_counts": action_counts, "error": error,
    })
    next_run = now + timedelta(minutes=CYCLE_INTERVAL_MINUTES)
    _ha_set_state("sensor.ops_agent_next_cycle", next_run.isoformat(), {
        "friendly_name": "Ops Agent Next Cycle", "device_class": "timestamp", "icon": "mdi:clock-fast",
    })

    # Rolling history for a dashboard timeline/log card.
    history = state.setdefault("cycle_history", [])
    history.append({
        "timestamp": now.isoformat(),
        "status": status,
        "action_count": len(actions),
        "action_counts": action_counts,
        "unhealthy": unhealthy_names,
        "error": error,
        # keep enough per-action detail for a log view without being huge
        "actions": [{"tool": a["tool"],
                     "icon": ACTION_ICONS.get(a["tool"], "mdi:cog"),
                     "summary": a["result_summary"][:180]} for a in actions][:15],
    })
    del history[:-CYCLE_HISTORY_LIMIT]
    save_state(state)

    _ha_set_state("sensor.ops_agent_cycle_history", len(history), {
        "friendly_name": "Ops Agent Cycle History", "icon": "mdi:history",
        "unit_of_measurement": "cycles", "cycles": list(reversed(history)),
    })


# ------------------------------------------------------------- model loop --

def gather_fleet_snapshot():
    """Per-host container listing is a read-only SSH/API round-trip with no
    shared state between hosts, so it's safe to parallelize - this is the
    slowest deterministic part of a cycle (three sequential SSH round-trips
    otherwise) and the only reason it used to take as long as it did.
    Anything that WRITES shared state (state.json via load_state/save_state
    - notify_human, check_ubuntu_lts_available, check_host_resource_trend)
    intentionally stays out of this pool and out of the model's own tool-call
    loop below, which stays sequential on purpose - see run_cycle()."""
    snapshot = {}
    with ThreadPoolExecutor(max_workers=len(DOCKER_HOSTS)) as ex:
        futures = {ex.submit(list_containers, host): host for host in DOCKER_HOSTS}
        for fut in as_completed(futures):
            host = futures[fut]
            try:
                snapshot[host] = fut.result()
            except Exception as e:
                log.exception(f"[{host}] list_containers raised")
                snapshot[host] = {"error": str(e)}
    return snapshot


def _gather_ha():
    health = check_ha_health()
    updates = check_ha_updates() if health.get("reachable") else {"pending": [], "note": "HA unreachable"}
    return health, updates


def call_model(messages, _retry=True):
    # Confirmed real 2026-10-04: this model was defaulting to Ollama's
    # keep_alive=-1 (never unload) on the workstation's shared GPU, sitting
    # permanently loaded all day even though this agent only actually runs
    # during the overnight heavy-cycle window (see HEAVY_CYCLE_HOURS) - no
    # reason to hold that memory hostage the rest of the day. Intentionally
    # NOT forced to CPU-only: voice/doorbell usage is naturally low
    # overnight anyway, so the time-window separation already avoids GPU
    # contention without paying CPU inference's ~4x slowdown for no benefit
    # (confirmed: 10.7 tok/s on CPU vs ~40+ on GPU for this model).
    payload = {"model": MODEL, "messages": messages, "tools": TOOLS, "stream": False, "keep_alive": 1800,
               "options": {"num_ctx": 49152}}  # default 4096 silently dropped the user prompt mid-cycle -> HTTP 500 "no user query found"
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError) as e:
        if not _retry:
            raise
        # Confirmed real on 2026-09-19 night: two cycle crashes (an HTTP 500 and
        # a hard 600s timeout) both coincided with other long-running requests
        # (4-11 min) queued on the same single GPU. _wait_for_ollama_idle() only
        # runs once at the very start of run_cycle(), so contention that builds
        # up mid-cycle (this agent's own earlier turns included) isn't caught -
        # wait for a real idle GPU, then retry this exact call once before
        # giving up and letting the cycle crash+notify as before.
        log.warning(f"call_model failed ({e}), waiting for GPU idle before one retry")
        _wait_for_ollama_idle()
        return call_model(messages, _retry=False)


def _wait_for_ollama_idle(max_wait_seconds=240, probe_threshold_seconds=4, poll_interval_seconds=20):
    """Best-effort contention avoidance for the workstation's single shared GPU,
    also used by the video doorbell's AI Task and interactive chat (OpenWebUI).
    Confirmed empirically this GPU does not parallelize: two concurrent
    generation requests took longer combined than one after another, and a
    real doorbell vision call that normally takes ~2-8s took 48s after
    colliding with another consumer's request. Ops-agent's own cycle is the
    lower-priority, deferrable consumer here (nothing else is waiting on it
    the way a person is waiting on a chat reply or a doorbell notification),
    so it briefly yields instead of piling onto whatever's already running.

    A quiet GPU answers a trivial 1-token probe in well under a second; a busy
    one takes many seconds because the probe queues behind whatever else is
    generating. Always gives up and proceeds after max_wait_seconds - this
    must never cause a scheduled cycle (including active-outage handling) to
    be skipped outright, only delayed by a bounded amount."""
    probe_url = OLLAMA_URL.replace("/api/chat", "/api/generate")
    waited = 0
    while waited < max_wait_seconds:
        payload = {"model": MODEL, "prompt": "hi", "stream": False, "keep_alive": 1800, "options": {"num_predict": 1, "num_ctx": 49152}}
        req = urllib.request.Request(
            probe_url, data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json"},
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
            elapsed = time.monotonic() - t0
        except TimeoutError:
            # A timed-out probe is the strongest possible busy signal (things are
            # backed up enough that even a 1-token request couldn't get a slot in
            # 30s) - confirmed happening for real, not hypothetical. Treat exactly
            # like the threshold-exceeded case below, not like a real error.
            elapsed = 30.0
        except Exception as e:
            # A real connection/network error (Ollama down, DNS, refused) has
            # nothing to wait for - proceed immediately rather than block on it.
            log.warning(f"GPU-idle probe failed ({e}), proceeding without waiting")
            return
        if elapsed < probe_threshold_seconds:
            if waited:
                log.info(f"GPU idle again after {waited}s wait, starting cycle")
            return
        log.info(f"GPU busy (probe took {elapsed:.1f}s) - waiting {poll_interval_seconds}s before "
                  f"retrying ({waited}/{max_wait_seconds}s waited so far)")
        time.sleep(poll_interval_seconds)
        waited += poll_interval_seconds
    log.info(f"GPU still busy after {max_wait_seconds}s max wait - proceeding anyway, this cycle's "
              f"checks (unhealthy containers, HA reachability) shouldn't be delayed indefinitely")


def run_lightweight_cycle():
    """The non-overnight counterpart to run_cycle() - same pure-Python data
    gathering and mechanical fixes (restart a crashed container/service,
    start a stopped LXC/VM), but makes every decision with plain Python
    rules instead of the model, so it never touches Ollama. See
    HEAVY_CYCLE_HOURS for why this exists. Intentionally narrower than
    run_cycle(): no updates, no changelog review, no resource-trend
    analysis, and - for automation-health - only acts on the flag types
    that are unambiguous without a model's judgment (sensor_stuck,
    expected-but-disabled); ambiguous missed_trigger flags are left for the
    next overnight cycle to actually reason about, same as this cycle
    already defers update decisions.

    ADS-B is also intentionally left entirely to the overnight run_cycle,
    not handled here at all - confirmed real 2026-10-04: its SSH control
    channel is chronically slow/flaky, and a blind mechanical restart loop
    over 3-4 services at 120s-timeout-each can burn most of this cycle's
    runtime for a Pi that's known to need the model's flapping/crash-loop
    judgment anyway (see run_cycle step 3), not a repeated blind retry."""
    log.info("=== lightweight cycle start (no model calls) ===")
    actions = []

    with ThreadPoolExecutor(max_workers=2) as ex:
        snapshot_future = ex.submit(gather_fleet_snapshot)
        ha_future = ex.submit(_gather_ha)
        snapshot = snapshot_future.result()
        ha_health, _ha_updates = ha_future.result()

    unhealthy, managed, unmanaged = [], [], []
    for host, containers in snapshot.items():
        if isinstance(containers, dict) and "error" in containers:
            log.error(f"[{host}] could not list containers: {containers['error']}")
            continue
        for c in containers:
            entry = {"host": host, **c}
            if c.get("expected_always_running"):
                managed.append(entry)
                if c.get("state") != "running" or c.get("health") == "unhealthy":
                    unhealthy.append(entry)
            else:
                unmanaged.append(entry)

    for entry in unhealthy:
        host, container = entry["host"], entry["name"]
        loop = check_restart_loop(host, container)
        if isinstance(loop, dict) and loop.get("looping"):
            notify_human(
                f"🔴 {container} on {host} is crash-looping",
                f"Restarted {loop.get('count')} times in the last 6h already - not restarting again "
                f"blindly during the lightweight (daytime) cycle. Needs a look.",
                urgent=True, topic=f"restart-loop:{host}:{container}",
            )
            continue
        result = restart_container(host, container)
        actions.append({"tool": "restart_container", "args": {"host": host, "container": container}, "result_summary": str(result)[:300]})
        if not (isinstance(result, dict) and result.get("success")):
            notify_human(
                f"🔴 {container} on {host} is down and restart failed",
                f"restart_container result: {result}",
                urgent=True, topic=f"restart-fail:{host}:{container}",
            )

    if not ha_health.get("reachable"):
        notify_human("🔴 Home Assistant unreachable", f"{ha_health}", urgent=True, topic="ha-unreachable")

    infra = check_proxmox_infra_status()
    for flag in infra.get("flags", []):
        result = start_proxmox_lxc_or_vm(flag["vmid"], flag["kind"])
        actions.append({"tool": "start_proxmox_lxc_or_vm", "args": {"flag": flag}, "result_summary": str(result)[:300]})
        notify_human(
            f"⚠️ {flag.get('name', flag['vmid'])} was stopped, started it back up",
            f"{flag}", urgent=True, topic=f"proxmox-stopped:{flag['vmid']}",
        )

    qnap = check_qnap_app_reachable()
    for flag in qnap.get("flags", []):
        notify_human(f"⚠️ QNAP app unreachable: {flag.get('name')}", f"{flag}", topic=f"qnap-unreachable:{flag.get('name')}")

    expected = check_expected_automations_enabled()
    for flag in expected.get("flags", []):
        notify_human(
            f"⚠️ Automation unexpectedly disabled: {flag.get('entity_id')}",
            f"{flag}", urgent=True, topic=f"automation-disabled:{flag.get('entity_id')}",
        )

    auto_health = check_automation_health()
    for flag in auto_health.get("flags", []):
        if flag.get("issue") == "sensor_stuck":
            notify_human(
                f"⚠️ Sensor may be stuck: {flag.get('trigger_entity')}",
                f"Frozen at {flag.get('value')} since {flag.get('unchanged_since')} - automation "
                f"{flag.get('automation')} depends on it. {flag}",
                topic=f"sensor-stuck:{flag.get('trigger_entity')}",
            )
        # missed_trigger flags need the model's judgment to tell a real miss
        # from a properly-blocked condition - intentionally not acted on
        # here, left for the next overnight run_cycle() to assess.

    log.info(f"=== lightweight cycle end, {len(actions)} actions ===")
    return {
        "snapshot": snapshot, "managed": managed, "unmanaged": unmanaged, "unhealthy": unhealthy,
        "ha_health": ha_health, "adsb_health": None, "actions": actions,
    }


MAX_TURNS = 25   # model turns per cycle; a confused model cannot loop forever


def run_model_loop(messages):
    """Let the model call tools until it gives a final answer, with the cycle's guard rails:
    an identical call is refused after two tries, there is a turn limit and a wall clock budget,
    model arguments are validated before any tool runs, and tool output is clipped and pruned.
    Returns (actions_taken, aborted). aborted is True when the cycle stopped early (budget used up
    or the model call failed), which makes the caller retry the update check later."""
    actions_taken = []
    aborted = False
    seen_calls = {}
    cycle_t0 = time.monotonic()
    for turn in range(MAX_TURNS):  # hard cap so a confused model can't loop forever
        if time.monotonic() - cycle_t0 > CYCLE_BUDGET_SECONDS:
            log.warning("cycle time budget exhausted - stopping with partial results")
            aborted = True
            _mem_log("ops_cycle", "Ops cycle stopped early: time budget exhausted after %d tool calls" % len(actions_taken))
            break
        try:
            resp = call_model(messages)
        except Exception as e:
            log.exception("model call failed mid-cycle - ending cycle with partial results")
            aborted = True
            notify_human("Ops agent model call failed mid-cycle",
                         f"{e} (after {len(actions_taken)} tool calls; will retry next cycle)")
            break
        msg = resp.get("message", {})
        tool_calls = msg.get("tool_calls", [])
        messages.append(msg)

        if not tool_calls:
            log.info(f"model final summary: {msg.get('content', '')}")
            _mem_log("ops_cycle", "Ops cycle summary: " + str(msg.get("content", "")))
            break

        for call in tool_calls:
            fn_name = call["function"]["name"]
            args = call["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            impl = TOOL_IMPL.get(fn_name)
            call_key = fn_name + json.dumps(args, sort_keys=True, default=str)
            seen_calls[call_key] = seen_calls.get(call_key, 0) + 1
            if seen_calls[call_key] > 2:
                result = {"error": "identical call already made twice this cycle; do not repeat it - report the outcome and finish"}
            elif not impl:
                result = {"error": f"unknown tool {fn_name}"}
            else:
                try:
                    validate_tool_args(args)
                except ValueError as e:
                    log.warning(f"tool call rejected, bad argument: {fn_name}: {e}")
                    result = {"error": f"rejected: {e}"}
                else:
                    try:
                        result = impl(**args)
                    except Exception as e:
                        log.exception(f"tool {fn_name} raised")
                        result = {"error": str(e)}
            log.info(f"tool call: {fn_name}({args}) -> {str(result)[:200]}")
            actions_taken.append({"tool": fn_name, "args": args, "result_summary": str(result)[:300]})
            messages.append({"role": "tool", "content": _clip_tool_result(result)})
        _prune_messages(messages)
    else:
        log.warning("hit max turns for this cycle - stopping")
    return actions_taken, aborted


def run_cycle(check_updates_this_cycle):
    log.info(f"=== cycle start (check_updates={check_updates_this_cycle}) ===")
    _wait_for_ollama_idle()
    # These three are fully independent of each other (different hosts/APIs,
    # nothing shared), so gathering them concurrently is a plain speedup with
    # no change in behavior - each still returns exactly what it always did,
    # just not one-after-another. Everything downstream of this block (the
    # model's own tool-call loop) stays sequential - see gather_fleet_snapshot().
    with ThreadPoolExecutor(max_workers=3) as ex:
        snapshot_future = ex.submit(gather_fleet_snapshot)
        adsb_future = ex.submit(check_adsb_service_health)
        ha_future = ex.submit(_gather_ha)
        snapshot = snapshot_future.result()
        adsb_health = adsb_future.result()
        ha_health, ha_updates = ha_future.result()

    unhealthy = []
    managed = []  # restart_policy always/unless-stopped: eligible for update/cleanup
    unmanaged = []  # restart_policy 'no': one-shot jobs or intentionally-stopped - hands off entirely
    for host, containers in snapshot.items():
        if isinstance(containers, dict) and "error" in containers:
            log.error(f"[{host}] could not list containers: {containers['error']}")
            continue
        for c in containers:
            entry = {"host": host, **c}
            if c.get("expected_always_running"):
                managed.append(entry)
                # Only flag as "needs attention" a managed container that
                # is either explicitly unhealthy via its own HEALTHCHECK,
                # or SUPPOSED to run continuously but currently isn't.
                crashed = c.get("state") != "running"
                explicitly_unhealthy = c.get("health") == "unhealthy"
                if crashed or explicitly_unhealthy:
                    unhealthy.append(entry)
            else:
                # restart_policy 'no': one-shot jobs (hello-world etc) or
                # anything a human intentionally stopped. Never restarted,
                # never updated, never started by this agent - confirmed
                # necessary during live testing (it restarted minikube and
                # exited hello-world containers before this fix existed).
                unmanaged.append(entry)

    system_prompt = (
        "You are an infrastructure ops agent for a homelab. You manage: docker containers on three hosts "
        "(workstation, docker-lxc, qnap - ALL containers on ALL of these are in full scope, including "
        "data-bearing ones like vaultwarden/nextcloud/immich, because the backup-before-update and "
        "auto-rollback-on-failed-health-check safety net makes that safe), the ADS-B Raspberry Pi's OS "
        "packages and receiver services, and Home Assistant's own updates via its update.* entities. "
        "UniFi (the network controller/gateway) is permanently out of scope - you have no tool that "
        "touches it, never attempt to. QNAP's OS/firmware is also out of scope (its containers are fine).\n\n"
        "Your actual job is keeping these services running reliably with minimal loss - not mechanically "
        "executing the steps below like a fixed script. The steps are how to do that, not a substitute for "
        "thinking about each situation on its own terms. Two different failure shapes need two different "
        "responses, and telling them apart is the actual work:\n"
        "- Something is DOWN RIGHT NOW and was working before (a container that crashed, a service that's "
        "unreachable): act immediately, that's what protects uptime. Don't wait-and-see on an active outage.\n"
        "- A signal is AMBIGUOUS or NOT URGENT (an automation flag, a resource-growth trend, an update's "
        "changelog, a restart that didn't obviously fix anything): investigate before acting or alerting - "
        "check the entity's own history, check whether a device/sensor was really unavailable rather "
        "than the thing you're monitoring actually being broken, check whether a condition plausibly "
        "explains it. You run again in " + str(CYCLE_INTERVAL_MINUTES) + " minutes with fresh data - for "
        "anything non-urgent and ambiguous, that recheck is available to you and is often the more correct "
        "move than acting on a single incomplete reading right now. This applies everywhere below a flag "
        "or anomaly is involved, not just the automation-health steps.\n"
        "- And when the SAME response has already been tried and hasn't worked (a container you've "
        "restarted repeatedly that keeps going unhealthy again, an update that's failed the same way more "
        "than once), repeating it again isn't the aware move - that's a pattern worth surfacing to a human "
        "with what's actually been tried, not another blind repeat of the same fix. check_restart_loop "
        "exists for exactly this on the container side (see step 1).\n\n"
        "Your job this cycle:\n"
        "0. Call check_proxmox_infra_status first, before anything else. If it flags anything, call "
        "start_proxmox_lxc_or_vm for it immediately (low-risk - it only brings up something already meant "
        "to run continuously) and notify_human about it. Doing this before any other step matters: a "
        "stopped LXC/VM can take down services other steps below depend on (e.g. docker-lxc also hosts "
        "ntfy, so leaving it down would silently swallow every notify_human call for the rest of this "
        "cycle) - fixing it first gives the rest of the cycle, including this one's own notifications, a "
        "chance to actually work. Also call check_qnap_app_reachable once this cycle; if it flags "
        "anything, use judgment before notifying - a NAS can look briefly unreachable while really busy "
        "(e.g. reindexing right after its own reboot) without being broken, so don't alert on a single "
        "reading immediately after a host-wide outage.\n"
        "1. For every container in the 'unhealthy' list below (already filtered for you - these have "
        "restart_policy always/unless-stopped, meaning they're SUPPOSED to be running continuously, but "
        "currently aren't, or are failing their own HEALTHCHECK): this is an active outage, so restart it "
        "now - don't wait-and-see here, that would cost real uptime for no benefit. But first call "
        "check_restart_loop for it - if it's already been restarted 3+ times in the last 6 hours, this is a "
        "crash loop, not a one-off; restarting a 4th time on the same footing is unlikely to help and isn't "
        "the aware move. In that case still call get_logs to see what's actually happening, but lead with "
        "notify_human (urgent=true) describing the loop and what the logs show, and use judgment on whether "
        "restarting once more while you wait for a human is still worth it or just delays the real fix. For "
        "a first/second occurrence, the normal path applies: call restart_container, then get_logs to "
        "understand why, and notify_human (urgent=true).\n"
        "2. Containers in the 'unmanaged' list have restart_policy 'no' - this means either a one-shot job "
        "that already ran to completion (e.g. hello-world exiting 0 is CORRECT, not a failure) or something "
        "a human intentionally stopped. NEVER call restart_container, apply_update, or anything else on "
        "anything in the unmanaged list, regardless of what state it's in. Leave it exactly as you found it.\n"
        "3. (ADS-B is not managed right now - ignore it.)\n"
        "4. If HA is unreachable, notify_human urgently - do not attempt any HA update in that state.\n"
        "5. Call check_automation_health once (every cycle, not just update-check cycles - this is cheap "
        "and freshness matters). It now returns two kinds of flag, distinguished by an 'issue' field:\n"
        "   - issue='sensor_stuck': the trigger sensor has been bit-for-bit frozen for 3+ hours (checked "
        "automatically for every numeric_state trigger, even ones that never crossed a threshold - a value "
        "frozen INSIDE an automation's dead zone, e.g. an illuminance sensor stuck at 139 when the automation "
        "only reacts below 80 or above 150, produces no threshold-crossing miss to detect any other way). "
        "This is an unambiguous finding, not a maybe - being 'available' is NOT the same as being correct, "
        "confirmed real twice (2026-09-24 stuck at 237.0, 2026-09-27 stuck at 139.0, both on a lighting "
        "automation's illuminance sensor). Judge only whether THIS particular sensor is one that should "
        "naturally fluctuate (illuminance, power draw, temperature - stuck is suspicious) vs. one that can "
        "properly hold a value for a while (a setpoint, a binary state with no reason to have changed); "
        "if the former, notify_human directly - no need to re-call check_sensor_value_stuck yourself, it's "
        "already been run. Topic 'sensor-stuck:<entity_id>' so it doesn't re-alert every cycle for the same "
        "still-unresolved one.\n"
        "   - issue='missed_trigger': the trigger entity crossed into relevant territory more recently than "
        "the automation's last_triggered. Use judgment here: does this look like a real miss, or could a "
        "condition plausibly have blocked it properly (e.g. 'only when dark', 'only when home')? Before "
        "concluding it's a real miss, also consider whether the trigger entity itself might explain it - call "
        "ha_get_history or ha_get_logbook on it if the flag looks ambiguous and check whether it went "
        "'unavailable'/'unknown' around the flagged time (a device/sensor outage, not a broken automation). "
        "If the entity is STILL unavailable right now, there's nothing actionable yet either way - don't "
        "notify prematurely on an outage that may resolve on its own; this same check runs again next cycle "
        "with fresh data, so let it. The one exception: if history shows the SAME entity unavailable across "
        "several consecutive cycles (a prolonged/repeated outage, not a one-off blip), that pattern itself is "
        "worth a notify about the device/sensor being flaky - a different, still-real finding from 'this "
        "automation is broken'. If it looks like a real miss with no outage explanation, notify_human with "
        "the automation name, what changed, and when - topic 'automation-silent:<automation entity_id>' so it "
        "doesn't re-alert every cycle for the same still-unresolved one. Don't notify for every flag "
        "reflexively - if it's plausibly just a condition working as designed, skip it.\n"
        "5b. Also call check_expected_automations_enabled once every cycle (cheap, same reasoning as above). "
        "Unlike check_automation_health, any flag here is unambiguous - there's no legitimate reason an "
        "automation on the always_enabled_automations list should be off or missing, so notify_human urgently "
        "for every flag (topic 'automation-disabled:<entity_id>' to dedupe repeats).\n"
        + ("6-0. FIRST call scan_container_updates ONCE. It checks all managed containers (both same-tag new builds and "
           "newer upstream tags) in parallel and returns only those with something available. Do NOT call "
           "check_for_update or check_upstream_tag_update per container. Then apply steps 6 and 6b below ONLY to "
           "the containers it lists in with_updates (changelog judgment first, then apply/bump or notify).\n"
           "6. For every container in the 'managed' list (restart_policy always/unless-stopped - these are "
           "the ones actually meant to run continuously), call check_for_update. If an update is available, "
           "call check_changelog before applying it. If check_changelog comes back available=true and the "
           "excerpt reads as a likely-breaking change (major version jump, migration/upgrade steps required, "
           "removed or renamed config, database schema changes, 'BREAKING' language) - do NOT call "
           "apply_update. Instead call notify_human with a topic of 'breaking-update:<container>', "
           "summarizing what changed and why you're holding off, so a human decides when to apply it "
           "manually. If check_changelog comes back available=false, or available=true but nothing in it "
           "reads as breaking, proceed with apply_update as normal (the backup+health-check+rollback net is "
           "the safety mechanism for those). Do this for ALL managed containers, not a sample. Never call "
           "any of this on anything in the 'unmanaged' list.\n"
           "6b. Also, once per managed container per check_updates cycle, call check_upstream_tag_update - "
           "this is separate from check_for_update and won't overlap with it: check_for_update only ever "
           "notices a new build under the SAME pinned tag, so a container pinned to e.g. 'nextcloud:32' "
           "would otherwise silently never surface that '33' exists. If it reports available=true, call "
           "check_upstream_tag_changelog on the candidate_image before doing anything else. Actually read "
           "the changelog_excerpt and reason about whether it looks breaking (major version jump, migration "
           "steps, removed/renamed config, data format changes) - the same judgment call as step 6's "
           "check_changelog, not a keyword match. If it reads as breaking, do NOT call "
           "bump_compose_tag_and_apply - notify_human instead with topic 'tag-update-available:<container>', "
           "the current and candidate tags, and why you're holding off. If check_upstream_tag_changelog came "
           "back unavailable (no changelog signal - the common case, most images don't carry the metadata), "
           "that alone should only hold you back for a MAJOR version jump (e.g. nextcloud 32->35, mariadb "
           "11->13) - for those, notify_human the same way (tags + 'no changelog was available, review "
           "before bumping a major version'). For a patch/minor bump (e.g. 2.45.0->2.45.1) with no "
           "changelog available, proceed anyway - do not hold back routine low-risk bumps just because the "
           "image lacks version-label metadata. Either way, if it clearly reads as safe, or is a patch/minor "
           "bump with nothing available to read, call bump_compose_tag_and_apply - its own revert-on-failure "
           "logic and apply_update's backup+health-check+rollback are the safety net, same as step 6.\n"
           "6c. Also call apply_ha_update for any HA update.* "
           "entity check_ha_updates reported as pending (Core/Supervisor/OS/add-ons all in scope; "
           "apply_ha_update handles its own pre-update backup and the special no-blind-retry failure path "
           "internally). Also run docker_cleanup once per host for housekeeping, and check_resource_limits "
           "on a couple of managed containers per host, notifying if anything is flagged.\n"
           "7. Call check_host_resource_trend once per docker host. If it reports 'ceiling_exceeded', "
           "notify_human urgently with the topic 'resource-ceiling:<host>'. If it reports any 'flags' "
           "(containers with significant memory growth), think about whether moving that container to the "
           "workstation would help - it has the most spare CPU/RAM/disk headroom in this fleet and is the "
           "only host with a GPU. Weigh this against the container actually holding data (migrating "
           "vaultwarden/nextcloud/immich means copying volumes, not just restarting somewhere else - more "
           "involved and riskier than a stateless service) and whether it needs anything workstation-"
           "specific. There is NO tool to actually migrate a container - if you think a move is worth doing, "
           "call notify_human with a specific, reasoned recommendation (name the container, its current vs. "
           "previous memory usage, and why workstation specifically) using topic 'migration-candidate:"
           "<container>'. Never take any other action based on resource trend data.\n"
           if check_updates_this_cycle else
           "6. Do NOT check for or apply container/HA updates or run cleanup this cycle - only handle "
           "unhealthy managed containers, the ADS-B Pi, and HA reachability.\n")
        + "Only use the tools provided. Do not guess container/service names that weren't shown to you. "
        "Work through the fleet systematically. When notifying about a container/update failure that could "
        "still be failing next cycle too (not yet fixed), always pass notify_human's 'topic' argument (e.g. "
        "'update-fail:<container>') so the human isn't re-alerted every 20 minutes about the same unresolved "
        "thing - this matters, it has caused real notification spam before. When you have finished handling "
        "everything that needs attention, respond with a final plain-text summary and no more tool calls."
    )

    user_content = "Full fleet snapshot (for context only - use the managed/unhealthy/unmanaged lists below to decide actions, not this raw list):\n" + json.dumps(snapshot, indent=1)
    if ADSB_ENABLED:
        user_content += "\n\nADS-B Pi service health:\n" + json.dumps(adsb_health, indent=1)
    user_content += "\n\nHome Assistant reachability:\n" + json.dumps(ha_health, indent=1)
    user_content += "\n\nHome Assistant pending updates:\n" + json.dumps(ha_updates, indent=1)
    user_content += "\n\nMANAGED containers (restart_policy always/unless-stopped - update/cleanup candidates):\n" + json.dumps(managed, indent=1)
    user_content += "\n\nUNMANAGED containers (restart_policy 'no' - NEVER restart, update, or start these):\n" + json.dumps(unmanaged, indent=1)
    if unhealthy:
        user_content += "\n\nUNHEALTHY managed containers needing restart:\n" + json.dumps(unhealthy, indent=1)
    else:
        user_content += "\n\nAll managed containers report running/healthy."

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

    actions_taken, aborted = run_model_loop(messages)

    log.info(f"=== cycle end, {len(actions_taken)} tool calls ===")
    return {
        "actions": actions_taken, "snapshot": snapshot, "managed": managed,
        "unmanaged": unmanaged, "unhealthy": unhealthy,
        "ha_health": ha_health, "adsb_health": adsb_health, "aborted": aborted,
    }


def check_only_os_hosts():
    """These hosts never get apply_os_updates called by the model (the
    tool itself refuses), but we still want a periodic report if patches
    are piling up, since nothing else will ever tell the human."""
    for host in OS_CHECK_ONLY_HOSTS:
        result = check_os_updates(host)
        if isinstance(result, dict) and result.get("upgradable_count", 0) > 15:
            notify_human(
                f"📋 {host} has {result['upgradable_count']} pending OS updates",
                f"This host is intentionally not auto-patched (single point of failure). "
                f"Worth a manual look: {', '.join(result['packages'][:10])}",
            )


TOOL_RESULT_MAX_CHARS = 6000       # one tool output
CONVERSATION_MAX_CHARS = 60000     # whole chat; ~20k tokens, well inside num_ctx


def _clip_tool_result(result):
    text = json.dumps(result, default=str)
    if len(text) > TOOL_RESULT_MAX_CHARS:
        text = text[:TOOL_RESULT_MAX_CHARS] + f'... [clipped {len(text) - TOOL_RESULT_MAX_CHARS} chars]'
    return text


def _prune_messages(messages):
    """Ollama silently drops the oldest messages when the prompt overflows num_ctx, which can remove the
    original user request and produce HTTP 500 'no user query found'. Shrink the OLDEST tool outputs first
    so the system prompt and the fleet snapshot (messages[0:2]) always survive."""
    def total():
        return sum(len(str(m.get("content", ""))) for m in messages)
    log.info("conversation size: %d chars, %d messages", total(), len(messages))
    for m in messages[2:]:
        if total() <= CONVERSATION_MAX_CHARS:
            break
        if m.get("role") == "tool" and len(str(m.get("content", ""))) > 200:
            m["content"] = "[older tool output trimmed to save context]"


def _immich_gate(action):
    # Immich ML and Ollama share one iGPU; yield it during the model-driven cycle.
    try:
        run("workstation", f"{IMMICH_GATE} {action}", timeout=30)
    except Exception as e:
        log.warning("immich-gate %s failed: %s", action, e)

def main():
    if datetime.now().hour not in HEAVY_CYCLE_HOURS:
        return _main_inner()
    _immich_gate("pause")
    try:
        return _main_inner()
    finally:
        _immich_gate("resume")

def _main_inner():
    setup_logging()
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()

    is_heavy_hour = datetime.now().hour in HEAVY_CYCLE_HOURS

    last_check = state.get("last_update_check")
    check_updates_this_cycle = True
    if last_check:
        last_dt = datetime.fromisoformat(last_check)
        check_updates_this_cycle = datetime.now(timezone.utc) - last_dt > timedelta(hours=24)
    # Updates/LTS/OS-patching are deferred-decision work, same reasoning as
    # the rest of the heavy cycle - never run them outside HEAVY_CYCLE_HOURS
    # even if the 24h timer would otherwise be due, so they can't land in
    # the middle of the day either.
    check_updates_this_cycle = check_updates_this_cycle and is_heavy_hour

    publish_status_to_ha("running", state)

    try:
        result = run_cycle(check_updates_this_cycle) if is_heavy_hour else run_lightweight_cycle()
    except Exception as e:
        log.exception("cycle failed")
        notify_human("🔴 Ops agent cycle crashed", f"{e}", urgent=True)
        publish_status_to_ha("error", state, error=str(e))
        return

    publish_status_to_ha(
        "idle", state, snapshot=result["snapshot"], managed=result["managed"],
        unmanaged=result["unmanaged"], unhealthy=result["unhealthy"],
        ha_health=result["ha_health"], adsb_health=result["adsb_health"],
        actions=result["actions"],
    )

    # Unconditional (every cycle, not gated like the blocks below) so the
    # admin UI's container list / button targets never go more than one
    # cycle (20 min) stale - this is read-only display data, not behavioral
    # state, so there's no reason to gate it.
    state["last_cycle_at"] = datetime.now(timezone.utc).isoformat()
    state["last_snapshot"] = result["snapshot"]
    state["last_managed"] = result["managed"]
    save_state(state)

    if not is_heavy_hour:
        return

    if check_updates_this_cycle and result.get("aborted"):
        attempts = state.get("update_attempts", 0) + 1
        state["update_attempts"] = attempts
        if attempts < 3:
            log.warning("update check did not complete (attempt %d/3) - will retry next cycle", attempts)
            save_state(state)
            check_updates_this_cycle = False
        else:
            notify_human("Ops agent update check keeps failing",
                         "3 attempts in a row ended early (model errors/time budget). Will try again in 24h.",
                         topic="update-check-failing")
    if check_updates_this_cycle:
        state["update_attempts"] = 0
        state["last_update_check"] = datetime.now(timezone.utc).isoformat()
        for host in OS_AUTO_HOSTS:
            os_check = check_os_updates(host)
            if isinstance(os_check, dict) and os_check.get("upgradable_count", 0) > 0:
                apply_os_updates(host)
        check_only_os_hosts()
        save_state(state)

    # LTS check is cheap but pointless more than weekly - Canonical doesn't
    # publish new LTS releases on any schedule this agent needs to poll for
    # more often than that.
    last_lts_check = state.get("last_lts_check")
    if not last_lts_check or datetime.now(timezone.utc) - datetime.fromisoformat(last_lts_check) > timedelta(days=7):
        state["last_lts_check"] = datetime.now(timezone.utc).isoformat()
        lts = check_ubuntu_lts_available("workstation")
        if isinstance(lts, dict) and lts.get("new_lts_available"):
            notify_human(
                f"🐧 Ubuntu {lts['latest_lts_version']} ({lts['latest_lts_codename']}) is out",
                f"workstation is currently on {lts['current_version']} ({lts['current_codename']}). "
                f"A new LTS is published - this agent will never run the release upgrade itself "
                f"(kernel/driver/GPU-stack risk is too high to automate), just flagging it's available "
                f"whenever you want to plan it.",
                topic="new-lts-available",
            )
        elif isinstance(lts, dict) and "error" in lts:
            log.warning(f"LTS check failed: {lts['error']}")
        save_state(state)


if __name__ == "__main__":
    main()
