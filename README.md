# homelab-ops-agent

An agent that looks after my home lab: the Docker containers on three hosts,
Home Assistant, and OS patching. A local Ollama model decides what to do by
calling a fixed set of tools. No cloud model is used.

It runs from a systemd timer on the Proxmox host every 20 minutes. During the
day it only fixes clear outages. At night (00:00 to 05:00) the model checks
for updates, reads changelogs, applies updates and sends a report.

This is the public, sanitized copy of an agent that runs on a real home lab. Host names, user
names, domains and addresses in the code are placeholders. Set the real ones through the
environment variables in `.env.example` and the settings in `config.example.yaml`. A separate,
newer project, [homelab-multi-agent](https://github.com/nishaero/homelab-multi-agent), adds
specialist agents, approvals and remote access beside it.

More detail:

- [docs/architecture.md](docs/architecture.md): the home lab layout and diagram
- [docs/ai-design.md](docs/ai-design.md): how the agents work, the guard rails and the problems I hit

## How it works

1. The script collects the state of every host and container.
2. It gives that state and the tool list to the model.
3. The model calls tools until it is done (25 turns at most, with a time budget).
4. The script sends a notification for everything that mattered.

The model has no shell. It can only call the functions in `TOOLS` in
`ops_agent.py`, such as `restart_container`, `apply_update`, `check_changelog`
and `notify_human`. A wrong decision is limited to what those functions allow.

## What it manages

| | Hosts |
|---|---|
| OS patching (apt) | workstation, npm-lxc, docker-lxc |
| New Ubuntu LTS check (notify only) | workstation, weekly |
| Containers | workstation, docker-lxc (LXC 102), QNAP |
| Home Assistant | REST API only (core, OS and add-on updates) |
| Never touched | UniFi, QNAP firmware, OS patching of the Proxmox host and the NAS |

The never-touched list is code, not config. `NEVER_PATCH_HOSTS` (the Proxmox host and the NAS)
is filtered out of `os_auto_hosts` when the config is read, and `apply_os_updates` refuses those
hosts a second time, so editing the config cannot enable patching them. A test starts the agent
with a config that lists them and checks they are dropped. The Proxmox host is the hypervisor
for everything else, and UniFi is the network, so a bad action on either could cut the agent off.

The ADS-B Pi is switched off for now. Set `adsb_enabled: true` in the config to
bring it back.

## Safety

- **Updates**: compose projects use `docker compose pull` and `up -d`, which
  keeps volumes and bind mounts. Containers without compose are recreated from
  a saved `docker inspect`: the old container is stopped and renamed to
  `<name>.ops-old` (not deleted), the new one is created with a faithful copy of
  its configuration, and a parity check compares the two (command, networks,
  ports, mounts, devices, privileges, user). If creation or the check fails, the
  old container is put back untouched. A healthy update removes the preserved one.
- **Backups**: containers that hold data (Vaultwarden, Nextcloud, Immich) get
  their bind mounts tarred first. Mounts that are too large are skipped and the
  log says so.
- **Rollback**: the running image is tagged `:ops-agent-rollback` before every
  update. If the health check fails, rollback tries three things in order:
  1. restore the preserved `<name>.ops-old` container, which has the exact
     original configuration;
  2. for compose projects, re-tag the saved image and force-recreate the service
     without pulling, but only when the compose file still names the original
     image (otherwise compose would read the `image:` line and ignore the tag);
  3. recreate from the saved inspect data.
  An early version reported a good rollback while the broken image kept running,
  and a later one dropped a container's command and network when it recreated it.
  Both are why the parity check and the preserved container exist.
- **Crash detection**: only containers with `restart: always` or
  `unless-stopped` count as crashed. One-shot jobs and containers I stopped on
  purpose are left alone.
- **Changelog check**: before an update, `check_changelog` looks up the release
  notes when the image has OCI labels. If the change looks breaking (major
  version, migration steps, renamed settings), the agent notifies and waits.
- **Home Assistant**: only through its API, never SSH. A backup is taken before
  each update. If Home Assistant does not come back, the agent sends one urgent
  notification naming the backup and stops. It does not try to restore on its
  own, because its only way to reach Home Assistant is the API that is down.
- **Home Assistant control**: looking at any entity is allowed. Calling
  services is limited to safe domains (lights, switches, climate, fans, media
  players, input helpers, timers and notify), one named entity of the same
  domain per call, with no target keys (`entity_id`, `area_id` and so on) in the
  extra data. Locks, covers and alarm panels are not in the list. Scripts and
  scenes are not in it either, because a script can call `lock.unlock`: each one
  is callable only if you list it under `ha_allowed_entities` in the config
  after reading it. Note that a switch or a helper wired to something dangerous
  is still a switch, so keep unlock paths out of the safe domains.
- **Automations**: only the chat sidecar can create them, and only after I
  confirm in a later message. The autonomous cycle cannot.
- **Reports only**: resource growth, restart loops and Ubuntu releases are
  reported. The agent has no tool to act on them.
- **Argument checks**: everything the model passes to a tool is untrusted, since a
  changelog or a log could carry an injected instruction. Container names, image
  references, tags, service names and entity ids must match strict patterns
  before any tool runs, and the QNAP command quotes its URL and body as single
  shell arguments. A test tries names with quotes, semicolons, newlines and
  `$(...)`.
- **Limits**: the same tool call is stopped after two tries, tool results are
  clipped, and a cycle that runs out of time stops and keeps what it finished.
  A cycle that ends early does not count as the daily update check. It is
  retried, and after three failures the agent notifies once.

## Configuration

Secrets are in a `.env` file loaded by systemd. Everything else is in
`ops-agent-config.yaml`. See `config.example.yaml`. Missing keys fall back to
defaults, so a missing file does not stop the agent.

Useful settings: `model`, `ollama_url`, `os_auto_hosts`, `docker_hosts`,
`heavy_cycle_start_hour`, `heavy_cycle_end_hour`, `cycle_budget_seconds`,
`always_enabled_automations`, `adsb_enabled` and the `ntfy` block.

The list of hosts (which ones are SSH or `pct exec` targets, how the QNAP
Docker socket is reached) is in `ops_agent.py`. This is a tool for one home
lab, not a general product.

```bash
cp .env.example ops-agent.env
cp config.example.yaml ops-agent-config.yaml
sudo cp ops-agent.env ops-agent-config.yaml /root/
sudo chmod 600 /root/ops-agent.env /root/ops-agent-config.yaml
sudo cp ops_agent.py /root/ops_agent.py
sudo cp ops-agent.service ops-agent.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ops-agent.timer
```

The config file holds the ntfy topic, so treat it like a secret. SSH access to
the hosts uses the normal `~/.ssh/config` and keys of the account that runs the
agent. No SSH credentials are stored here.

## Memory

`remember_note` saves a short note about what the agent did or what failed.
`recall_past` reads old notes before a cycle starts. Notes live in a separate
small service (agent-memory) reached through `OPS_AGENT_MEMORY_URL` and
`OPS_AGENT_MEMORY_TOKEN`. Without those the agent still runs and just has no
notes.

## Sharing the GPU

The workstation GPU is shared with the voice assistant, doorbell descriptions
and Immich. The agent waits for Ollama to be idle before each model call and
pauses the Immich queues before the heavy cycle with `immich-gate`. It resumes
them at the end, even if the cycle fails. See docs/ai-design.md.

## Notifications (ntfy)

Alerts go to a self-hosted [ntfy](https://ntfy.sh) container, separate from Home
Assistant's own notifications. Use a long random topic name. ntfy has no
discovery, so the name is the secret. Subscribe in the ntfy app using the
public HTTPS address.

`ntfy.base_url` is the address the agent publishes to. If the agent and ntfy
are on the same network, use the LAN address. That avoids a trip through the
public internet and still works when the public DNS or certificate is broken.

Urgent notifications use ntfy priority 5 and break through iOS Focus modes.
Routine reports use priority 3, which iOS can fold into a summary if you turn
on Scheduled Summary for ntfy in iOS settings.

## Secrets in Vaultwarden

The files above are what the services read at run time. Their values come from
Vaultwarden (for example `vault.example.com`), one Secure Note per service in a
`homelab-secrets` folder, with the secrets as hidden fields.

`sync_secrets_from_vault.py` runs on Proxmox. It reads those notes and writes
the right files on each host. It writes a local file and copies it over. It
never puts a secret inside a remote shell command, because some values contain
`$`. The basic-auth password is not written to a file. The script hashes it and
adds it through the API sidecar's `/users` endpoint.

```bash
export BW_CLIENTID=... BW_CLIENTSECRET=...
bw config server https://your-vaultwarden-instance
bw login --apikey

export BW_SESSION="$(bw unlock --raw)"   # asks for the master password
python3 sync_secrets_from_vault.py            # all services
python3 sync_secrets_from_vault.py ops-agent  # one service
bw lock
```

This step is manual on purpose. The API key can log in but cannot decrypt the
vault, so unlocking needs the master password. A tool built for machine secrets
would be the right choice if this ever has to run unattended.

## Tests

```bash
pip install pyyaml pytest
pytest
```

119 tests cover what does not need a real host:

- the hard exclusions, including a config that lists the Proxmox host and the NAS;
- argument validation and shell quoting, with hostile container names;
- Home Assistant limits (safe domains, scripts and scenes only when listed, same domain entity, no retargeting);
- the model loop: an identical call is refused after two tries, the turn limit, the time budget, a failing
  model call, unknown tools, tools that raise, and clipping of large output;
- pure helpers: image reference parsing, newer tag selection and container parity;
- no shell tool, every tool has an implementation, pruning, and notification dedupe.

The tests that cover the fixes fail against the code before them. The update and rollback sequence
itself (pull, recreate, health check, restore) is checked on the live system, not by automated tests.

## Limits

- The repeat cap stops an identical tool call after two tries. It does not detect a model that
  changes its arguments slightly. The turn limit and the time budget are the backstop.
  `homelab-multi-agent` has stronger loop detection.
- The 190 KB main file is hard to review. Splitting it into tools, guard rails and cycles is planned.
- Prompt injection from changelogs and logs has not been tested systematically.

## Admin UI

`ops-ui/` is a small Streamlit app with a config form, a log and history
viewer, a chat and container buttons. It talks to two services on the same host
as the agent:

- `ops_agent_api.py` reads and writes a small allowed set of config values and
  serves history and logs. It does not import `ops_agent.py` and holds none of
  its credentials.
- `ops_agent_chat.py` is the chat and button backend. It imports `ops_agent.py`
  and runs the same tools right away.

**Chat tab**: a chat with the same local model and nearly the same tools as the
autonomous cycle, without `notify_human`. Ask a question and it checks live
state first. Ask it to fix or update something and it does it, with the same
changelog check. If you ask for something it has no tool for, such as new code
or a new service, it says so.

**Containers tab**: the Restart and Update buttons call `POST /action` on the
chat service. There is no model involved, but backups, health checks, rollback
and changelog gating still apply. `ACTION_ALLOWLIST` in `ops_agent_chat.py`
lists the two tools a button can trigger.

**Config is limited on purpose.** `EDITABLE_KEYS` in the API service allows only
timing and tuning values. It does not allow hosts or containers, and the hard
exclusions are not config at all. Edit anything else by hand over SSH.

**Protect the sidecars.** The chat service can restart and update containers on
request. Allow only the UI container to reach it (a firewall rule for one source
IP, plus the bearer token on each sidecar). The UI needs its own login as well,
because its port is often open on the whole LAN. The GitHub OAuth check in
`app.py` is that login. If you redeploy `app.py` from an old checkout, compare it
with the running copy first. The panel once lost its login for a while that way.

```bash
# API service, on the same host as ops_agent.py
cp ops-agent-api.env.example ops-agent-api.env
sudo cp ops-agent-api.env ops_agent_api.py /root/
sudo cp ops-agent-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ops-agent-api.service

# Chat service, same host. It shares ops-agent.env, add OPS_AGENT_CHAT_TOKEN there
sudo cp ops_agent_chat.py /root/
sudo cp ops-agent-chat.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ops-agent-chat.service

# UI, on any host with Docker
cd ops-ui
cp .env.example .env     # sidecar URLs and tokens, GitHub OAuth app
docker compose up -d --build
```

If the UI and the sidecars are on different hosts behind a host firewall such
as `pve-firewall`, check that there is an ACCEPT rule for each sidecar port from
the UI host. Port 8099 had one and port 8100 did not, so every chat request was
dropped and it looked like a broken chat box. Run
`pct exec <ui-lxc-id> -- nc -zv <sidecar-host-ip> <port>` to tell a network
problem from a code problem.

**Login**: GitHub OAuth is the main login. A username and password form shows
up only after at least one basic-auth user exists. It is there for when GitHub
is unreachable and you need in over the LAN.

Allowed users are managed in the UI (Config tab, Login access). They are saved
by `ops_agent_api.py` in `/var/lib/ops-agent/ui-users.json` (mode 600), so they
survive a rebuild of `ops-ui`. The values in `ops-ui/.env`
(`GITHUB_ALLOWED_LOGIN`, `BASIC_AUTH_USERNAME`, `BASIC_AUTH_PASSWORD_HASH`) are
only a first-run seed. If the list is ever emptied, the next page load seeds it
again from those values, so you cannot lock yourself out.

Passwords added in the UI are hashed with salted PBKDF2 in the UI's server code
before they go to the API. To make the first hash by hand:

```bash
python3 -c "
import base64, getpass, hashlib, os
salt = os.urandom(16)
pw = getpass.getpass('Password: ').encode()
h = hashlib.pbkdf2_hmac('sha256', pw, salt, 600000)
print(f'pbkdf2_sha256\${600000}\${base64.b64encode(salt).decode()}\${base64.b64encode(h).decode()}')
"
```

Neither login has its own rate limiting, so keep the panel on the LAN and use
strong passwords. The UI has CSS for narrow screens, and the chat tab is a
centered column with a scrolling history and a few suggested questions.

## Requirements

- Python 3 with PyYAML (the API sidecar uses only the standard library)
- A running Ollama with a model that supports tool calls
- Passwordless `sudo` for `apt` on the hosts that get OS patching
- A self-hosted ntfy (or ntfy.sh) and its mobile app
- Docker for the admin UI (optional)
