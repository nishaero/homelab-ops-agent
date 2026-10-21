# Home lab architecture

This is the current layout of the home lab that the ops agent looks after.
Last updated 2026-10-21.

```mermaid
flowchart TB
    Internet((Internet))
    CF["Public DNS"]

    subgraph UCG["UCG Max (gateway, firewall, VPN)"]
        Default["Default network"]
        IoT["IoT VLAN\nIoT Wi-Fi"]
        VPN["WireGuard server\nTeleport"]
    end

    subgraph Proxmox["Proxmox host (pve)"]
        NPM["LXC 101\nnginx-proxy-manager"]
        subgraph LXC102["LXC 102 docker-lxc"]
            Mon["Prometheus, Grafana,\nAlertmanager, blackbox"]
            Ntfy["ntfy"]
            OpsUI["ops-ui"]
            Misc["Vaultwarden, Portainer,\nbentopdf"]
        end
        HAOS["VM 100\nHome Assistant OS\n(USB Wi-Fi stick on the IoT VLAN)"]
        Agent["ops-agent\nsystemd timer, every 20 min"]
        Cleanup["ha-backup-cleanup.timer"]
    end

    subgraph WS["Workstation (Ubuntu, Ethernet, AMD iGPU)"]
        Ollama["Ollama\n(shared GPU)"]
        OWUI["Open WebUI + mcpo"]
        Immich["Immich\nserver, ML, postgres, redis"]
        Gate["immich-gate\nimmich-yield (temporary)"]
        Face["ha-face\ndoorbell face service"]
        Mem["agent-memory\nAPI + MCP"]
        NC["Nextcloud stack"]
        Exp["exporters"]
    end

    subgraph NAS["QNAP NAS"]
        Plex["Plex"]
        SMB["SMB shares"]
    end

    IoTDev["IoT devices\nAV receiver, TV, doorbell camera,\nplugs, chargers, sensors"]
    DR[("homelab-dr\nprivate GitHub repo +\nexternal drive")]

    Internet --> CF --> UCG
    UCG -->|"ports 80 and 443 only"| NPM
    NPM --> HAOS
    NPM --> OpsUI
    NPM --> Ntfy
    NPM --> Misc
    NPM --> NC
    NPM --> Immich
    IoT --- IoTDev
    HAOS ---|"Wi-Fi stick"| IoT
    VPN -.->|"remote access to Plex and the rest"| Default

    Agent -->|"checks and fixes"| LXC102
    Agent --> HAOS
    Agent --> WS
    Agent --> NAS
    Agent -->|"model calls"| Ollama
    Agent -->|"alerts"| Ntfy
    Agent <-->|"notes"| Mem
    Agent -->|"pause and resume at night"| Gate

    HAOS -->|"events"| Face
    Face -->|"faces"| Immich
    Face -->|"identity"| HAOS
    HAOS -->|"voice and descriptions"| Ollama
    Gate -->|"pauses Immich queues"| Immich
    Gate -.->|"watches doorbell and voice"| HAOS
    OWUI --> Ollama
    Immich --> SMB

    Mon -->|scrapes| Exp
    Mon -->|scrapes| HAOS
    DR -.->|"daily backup"| WS
```

## Hosts

- **UCG Max**: gateway, firewall and VPN. It routes four networks: Default,
  IoT, Work and Guest. Public services are published only through
  nginx-proxy-manager. The agent never touches it.
- **Proxmox (pve)**: runs LXC 101 (nginx-proxy-manager),
  LXC 102 (docker-lxc, the monitoring and admin stack), VM 100 (Home Assistant
  OS) and the ops agent itself as a native systemd timer.
- **Home Assistant OS (VM 100)**: the USB Wi-Fi stick is passed
  through to the VM and joins the IoT Wi-Fi, so Home Assistant can talk to
  devices on the IoT network while the admin connection stays on the wired network.
- **Workstation**: Ubuntu on Ethernet with an AMD Strix Halo
  GPU. It runs Ollama, Open WebUI, mcpo, Immich, Nextcloud, the doorbell face
  service, the agent memory service and the metrics exporters.
- **QNAP NAS**: Plex and the SMB shares that Immich mounts.
- **ADS-B Pi**: flight tracking receiver. The agent does not manage it right
  now (adsb_enabled is false).

## Public services

All public names end at nginx-proxy-manager (LXC 101) and are proxied to a
private address. Examples: Home Assistant, the ops UI, ntfy, Vaultwarden,
Immich, Nextcloud, Grafana, Proxmox, Portainer, Open WebUI and a PDF tool.

## Resilience

- **Boot order**: all Proxmox guests start on boot. Immich waits for the NAS
  mount before it starts.
- **Self-healing**: every cycle starts by checking that the LXCs, the VM and
  the NAS apps are up, and starts anything that is down.
- **Backups**: Home Assistant keeps 5 days of backups plus a daily cleanup
  timer and a duplicate check. The `homelab-dr` repo (secrets removed) and an
  external drive (full copy) hold everything needed to rebuild.
- **GPU sharing**: the workstation GPU is shared between the ops agent, the
  voice assistant, the doorbell service and Immich. See ai-design.md.
- **Alerts**: a self-hosted ntfy instance, separate from Home Assistant
  notifications. Only a wake-up ping goes through ntfy.sh for iOS push, never
  the message text.
