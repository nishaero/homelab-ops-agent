#!/usr/bin/env python3
"""
Deletes Home Assistant backups older than RETENTION_DAYS, via HA's websocket
admin API (backup.create_automatic is the only plain service HA exposes for
backups - list/delete are websocket-only, no REST equivalent).

Built after HAOS's data disk hit 1.6GB free: HA's own built-in retention
(retention.days, set via backup/config/update) already prunes on this same
age basis, but this runs independently on Proxmox as a second enforcement
layer in case that internal mechanism silently stops working again - it
should mostly find nothing to do, and that's the expected/healthy case.

Runs as ha-backup-cleanup.timer on Proxmox (daily). See ops-agent.env for
the shared OPS_AGENT_HA_TOKEN this reuses.
"""
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import websockets

HA_URL = os.environ.get("HA_WS_URL", "wss://ha.example.com/api/websocket")
TOKEN = os.environ["OPS_AGENT_HA_TOKEN"]
RETENTION_DAYS = 5


async def main():
    cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    async with websockets.connect(HA_URL, max_size=None) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "auth", "access_token": TOKEN}))
        auth = json.loads(await ws.recv())
        if auth.get("type") != "auth_ok":
            print(f"auth failed: {auth}", file=sys.stderr)
            sys.exit(1)

        await ws.send(json.dumps({"id": 1, "type": "backup/info"}))
        resp = json.loads(await ws.recv())
        backups = resp["result"]["backups"]

        old = [b for b in backups if datetime.fromisoformat(b["date"]) < cutoff]
        print(f"{len(backups)} backups total, {len(old)} older than {RETENTION_DAYS}d")

        for i, b in enumerate(old, start=2):
            await ws.send(json.dumps({"id": i, "type": "backup/delete", "backup_id": b["backup_id"]}))
            r = json.loads(await ws.recv())
            status = "ok" if r.get("success") else r.get("error")
            print(f"  deleted {b['backup_id']} ({b['date'][:10]} {b['name']}): {status}")


if __name__ == "__main__":
    asyncio.run(main())
