#!/usr/bin/env python3
"""
Ops Agent Chat - a small, always-on HTTP service for synchronous,
interactive access to the homelab, as opposed to ops_agent.py's
autonomous 20-minute cycle. Nothing here is queued for later - a chat
reply or a button click gets acted on right now, in this request.

Runs on the SAME host as ops_agent.py (Proxmox), under the SAME
EnvironmentFile (/root/ops-agent.env) - unlike ops_agent_api.py, which
intentionally does NOT hold ops_agent.py's credentials, this process
imports ops_agent.py directly and reuses its tool implementations, so it
needs the same HA token / ntfy config ops_agent.py already has. That's a
deliberate, scoped exception: this process can synchronously execute the
same actions the autonomous cycle can (restart_container, apply_update,
with their full existing safety nets), immediately.

Two endpoints:
- POST /chat: takes the conversation so far and runs it through the same
  local Ollama model ops_agent.py uses, tool-calling against a curated
  subset of ops_agent.py's TOOL_IMPL (everything except notify_human,
  which is redundant here - the chat reply itself is the response
  channel).
- POST /action: a per-container button (Restart now / Update now) calls
  one of ACTION_ALLOWLIST's two tools directly, bypassing the model
  entirely - a button click is already a deterministic, human-confirmed
  action, no reasoning needed, but it still gets the tool's own full
  safety net (backup/health-check/rollback for an update).

What this can NOT do: build new capabilities into itself. There is no
tool for "write code," "deploy a new service," or "edit ops_agent.py" -
if asked to implement something really new, the system prompt tells
it to say so honestly and describe what it'd take, not pretend to do it.
"""
import json
import os
import secrets
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ops_agent as oa  # noqa: E402 - needs the sys.path tweak above first

TOKEN = os.environ["OPS_AGENT_CHAT_TOKEN"]  # required, shared secret with the UI
PORT = int(os.environ.get("OPS_AGENT_CHAT_PORT", "8100"))
MAX_TURNS = 12  # chat should stay snappier than run_cycle's 25-turn cap
MODEL_CALL_TIMEOUT = 180  # seconds per model call. Was 120, but structured-output turns (drafting a
# full automation config, not just a tool-call decision) measured ~115s against the local model in
# practice - too close to that ceiling for comfort. Still fails fast relative to no timeout at all.

# notify_human is redundant here since the chat reply itself IS the
# response channel - nothing else is excluded, chat has the exact same
# action authority the autonomous cycle has, just synchronous.
_EXCLUDED_TOOLS = {"notify_human"}
CHAT_TOOLS = [t for t in oa.TOOLS if t["function"]["name"] not in _EXCLUDED_TOOLS]
CHAT_TOOL_IMPL = {k: v for k, v in oa.TOOL_IMPL.items() if k not in _EXCLUDED_TOOLS}

# POST /action calls one of these directly and synchronously, bypassing
# the model entirely - a button click is already a deterministic, human-
# confirmed action, no reasoning needed. Deliberately a tiny allowlist,
# not "any tool in CHAT_TOOL_IMPL": both of these already have their own
# full safety net regardless of caller (restart is idempotent;
# apply_update still goes through check_for_update + check_changelog
# gating + backup + health-check + rollback internally) - a button
# doesn't bypass any of that, it just skips the conversational reasoning
# step for an action the human already decided on.
ACTION_ALLOWLIST = {
    "restart_container": {"host", "container"},
    "apply_update": {"host", "container"},
}

# --- Natural-language HA automation authoring, chat-only (never added to
# oa.TOOLS/oa.TOOL_IMPL) - the autonomous cycle must never gain the
# ability to create/edit automations on its own, only a human-in-the-loop
# chat session does, and only via an explicit two-step draft-then-apply
# flow (see the docstrings below and the SYSTEM_PROMPT instruction never
# to draft and apply in the same turn).
#
# The bigger gap this closes: once an automation is SAVED, it runs under
# HA's own engine with the HA token's full privileges - completely
# bypassing ha_call_service's HA_SAFE_SERVICE_DOMAINS allowlist. A drafted
# automation's actions are validated against that SAME allowlist before
# they can ever be drafted, let alone applied - locks/covers/alarm panels
# stay unreachable through this path too, not just the direct-call one.
_AUTOMATION_DRAFTS = {}
_DRAFT_TIMESTAMPS = {}
_DRAFT_TTL_SECONDS = 3600


def _prune_drafts():
    now = time.time()
    for k in [k for k, t in _DRAFT_TIMESTAMPS.items() if now - t > _DRAFT_TTL_SECONDS]:
        _AUTOMATION_DRAFTS.pop(k, None)
        _DRAFT_TIMESTAMPS.pop(k, None)


def _extract_action_domains(actions):
    domains = set()
    for a in actions or []:
        if not isinstance(a, dict):
            continue
        if isinstance(a.get("action"), str) and "." in a["action"]:
            domains.add(a["action"].split(".", 1)[0])
        if isinstance(a.get("service"), str) and "." in a["service"]:  # legacy key, still valid HA syntax
            domains.add(a["service"].split(".", 1)[0])
        if isinstance(a.get("domain"), str):  # device-action style
            domains.add(a["domain"])
        for key in ("sequence", "then", "else", "default", "parallel", "repeat"):
            if key in a:
                nested = a[key]
                domains |= _extract_action_domains(nested if isinstance(nested, list) else nested.get("sequence", []))
        for branch in a.get("choose", []):
            domains |= _extract_action_domains(branch.get("sequence", []))
    return domains


def ha_draft_automation(config):
    """Stage a Home Assistant automation config for review - does NOT
    touch HA at all. config is a full automation definition (alias,
    triggers, conditions, actions, mode) in HA's own format - include an
    existing automation's 'id' to draft an edit to it, omit it to draft a
    new one. Rejects (with an explanation) any config whose actions call
    a service domain outside HA_SAFE_SERVICE_DOMAINS - locks, covers, and
    alarm panels can't be authored this way, same as they can't be called
    directly. Returns a draft_id and a summary - show the human the full
    config and a plain-English explanation of what it does in your reply,
    and WAIT for them to explicitly confirm in a LATER message before
    ever calling ha_apply_automation_draft. Never draft and apply in the
    same turn, no matter how clear the request seemed - that defeats the
    entire point of the review step."""
    _prune_drafts()
    if not isinstance(config, dict):
        return {"error": "config must be an object"}
    missing = {"alias", "triggers", "actions"} - set(config)
    if missing:
        return {"error": f"config is missing required keys: {sorted(missing)}"}
    unsafe = _extract_action_domains(config.get("actions")) - oa.HA_SAFE_SERVICE_DOMAINS
    if unsafe:
        return {
            "error": (
                f"this automation's actions call domain(s) outside what's allowed: {sorted(unsafe)}. "
                "Locks, covers, and alarm panels can't be authored this way - if the human needs one of "
                "those, tell them it has to be created by hand in the HA UI, not through this tool."
            )
        }
    draft_id = secrets.token_hex(4)
    _AUTOMATION_DRAFTS[draft_id] = config
    _DRAFT_TIMESTAMPS[draft_id] = time.time()
    return {
        "draft_id": draft_id,
        "mode": "update existing automation" if config.get("id") else "create new automation",
        "alias": config.get("alias"),
        "trigger_count": len(config.get("triggers", [])),
        "condition_count": len(config.get("conditions", [])),
        "action_count": len(config.get("actions", [])),
        "full_config": config,
        "note": "Not saved. Show this to the human and wait for their explicit confirmation before applying.",
    }


def ha_apply_automation_draft(draft_id):
    """Actually create/update the Home Assistant automation from a
    previously-staged draft (see ha_draft_automation). Only call this
    after the human has seen the full draft and explicitly confirmed in
    their own message - never in the same turn as drafting it. Drafts
    expire after 1 hour."""
    _prune_drafts()
    config = _AUTOMATION_DRAFTS.get(draft_id)
    if not config:
        return {"error": f"no draft with id {draft_id!r} - it may have expired (1h) or already been applied"}
    automation_id = config.get("id") or str(int(time.time() * 1000))
    config = {**config, "id": automation_id}
    try:
        oa._ha_request("POST", f"/api/config/automation/config/{automation_id}", body=config, timeout=20)
    except Exception as e:
        return {"error": str(e)}
    _AUTOMATION_DRAFTS.pop(draft_id, None)
    _DRAFT_TIMESTAMPS.pop(draft_id, None)
    return {
        "success": True,
        "automation_id": automation_id,
        "note": "Saved. Use ha_search_entities to confirm the resulting entity_id if you need it.",
    }


_AUTOMATION_TOOLS = [
    {"type": "function", "function": {
        "name": "ha_draft_automation",
        "description": (
            "Stage a Home Assistant automation config (alias/triggers/conditions/actions/mode, HA's own "
            "format) for review - does NOT touch HA. Include an existing automation's 'id' to draft an "
            "edit, omit it for a new one. Rejects configs whose actions call a domain outside the safe "
            "allowlist (no locks/covers/alarm panels this way either). Returns a draft_id - show the "
            "human the config and wait for their explicit confirmation before ever calling "
            "ha_apply_automation_draft, never in the same turn."
        ),
        "parameters": {"type": "object", "properties": {
            "config": {"type": "object", "description": "Full HA automation config: alias, triggers, conditions, actions, mode"}},
            "required": ["config"]},
    }},
    {"type": "function", "function": {
        "name": "ha_apply_automation_draft",
        "description": "Actually save a previously-drafted automation (see ha_draft_automation). Only call after the human explicitly confirmed the draft in their own message.",
        "parameters": {"type": "object", "properties": {
            "draft_id": {"type": "string"}},
            "required": ["draft_id"]},
    }},
]
_AUTOMATION_TOOL_IMPL = {
    "ha_draft_automation": ha_draft_automation,
    "ha_apply_automation_draft": ha_apply_automation_draft,
}
CHAT_TOOLS = CHAT_TOOLS + _AUTOMATION_TOOLS
CHAT_TOOL_IMPL = {**CHAT_TOOL_IMPL, **_AUTOMATION_TOOL_IMPL}

SYSTEM_PROMPT = (
    "You are a conversational assistant for a homelab's infrastructure - the same fleet "
    "ops_agent.py maintains autonomously every 20 minutes (docker containers across the "
    "workstation, docker-lxc, and qnap hosts, Home Assistant, and OS packages on the "
    "workstation + ADS-B Pi). Right now you're answering a direct question or handling a "
    "direct request in a live conversation, not running a scheduled cycle - answer "
    "conversationally, and use tools to check LIVE state before answering anything you're "
    "not certain of, rather than guessing or relying on stale assumptions.\n\n"
    "If asked to fix, restart, or update something, use the tools to do it now, directly in "
    "this conversation - there is no queue here, you have the tools right now, don't say "
    "you'll get to it later. Updates still go through check_for_update -> check_changelog -> "
    "apply_update (hold off and explain if the changelog looks breaking, same policy as the "
    "autonomous cycle) - a direct request doesn't skip that safety check.\n\n"
    "Scope is identical to the autonomous cycle: UniFi, QNAP's OS/firmware, and OS-level "
    "patching of Proxmox or its LXCs are permanently out of scope - you have no tool that "
    "touches them, say so plainly if asked rather than attempting a workaround. The 'no "
    "manual restart_policy=no containers' rule also still applies.\n\n"
    "You also have general Home Assistant visibility and limited control, beyond just HA's "
    "own updates: ha_get_state/ha_get_history/ha_get_logbook are read-only and unrestricted - "
    "use them to actually check what's going on (an entity's current value, whether it changed "
    "when it should have, whether an automation actually fired) rather than guessing, the same "
    "way you'd debug anything else here. ha_call_service can control lights, switches, climate, "
    "fans, media players, timers and input_* helpers directly (scripts and scenes only if named in the config) - but it flatly refuses "
    "any domain outside that allowlist. Locks, covers (garage doors), and alarm panels are NOT "
    "reachable through this tool at all - if asked to touch one of those, say plainly that you "
    "don't have a path to it, don't look for a workaround.\n\n"
    "If asked to build or implement something you don't have a tool for (a new dashboard, a "
    "new service, a code change to how you or ops_agent.py works) - be honest that this is "
    "outside what you can do through this chat. You cannot write code or deploy new services "
    "yourself. Describe what it would actually take and the tradeoffs involved, the way a "
    "knowledgeable engineer would, rather than pretending to do it or refusing without "
    "explanation. The one exception is Home Assistant automations specifically: if asked to "
    "create or modify one, you CAN actually do it, but only through the two-step "
    "ha_draft_automation -> ha_apply_automation_draft flow. Draft it, show the human the full "
    "config and explain in plain English what it will do, then STOP and wait for their next "
    "message. Only call ha_apply_automation_draft after they've explicitly said yes to that "
    "specific draft - never in the same turn as drafting it, even if the request was clear and "
    "you're confident it's right. If they ask for changes, draft again with a new config; don't "
    "try to patch the old draft. ha_draft_automation itself refuses anything whose actions touch "
    "a domain outside the same safe allowlist ha_call_service uses - locks/covers/alarm panels "
    "can't be authored this way either, tell the human that plainly if it comes up.\n\n"
    "Keep answers concise and direct. Only use the tools provided, and never guess a "
    "container/service/host name that a tool hasn't confirmed exists."
)


def call_model_chat(messages):
    payload = {"model": oa.MODEL, "messages": messages, "tools": CHAT_TOOLS, "stream": False}
    req = urllib.request.Request(
        oa.OLLAMA_URL, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=MODEL_CALL_TIMEOUT) as resp:
        return json.loads(resp.read())


def run_chat(history):
    """history: list of {"role": "user"|"assistant", "content": str}, the
    conversation so far (most recent user message last). Returns
    {"reply": str, "tool_calls": [{"tool", "args", "result_summary"}]}."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    tool_calls_made = []
    for _ in range(MAX_TURNS):
        resp = call_model_chat(messages)
        msg = resp.get("message", {})
        tcalls = msg.get("tool_calls", [])
        messages.append(msg)

        if not tcalls:
            return {"reply": msg.get("content", ""), "tool_calls": tool_calls_made}

        for call in tcalls:
            fn_name = call["function"]["name"]
            args = call["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            impl = CHAT_TOOL_IMPL.get(fn_name)
            if not impl:
                result = {"error": f"unknown or disallowed tool {fn_name!r}"}
            else:
                try:
                    result = impl(**args)
                except Exception as e:
                    oa.log.exception(f"chat tool {fn_name} raised")
                    result = {"error": str(e)}
            tool_calls_made.append({"tool": fn_name, "args": args, "result_summary": str(result)[:300]})
            messages.append({"role": "tool", "content": json.dumps(result, default=str)})

    return {
        "reply": "(hit the turn limit before finishing - try rephrasing or asking something narrower)",
        "tool_calls": tool_calls_made,
    }


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

    def do_POST(self):
        if not self._auth_ok():
            return self._send(401, {"error": "unauthorized"})

        if self.path == "/chat":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                return self._send(400, {"error": "invalid json body"})

            history = body.get("messages")
            if not isinstance(history, list) or not history:
                return self._send(400, {"error": "messages (non-empty list) is required"})
            if len(history) > 40:
                return self._send(400, {"error": "conversation too long - start a new chat"})

            try:
                result = run_chat(history)
            except Exception as e:
                oa.log.exception("chat cycle failed")
                return self._send(502, {"error": f"model call failed: {e}"})
            return self._send(200, result)

        if self.path == "/action":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                return self._send(400, {"error": "invalid json body"})

            tool = body.get("tool")
            if tool not in ACTION_ALLOWLIST:
                return self._send(400, {"error": f"tool must be one of {sorted(ACTION_ALLOWLIST)}"})
            required_args = ACTION_ALLOWLIST[tool]
            args = {k: body.get(k) for k in required_args}
            missing = [k for k in required_args if not args.get(k)]
            if missing:
                return self._send(400, {"error": f"missing required args: {missing}"})

            try:
                result = CHAT_TOOL_IMPL[tool](**args)
            except Exception as e:
                oa.log.exception(f"direct action {tool} raised")
                return self._send(502, {"error": str(e)})
            return self._send(200, {"tool": tool, "args": args, "result": result})

        return self._send(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        pass  # keep the systemd journal quiet - same as ops_agent_api.py


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"ops-agent-chat listening on :{PORT}")
    server.serve_forever()
