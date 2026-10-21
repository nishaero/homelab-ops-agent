"""Tests for the guard rails of the ops agent. No hosts, Home Assistant or model are needed."""
import pytest

import ops_agent as o


def names(tools):
    return [t["function"]["name"] for t in tools]


# ---- what the model can call -------------------------------------------------------------

def test_the_model_has_no_shell_or_arbitrary_command_tool():
    for n in names(o.TOOLS):
        assert not any(bad in n for bad in ("shell", "exec", "bash", "command", "ssh")), n


def test_every_advertised_tool_has_an_implementation():
    assert [n for n in names(o.TOOLS) if n not in o.TOOL_IMPL] == []


def test_adsb_tools_are_hidden_while_adsb_is_disabled():
    assert not o.ADSB_ENABLED
    assert [n for n in names(o.TOOLS) if "adsb" in n] == []


# ---- Home Assistant control is limited to safe domains ---------------------------------

@pytest.mark.parametrize("domain", ["lock", "cover", "alarm_control_panel", "shell_command", "rest_command",
                                    "homeassistant", "hassio", "automation", "script_unsafe"])
def test_dangerous_domains_are_refused_before_any_request(domain, monkeypatch):
    monkeypatch.setattr(o, "_ha_request", lambda *a, **k: pytest.fail("a request was sent"))
    assert "not in the allowed set" in o.ha_call_service(domain, "any", "x.y")["error"]


def test_a_safe_domain_goes_through(monkeypatch):
    sent = []
    monkeypatch.setattr(o, "_ha_request", lambda method, path, **k: sent.append((method, path, k["body"])) or {})
    assert o.ha_call_service("light", "turn_on", "light.kitchen")["success"]
    assert sent[0][1] == "/api/services/light/turn_on" and sent[0][2]["entity_id"] == "light.kitchen"


def test_the_safe_domain_list_has_no_lock_cover_or_alarm():
    assert not {"lock", "cover", "alarm_control_panel"} & set(o.HA_SAFE_SERVICE_DOMAINS)


# ---- some hosts are check only ---------------------------------------------------------

@pytest.mark.parametrize("host", ["proxmox-host", "qnap", "docker-lxc", "unknown"])
def test_os_updates_are_refused_for_hosts_that_are_not_auto_apply(host, monkeypatch):
    monkeypatch.setattr(o, "run", lambda *a, **k: pytest.fail("a command was run"))
    assert o.apply_os_updates(host)["success"] is False


# ---- context limits --------------------------------------------------------------------

def test_a_tool_result_is_clipped():
    out = o._clip_tool_result({"text": "x" * 20000})
    assert len(out) < o.TOOL_RESULT_MAX_CHARS + 100 and "[clipped" in out
    assert "clipped" not in o._clip_tool_result({"ok": True})


def test_pruning_trims_the_oldest_tool_output_and_keeps_the_request():
    big = "y" * 30000
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "the request"},
                {"role": "tool", "content": big}, {"role": "tool", "content": big}, {"role": "tool", "content": big}]
    o._prune_messages(messages)
    assert messages[0]["content"] == "rules" and messages[1]["content"] == "the request"
    assert messages[2]["content"].startswith("[older tool output trimmed")
    assert sum(len(m["content"]) for m in messages) <= o.CONVERSATION_MAX_CHARS + 100


def test_pruning_leaves_a_small_conversation_alone():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "tool", "content": "z" * 500}]
    o._prune_messages(messages)
    assert messages[2]["content"] == "z" * 500


# ---- notification dedupe ---------------------------------------------------------------

@pytest.fixture
def notifier(monkeypatch):
    state, sent = {}, []
    monkeypatch.setattr(o, "load_state", lambda: state)
    monkeypatch.setattr(o, "save_state", lambda s: None)
    monkeypatch.setattr(o, "_mem_log", lambda *a, **k: None)
    monkeypatch.setattr(o.urllib.request, "urlopen", lambda req, timeout=0: sent.append(req))
    return sent


def test_a_repeated_topic_is_sent_once_per_cooldown(notifier):
    assert o.notify_human("t", "m", topic="update-fail:web")["sent"]
    again = o.notify_human("t", "m", topic="update-fail:web")
    assert again["suppressed"] and len(notifier) == 1


def test_different_topics_and_untagged_messages_always_send(notifier):
    o.notify_human("t", "m", topic="a")
    o.notify_human("t", "m", topic="b")
    o.notify_human("t", "m")
    o.notify_human("t", "m")
    assert len(notifier) == 4


def test_urgent_uses_the_urgent_priority(notifier):
    o.notify_human("t", "m", urgent=True)
    assert notifier[0].headers["Priority"] == "urgent"
