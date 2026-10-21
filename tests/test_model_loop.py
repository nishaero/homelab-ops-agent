"""The headline guard rails of a cycle: repeat cap, turn limit, time budget, model failure, validation."""
import json
import time

import pytest

import ops_agent as o


def call(tool, **args):
    return {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": tool, "arguments": args}}]}}


def last_tool_output(msgs):
    return [m for m in msgs if m["role"] == "tool"][-1]["content"]


def final(text="all done"):
    return {"message": {"role": "assistant", "content": text}}


@pytest.fixture
def loop(monkeypatch):
    """Run the model loop with a scripted model and a recording 'probe' tool."""
    ran, notes = [], []
    monkeypatch.setattr(o, "TOOL_IMPL", {**o.TOOL_IMPL, "probe": lambda **kw: ran.append(kw) or {"ok": True, "n": len(ran)}})
    monkeypatch.setattr(o, "notify_human", lambda *a, **k: notes.append((a, k)))
    monkeypatch.setattr(o, "_mem_log", lambda *a, **k: None)

    def go(script, messages=None):
        it = iter(script)
        seen = []

        def fake(msgs, **k):
            seen.append(len(msgs))
            nxt = next(it)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt(msgs) if callable(nxt) else nxt
        monkeypatch.setattr(o, "call_model", fake)
        msgs = messages or [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        actions, aborted = o.run_model_loop(msgs)
        return actions, aborted, msgs, seen
    go.ran, go.notes = ran, notes
    return go


def test_a_final_answer_ends_the_cycle(loop):
    actions, aborted, _, seen = loop([final()])
    assert actions == [] and aborted is False and seen == [2]


def test_an_identical_call_is_refused_after_two_tries(loop):
    actions, aborted, msgs, _ = loop([call("probe", x=1), call("probe", x=1), call("probe", x=1), final()])
    assert len(loop.ran) == 2                                    # the third identical call never ran
    assert "identical call already made twice" in last_tool_output(msgs)
    assert len(actions) == 3 and aborted is False


def test_different_arguments_are_not_treated_as_a_repeat(loop):
    loop([call("probe", x=1), call("probe", x=2), call("probe", x=3), final()])
    assert len(loop.ran) == 3


def test_the_turn_limit_stops_a_model_that_never_finishes(loop, monkeypatch):
    monkeypatch.setattr(o, "MAX_TURNS", 5)
    actions, aborted, _, seen = loop([call("probe", x=i) for i in range(50)])
    assert len(actions) == 5 and len(seen) == 5


def test_the_time_budget_stops_the_cycle_before_asking_the_model(loop, monkeypatch):
    monkeypatch.setattr(o, "CYCLE_BUDGET_SECONDS", -1)
    actions, aborted, _, seen = loop([final()])
    assert aborted is True and seen == [] and actions == []


def test_the_time_budget_stops_a_cycle_that_runs_long(loop, monkeypatch):
    monkeypatch.setattr(o, "CYCLE_BUDGET_SECONDS", 0.05)
    monkeypatch.setitem(o.TOOL_IMPL, "probe", lambda **kw: time.sleep(0.1) or {"ok": True})
    actions, aborted, _, seen = loop([call("probe", x=1), call("probe", x=2), final()])
    assert aborted is True and seen == [2] and len(actions) == 1


def test_a_failing_model_call_ends_the_cycle_and_tells_a_human(loop):
    actions, aborted, _, _ = loop([RuntimeError("ollama is down")])
    assert aborted is True and len(loop.notes) == 1 and "ollama is down" in loop.notes[0][0][1]


def test_an_unknown_tool_is_an_error_not_a_crash(loop):
    actions, aborted, msgs, _ = loop([call("format_disk", target="/"), final()])
    assert "unknown tool" in last_tool_output(msgs) and aborted is False


def test_a_tool_that_raises_is_reported_to_the_model(loop, monkeypatch):
    def boom(**kw):
        raise RuntimeError("disk on fire")
    monkeypatch.setitem(o.TOOL_IMPL, "probe", boom)
    _, _, msgs, _ = loop([call("probe"), final()])
    assert "disk on fire" in last_tool_output(msgs)


def test_arguments_sent_as_a_json_string_are_understood(loop):
    msg = {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "probe", "arguments": json.dumps({"x": 7})}}]}}
    loop([msg, final()])
    assert loop.ran == [{"x": 7}]


def test_a_hostile_container_name_never_reaches_a_tool(loop, monkeypatch):
    seen = []
    monkeypatch.setitem(o.TOOL_IMPL, "get_logs", lambda **kw: seen.append(kw) or {})
    _, _, msgs, _ = loop([call("get_logs", host="qnap", container="x'; touch /tmp/pwned; '"), final()])
    assert seen == [] and "rejected" in last_tool_output(msgs)


def test_large_tool_output_is_clipped_in_the_conversation(loop, monkeypatch):
    monkeypatch.setitem(o.TOOL_IMPL, "probe", lambda **kw: {"text": "x" * 50000})
    _, _, msgs, _ = loop([call("probe"), final()])
    out = last_tool_output(msgs)
    assert len(out) < o.TOOL_RESULT_MAX_CHARS + 100 and "[clipped" in out
