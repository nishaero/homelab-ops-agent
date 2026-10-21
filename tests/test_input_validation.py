"""Arguments from the model are untrusted. A prompt injected into a changelog or log must not be able to
escape a shell command or a URL through a container name."""
import shlex

import pytest

import ops_agent as o

BAD_NAMES = ["x'; touch /tmp/pwned; '", "a b", "../etc/passwd", "web\nrm -rf /", "$(id)", "a;b", "`id`", "a|b", "a&b",
             "", "-leading-dash", "x" * 129, "naïve", "web?force=1", "web/restart", None, 42, ["web"]]


@pytest.mark.parametrize("name", BAD_NAMES)
def test_bad_container_names_are_rejected(name):
    with pytest.raises(ValueError):
        o.validate_tool_args({"container": name})


@pytest.mark.parametrize("name", ["web", "immich_server", "nextcloud-app", "a.b-c_d", "web-1"])
def test_normal_container_names_are_accepted(name):
    o.validate_tool_args({"container": name})


@pytest.mark.parametrize("key,value", [("candidate_image_ref", "x'; id; '"), ("candidate_image_ref", "a b"), ("candidate_tag", "1.0; id"),
                                         ("domain", "light; id"), ("service", "turn_on/../x"), ("entity_id", "light.kitchen; id"),
                                         ("entity_id", "all"), ("entity_id", "light.a,light.b")])
def test_other_free_text_arguments_are_checked(key, value):
    with pytest.raises(ValueError):
        o.validate_tool_args({key: value})


@pytest.mark.parametrize("key,value", [("candidate_image_ref", "ghcr.io/org/app:1.2.3"), ("candidate_image_ref", "nginx@sha256:abcdef"),
                                         ("candidate_tag", "v1.2.3-rocm"), ("domain", "light"), ("service", "turn_on"),
                                         ("entity_id", "update.home_assistant_core_update")])
def test_normal_values_pass(key, value):
    o.validate_tool_args({key: value})


def test_null_for_an_optional_number_means_the_default():
    args = {"lines": None}
    o.validate_tool_args(args)
    assert args == {"lines": None}


def test_numbers_are_coerced_and_junk_is_rejected():
    args = {"lines": "50", "hours": 24}
    o.validate_tool_args(args)
    assert args == {"lines": 50, "hours": 24}
    for bad in ({"lines": "all"}, {"hours": True}, {"vmid": "1; id"}, {"days": [1]}):
        with pytest.raises(ValueError):
            o.validate_tool_args(bad)


def test_the_qnap_command_keeps_a_hostile_path_inside_one_argument(monkeypatch):
    seen = []
    monkeypatch.setattr(o, "run", lambda host, cmd, **k: seen.append(cmd) or (0, "", ""))
    hostile = "/containers/x'; touch /tmp/pwned; '/json"
    o.qnap_docker_api("GET", hostile)
    words = shlex.split(seen[0])
    assert words[-1] == "http://localhost" + hostile              # one argument, nothing was executed
    assert "touch" not in words


def test_the_qnap_request_body_is_one_argument_too(monkeypatch):
    seen = []
    monkeypatch.setattr(o, "run", lambda host, cmd, **k: seen.append(cmd) or (0, "", ""))
    o.qnap_docker_api("POST", "/containers/create?name=web", data={"Cmd": ["sh", "-c", "echo 'hi'; id"]})
    words = shlex.split(seen[0])
    assert words[words.index("-d") + 1] == '{"Cmd": ["sh", "-c", "echo \'hi\'; id"]}'
    assert words[-1].startswith("http://localhost")


def test_qnap_api_refuses_an_odd_method(monkeypatch):
    monkeypatch.setattr(o, "run", lambda *a, **k: pytest.fail("ran"))
    with pytest.raises(ValueError):
        o.qnap_docker_api("GET; id", "/x")


def test_get_logs_on_the_qnap_cannot_be_used_to_run_a_command(monkeypatch):
    seen = []
    monkeypatch.setattr(o, "run", lambda host, cmd, **k: seen.append(cmd) or (0, "", ""))
    o.get_logs("qnap", "x'; touch /tmp/pwned; '")           # even if validation were bypassed, quoting holds
    assert "touch" not in shlex.split(seen[0])


# ---- Home Assistant ----------------------------------------------------------------------

def test_scripts_and_scenes_need_to_be_named_in_the_config(monkeypatch):
    monkeypatch.setattr(o, "_ha_request", lambda *a, **k: pytest.fail("a request was sent"))
    for domain, ent in (("script", "script.unlock_front_door"), ("scene", "scene.arrive_home")):
        assert "ha_allowed_entities" in o.ha_call_service(domain, "turn_on", ent)["error"]


def test_a_named_script_is_allowed(monkeypatch):
    monkeypatch.setattr(o, "HA_ALLOWED_ENTITIES", {"script.goodnight_lights"})
    monkeypatch.setattr(o, "_ha_request", lambda *a, **k: {})
    assert o.ha_call_service("script", "turn_on", "script.goodnight_lights")["success"]
    assert "error" in o.ha_call_service("script", "turn_on", "script.other")


@pytest.mark.parametrize("domain,entity", [("light", "lock.front_door"), ("switch", "cover.garage"), ("light", "all"),
                                           ("light", "light.a,lock.b"), ("light", "light.kitchen; id")])
def test_the_entity_must_be_one_entity_of_the_same_domain(domain, entity, monkeypatch):
    monkeypatch.setattr(o, "_ha_request", lambda *a, **k: pytest.fail("a request was sent"))
    assert "error" in o.ha_call_service(domain, "turn_on", entity)


@pytest.mark.parametrize("key", ["entity_id", "device_id", "area_id", "floor_id", "label_id"])
def test_data_cannot_retarget_the_call(key, monkeypatch):
    monkeypatch.setattr(o, "_ha_request", lambda *a, **k: pytest.fail("a request was sent"))
    assert "target" in o.ha_call_service("light", "turn_on", "light.kitchen", data={key: "lock.front_door"})["error"]


def test_notify_is_in_the_safe_list_as_documented():
    assert "notify" in o.HA_SAFE_SERVICE_DOMAINS and not {"script", "scene"} & set(o.HA_SAFE_SERVICE_DOMAINS)
