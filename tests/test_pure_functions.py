"""Pure helpers with no host or network: image references, tag comparison, container parity."""
import pytest

import ops_agent as o


@pytest.mark.parametrize("ref,expected", [
    ("nginx", (None, "nginx", None, None)),
    ("nginx:1.25", (None, "nginx", "1.25", None)),
    ("user/app:latest", (None, "user/app", "latest", None)),
    ("ghcr.io/immich-app/immich-server:v3.3.3", ("ghcr.io", "immich-app/immich-server", "v3.3.3", None)),
    ("localhost:5000/app", ("localhost:5000", "app", None, None)),
    ("registry.local:5000/team/app:2.0", ("registry.local:5000", "team/app", "2.0", None)),
    ("repo/app@sha256:abc", (None, "repo/app", None, "sha256:abc")),
])
def test_parse_image_ref(ref, expected):
    assert o._parse_image_ref(ref) == expected


def test_find_newer_tag_picks_the_highest_of_the_same_shape():
    assert o._find_newer_tag("v3.3.3", ["v3.3.4", "v3.4.0", "v4.0.0", "v3.3.3", "latest"]) == "v4.0.0"


def test_find_newer_tag_keeps_the_variant_suffix():
    assert o._find_newer_tag("v3.3.3-rocm", ["v3.3.4", "v3.3.4-rocm", "v3.4.0-rocm"]) == "v3.4.0-rocm"


def test_find_newer_tag_keeps_the_version_granularity():
    """A bare major pin ('32') must only be compared with other bare majors, never with '34.0.4'."""
    assert o._find_newer_tag("32", ["33", "34.0.4", "31", "32"]) == "33"


@pytest.mark.parametrize("current,tags", [("v3.3.3", ["v3.3.3", "v3.3.2"]), ("latest", ["1.0"]), ("3.3.3", ["v3.4.0"]), ("1.2.3", [])])
def test_find_newer_tag_returns_none_when_nothing_qualifies(current, tags):
    assert o._find_newer_tag(current, tags) is None


def inspect(**over):
    base = {"Image": "sha256:old", "Config": {"Entrypoint": ["/run"], "Cmd": ["--serve"], "User": ""},
            "HostConfig": {"NetworkMode": "bridge", "PortBindings": {"80/tcp": [{}]}, "Privileged": False, "CapAdd": None,
                           "Devices": [], "DeviceRequests": [], "ExtraHosts": []},
            "NetworkSettings": {"Networks": {"app_default": {}}},
            "Mounts": [{"Type": "bind", "Destination": "/data", "Source": "/srv/data"}]}
    for k, v in over.items():
        base[k] = v
    return base


def test_parity_of_identical_containers_has_no_problems():
    assert o._parity_problems(inspect(), inspect()) == []


def test_parity_notices_a_lost_command_when_the_image_is_unchanged():
    new = inspect(Config={"Entrypoint": ["/run"], "Cmd": [], "User": ""})
    assert any(p.startswith("command") for p in o._parity_problems(inspect(), new))


def test_parity_ignores_command_changes_when_the_image_changed():
    new = inspect(Image="sha256:new", Config={"Entrypoint": ["/run"], "Cmd": ["--other"], "User": ""})
    assert o._parity_problems(inspect(), new) == []


@pytest.mark.parametrize("field,new,needle", [
    ("HostConfig", {"NetworkMode": "host", "PortBindings": {"80/tcp": [{}]}, "Privileged": False, "CapAdd": None, "Devices": [], "DeviceRequests": [], "ExtraHosts": []}, "network mode"),
    ("NetworkSettings", {"Networks": {"other": {}}}, "networks"),
    ("Mounts", [{"Type": "bind", "Destination": "/data", "Source": "/elsewhere"}], "mounts"),
    ("Mounts", [], "mounts"),
])
def test_parity_notices_lost_network_and_mounts(field, new, needle):
    assert any(needle in p for p in o._parity_problems(inspect(), inspect(**{field: new})))


def test_parity_notices_privilege_devices_gpu_and_user():
    hc = inspect()["HostConfig"]
    for change, needle in (({"Privileged": True}, "privileged"), ({"Devices": [{"PathOnHost": "/dev/dri"}]}, "devices"),
                           ({"DeviceRequests": [{"Driver": "nvidia"}]}, "GPU"), ({"CapAdd": ["NET_ADMIN"]}, "cap_add"),
                           ({"PortBindings": {}}, "ports")):
        assert any(needle in p for p in o._parity_problems(inspect(), inspect(HostConfig={**hc, **change}))), needle
    assert any("user" in p for p in o._parity_problems(inspect(), inspect(Config={"Entrypoint": ["/run"], "Cmd": ["--serve"], "User": "1000"})))
