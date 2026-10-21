"""The exclusions the README calls hard must hold even when the config says otherwise."""
import subprocess
import sys
import textwrap

import pytest

import ops_agent as o

ROOT = __file__.rsplit("/tests/", 1)[0]


def import_with_config(tmp_path, yaml_text):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(yaml_text))
    code = "import ops_agent as o; print(o.OS_AUTO_HOSTS); print(sorted(o.NEVER_PATCH_HOSTS))"
    env = {"PATH": "/usr/bin:/bin", "OPS_AGENT_HA_TOKEN": "t", "OPS_AGENT_NTFY_TOPIC": "x", "OPS_AGENT_CONFIG_FILE": str(cfg), "PYTHONPATH": ROOT}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    return out.stdout.splitlines()


def test_the_hypervisor_and_the_nas_are_dropped_from_a_hand_edited_config(tmp_path):
    hosts, never = import_with_config(tmp_path, "os_auto_hosts: [workstation, proxmox-host, qnap, npm-lxc, docker-lxc]\n")
    assert hosts == "['workstation', 'npm-lxc', 'docker-lxc']"
    assert never == "['proxmox-host', 'qnap']"


def test_a_config_with_only_forbidden_hosts_patches_nothing(tmp_path):
    hosts, _ = import_with_config(tmp_path, "os_auto_hosts: [proxmox-host, qnap]\n")
    assert hosts == "[]"


@pytest.mark.parametrize("host", ["proxmox-host", "qnap"])
def test_apply_os_updates_refuses_even_if_the_host_is_somehow_in_the_list(host, monkeypatch):
    monkeypatch.setattr(o, "OS_AUTO_HOSTS", [host, "workstation"])       # simulate a bypass of the config filter
    monkeypatch.setattr(o, "run", lambda *a, **k: pytest.fail("a command was run on a never-patch host"))
    result = o.apply_os_updates(host)
    assert result["success"] is False and "never OS-patched" in result["error"]


def test_the_lxcs_in_the_config_are_patchable_as_the_readme_says(monkeypatch):
    calls = []
    monkeypatch.setattr(o, "OS_AUTO_HOSTS", ["npm-lxc"])
    monkeypatch.setattr(o, "run", lambda host, cmd, **k: calls.append(host) or (0, "no", ""))
    o.apply_os_updates("npm-lxc")
    assert calls and set(calls) == {"npm-lxc"}
