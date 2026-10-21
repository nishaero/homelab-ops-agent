import os
import sys
from pathlib import Path

os.environ.setdefault("OPS_AGENT_HA_TOKEN", "test-token")
os.environ.setdefault("OPS_AGENT_NTFY_TOPIC", "test-topic")
os.environ.setdefault("OPS_AGENT_CONFIG_FILE", "/nonexistent/config.yaml")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
