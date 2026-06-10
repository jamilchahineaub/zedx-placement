# isaac/probe_node_types.py
# Run once with Isaac Python to discover the ZED Helper node type string.
# Usage: /home/jimmy/isaacsim/python.sh isaac/probe_node_types.py

import sys
import os

# Load machine config to get ext path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "config", "machine.laptop.yaml")
with open(cfg_path) as f:
    machine = yaml.safe_load(f)

ext_path = machine["zed_ext_path"]
print(f"Using ext path: {ext_path}")

# Boot minimal Isaac
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 256, "height": 256})

import omni.kit.app
import omni.graph.core as og

# Enable the ZED extension
manager = omni.kit.app.get_app().get_extension_manager()

# Add the ext folder to search paths
manager.add_path(ext_path)

# Find and enable any ZED extension
exts = manager.get_extensions()
zed_exts = [e for e in exts if "zed" in e["id"].lower()]

print(f"\nFound {len(zed_exts)} ZED extension(s):")
for e in zed_exts:
    print(f"  id: {e['id']}  enabled: {e.get('enabled', False)}")
    if not e.get("enabled", False):
        print(f"  Enabling {e['id']}...")
        manager.set_extension_enabled_immediate(e["id"], True)

# Wait a couple frames for nodes to register
for _ in range(5):
    app.update()

# List all registered node types containing 'zed'
reg = og.GraphRegistry()
all_types = reg.get_registered_node_types()
zed_types = [t for t in all_types if "zed" in t.lower()]

print(f"\nRegistered OmniGraph node types containing 'zed':")
if zed_types:
    for t in sorted(zed_types):
        print(f"  {t}")
else:
    print("  NONE FOUND — extension may not have loaded correctly")
    print("  Check that zed_ext_path is correct in machine.laptop.yaml")

print(f"\nCopy the ZED Camera Helper node type into CLAUDE.md")
print(f"It will look something like: sl.zed.ZEDCamera or stereolabs.zed.ZEDCameraHelper")

app.close()
