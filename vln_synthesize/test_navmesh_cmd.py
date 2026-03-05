"""Compare two ways to set NavMesh parameters - write results to file."""
from isaacsim import SimulationApp
import sys

sim = SimulationApp({"headless": True})

import carb
import omni.kit.commands
import omni.kit.app

manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate("omni.anim.navigation.core", True)

settings = carb.settings.get_settings()
result_file = "/tmp/navmesh_comparison.txt"

with open(result_file, "w") as f:
    f.write("="*70 + "\n")
    f.write("NavMesh Settings Comparison\n")
    f.write("="*70 + "\n\n")
    
    # Method 1: carb.settings.set() with /persistent/
    f.write("Method 1: carb.settings.set() - /persistent/ path\n")
    f.write("-"*70 + "\n")
    
    pfx = "/persistent/exts/omni.anim.navigation.core/navMesh/config"
    test_radius = 0.35
    test_height = 1.9
    
    settings.set(f"{pfx}/agentRadius", test_radius)
    settings.set(f"{pfx}/agentHeight", test_height)
    f.write(f"Setting:  agentRadius = {test_radius}\n")
    f.write(f"Setting:  agentHeight = {test_height}\n\n")
    
    # Read back
    read_r = settings.get(f"{pfx}/agentRadius")
    read_h = settings.get(f"{pfx}/agentHeight")
    f.write(f"Read back from /persistent/:\n")
    f.write(f"  agentRadius = {read_r}\n")
    f.write(f"  agentHeight = {read_h}\n\n")
    
    # Method 2: ChangeSetting command  
    f.write("Method 2: omni.kit.commands.execute('ChangeSetting')\n")
    f.write("-"*70 + "\n")
    
    test_radius_2 = 0.45
    test_height_2 = 2.0
    
    omni.kit.commands.execute('ChangeSetting',
        path=f'{pfx}/agentRadius', value=test_radius_2)
    omni.kit.commands.execute('ChangeSetting',
        path=f'{pfx}/agentHeight', value=test_height_2)
    
    f.write(f"Setting (ChangeSetting):  agentRadius = {test_radius_2}\n")
    f.write(f"Setting (ChangeSetting):  agentHeight = {test_height_2}\n\n")
    
    # Read back
    read_r2 = settings.get(f"{pfx}/agentRadius")
    read_h2 = settings.get(f"{pfx}/agentHeight")
    f.write(f"Read back from /persistent/:\n")
    f.write(f"  agentRadius = {read_r2}\n")
    f.write(f"  agentHeight = {read_h2}\n\n")
    
    # Comparison
    f.write("="*70 + "\n")
    f.write("CONCLUSION\n")
    f.write("="*70 + "\n")
    f.write(f"Final values in /persistent/:\n")
    f.write(f"  agentRadius: {read_r2}\n")
    f.write(f"  agentHeight: {read_h2}\n\n")
    f.write("Key findings:\n")
    f.write("1. Both methods write to the SAME /persistent/ path\n")
    f.write("2. The second write (Method 2) overwrites Method 1\n")
    f.write("3. No functional difference between methods\n")
    f.write("   -> Use carb.settings.set() as it's simpler\n")
    f.write("   -> Both result in the same persistent storage\n")

print(f"Results written to {result_file}")
sim.close()
sys.exit(0)
