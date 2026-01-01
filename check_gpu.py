import time
import sys

print("=== 1. CHECKING PYTHON & HARDWARE ===")
try:
    import torch
    print(f"✅ Python Version: {sys.version.split()[0]}")
    print(f"✅ PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("❌ WARNING: GPU not detected. You are running on CPU.")
        print("   Solution: Re-install PyTorch with the specific CUDA command provided earlier.")
except ImportError:
    print("❌ ERROR: PyTorch is not installed.")

print("\n=== 2. CHECKING RL LIBRARIES ===")
try:
    import pettingzoo
    from pettingzoo.mpe import simple_tag_v3
    print(f"✅ PettingZoo Version: {pettingzoo.__version__}")
except ImportError as e:
    print(f"❌ ERROR: PettingZoo not found. ({e})")

try:
    import stable_baselines3
    import supersuit
    print(f"✅ Stable-Baselines3 & SuperSuit detected.")
except ImportError as e:
    print(f"❌ ERROR: SB3 or SuperSuit not found. ({e})")

print("\n=== 3. RUNNING VISUAL SIMULATION (5 Seconds) ===")
print("A window should pop up showing 3 Red Agents chasing 1 Green Agent...")
time.sleep(2) # Give user time to read

try:
    # Initialize the specific 'Predator-Prey' environment
    # render_mode='human' is what pops up the Windows GUI
    env = simple_tag_v3.env(render_mode='human', num_good=1, num_adversaries=3, num_obstacles=0)
    env.reset()

    # Run for 100 steps (approx 5 seconds)
    step_count = 0
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            action = None
        else:
            # Pick a random action for the agent
            action = env.action_space(agent).sample()

        env.step(action)
        
        # Slow down slightly so you can see the movement
        time.sleep(0.05)
        
        step_count += 1
        if step_count > 100:
            break

    env.close()
    print("\n✅ SUCCESS: Simulation closed successfully.")
    print("You are ready to start Month 1, Week 2!")

except Exception as e:
    print(f"\n❌ ERROR during simulation: {e}")
    print("Common fix: Ensure you are not running this inside a Docker container or headless server.")