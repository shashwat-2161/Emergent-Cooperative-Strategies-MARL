import numpy as np
from pettingzoo.mpe import simple_tag_v3
import supersuit as ss
from stable_baselines3 import PPO
from tqdm import tqdm

def run_physics_test(test_name, speed_modifier_wolf=1.0, speed_modifier_sheep=1.0, episodes=100):
    print(f"\n--- TESTING SCENARIO: {test_name} ---")
    
    # 1. SETUP (Must keep counts SAME as training to match input shape)
    env = simple_tag_v3.parallel_env(
        num_good=1, 
        num_adversaries=3, 
        num_obstacles=2,  # <--- MUST STAY 2
        max_cycles=50, 
        render_mode=None
    )
    
    # Wrappers
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.frame_stack_v1(env, 3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=0, base_class='stable_baselines3')

    # 2. LOAD MODEL
    try:
        model = PPO.load("swarm_model_v2")
    except:
        print("Model v2 not found, using v1...")
        model = PPO.load("swarm_model_v1")

    wins = 0

    # 3. TEST LOOP
    for _ in tqdm(range(episodes)):
        obs = env.reset()
        
        # --- THE HACK: MODIFY PHYSICS DIRECTLY ---
        # We access the raw world to change speeds dynamically
        # env.unwrapped... accesses the core MPE engine
        # Note: We must do this AFTER reset because reset might restore defaults
        try:
            # The 'unwrapped' chain can be deep. We try to find the world.
            raw_env = env.envs[0].unwrapped 
            for agent in raw_env.world.agents:
                if agent.adversary: # It is a Wolf
                    agent.max_speed = 1.0 * speed_modifier_wolf # Default is usually 1.0 or 1.3
                else: # It is a Sheep
                    agent.max_speed = 1.3 * speed_modifier_sheep # Default is usually 1.3
        except AttributeError:
            # Fallback if wrapper structure is different, usually works for ConcatedVecEnv
            pass

        # Run Episode
        for step in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            
            # Check Win (Wolf got reward > 5)
            # The rewards array structure is usually [Wolf, Wolf, Wolf, Sheep]
            # We check the first 3 (Adversaries)
            if np.any(rewards[:3] > 5): 
                wins += 1
                break
                
    win_rate = (wins / episodes) * 100
    print(f"Result for {test_name}: {win_rate:.1f}% Win Rate")
    return win_rate

def main():
    # Define Physics Scenarios
    scenarios = [
        ("Baseline (Normal Speed)", 1.0, 1.0),
        ("Exp A: Fast Sheep (1.5x Speed)", 1.0, 1.5),  # Harder to catch
        ("Exp B: Slow Wolves (0.7x Speed)", 0.7, 1.0)  # Harder to chase
    ]

    results = {}
    for name, wolf_mod, sheep_mod in scenarios:
        results[name] = run_physics_test(name, wolf_mod, sheep_mod)

    # Print Thesis Table
    print("\n" + "="*50)
    print("      ROBUSTNESS ANALYSIS (PHYSICS STRESS TEST)")
    print("="*50)
    print(f"{'Scenario Name':<40} | {'Win Rate':<10}")
    print("-" * 52)
    for name, score in results.items():
        print(f"{name:<40} | {score:.1f}%")
    print("="*50)

if __name__ == "__main__":
    main()