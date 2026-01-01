import numpy as np
from pettingzoo.mpe import simple_tag_v3
import supersuit as ss
from stable_baselines3 import PPO
from tqdm import tqdm # Progress bar

def evaluate(episodes=100):
    print(f"--- STARTING EVALUATION ({episodes} Episodes) ---")
    
    # 1. SETUP (Must match training config exactly)
    env = simple_tag_v3.parallel_env(
        num_good=1, 
        num_adversaries=3, 
        num_obstacles=2, 
        max_cycles=50, # If they don't catch by 50 steps, it's a "Draw"
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
        print("Model v2 not found, trying v1...")
        model = PPO.load("swarm_model_v1")

    # 3. METRICS TO TRACK
    wins = 0
    draws = 0
    total_steps = []
    total_rewards = []

    # 4. TESTING LOOP
    for i in tqdm(range(episodes)):
        obs = env.reset()
        episode_reward = 0
        steps = 0
        caught = False
        
        for step in range(50): # Max cycles
            # Predict
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            
            # Aggregate reward (just to see performance)
            episode_reward += rewards[0]
            steps += 1
            
            # CHECK WIN CONDITION
            # In simple_tag, if a Wolf (Adversary) gets a collision reward (+10), they caught the sheep.
            # The rewards array is [Wolf1, Wolf2, Wolf3, Sheep] (order varies, but usually adversaries first)
            # A reward > 5 usually means a catch occurred.
            if np.any(rewards > 5): 
                caught = True
                break
        
        total_steps.append(steps)
        total_rewards.append(episode_reward)
        
        if caught:
            wins += 1
        else:
            draws += 1

    # 5. PRINT RESULTS FOR THESIS
    win_rate = (wins / episodes) * 100
    avg_steps = np.mean(total_steps)
    
    print("\n" + "="*40)
    print("       THESIS RESULTS TABLE       ")
    print("="*40)
    print(f"Model Tested:      swarm_model_v2")
    print(f"Total Episodes:    {episodes}")
    print(f"Win Rate (Catch):  {win_rate:.1f}%")
    print(f"Avg Time to Catch: {avg_steps:.1f} steps")
    print(f"Avg Episode Score: {np.mean(total_rewards):.1f}")
    print("="*40)
    
    if win_rate > 80:
        print("✅ CONCLUSION: The Swarm Strategy is HIGHLY EFFECTIVE.")
    elif win_rate > 50:
        print("⚠️ CONCLUSION: The Swarm is MODERATELY EFFECTIVE.")
    else:
        print("❌ CONCLUSION: The Swarm fails to corner the target.")

if __name__ == "__main__":
    evaluate()