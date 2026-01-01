import time
from pettingzoo.mpe import simple_tag_v3
import supersuit as ss
from stable_baselines3 import PPO

def watch():
    # 1. SETUP: Create the environment with render_mode='human'
    env = simple_tag_v3.parallel_env(
        num_good=1, 
        num_adversaries=3, 
        num_obstacles=2, 
        max_cycles=100, # Run longer so you can enjoy watching
        render_mode='human'
    )

    # 2. WRAPPERS: Apply EXACTLY the same wrappers as train.py
    #    This ensures the "Prey" (size 42) gets padded to "Predator" size (48)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.frame_stack_v1(env, 3)

    # 3. VECTORIZE: This is the magic step missing before.
    #    It converts the PettingZoo dictionary into the SB3 Vector format.
    #    This handles the batch dimension automatically.
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    #    num_vec_envs=1: We only want to watch ONE game window
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=0, base_class='stable_baselines3')

    # 4. LOAD MODEL
    try:
        model = PPO.load("swarm_model_v2")
        print("Model loaded successfully. Starting simulation...")
    except FileNotFoundError:
        print("Error: 'swarm_model_v2.zip' not found.")
        return

    # 5. PLAY LOOP (Standard Gym API)
    #    Because we vectorized the env, we don't need the complex agent loop anymore.
    #    We just treat it like a single standard environment.
    
    obs = env.reset()
    
    try:
        while True:
            # Predict actions for ALL agents at once (Batch Prediction)
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, rewards, dones, infos = env.step(action)
            
            # Render happens automatically because we set 'human' in step 1
            time.sleep(0.05) 
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    watch()