import os
from pettingzoo.mpe import simple_tag_v3
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

def train():
    # 1. SETUP
    env = simple_tag_v3.parallel_env(
        num_good=1, 
        num_adversaries=3, 
        num_obstacles=2, 
        max_cycles=25, 
        render_mode=None
    )

    # 2. WRAPPERS (THE FIX IS HERE)
    # ----------------------------------------------------------------
    # FIX: Pad observations so Wolf and Sheep have same input shape
    env = ss.pad_observations_v0(env)
    
    # FIX: Pad actions (Just in case, good practice in MARL)
    env = ss.pad_action_space_v0(env)
    # ----------------------------------------------------------------

    # Now continue with the rest of the wrappers
    env = ss.frame_stack_v1(env, 3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # Use num_cpus=0 for Windows compatibility
    env = ss.concat_vec_envs_v1(env, num_vec_envs=8, num_cpus=0, base_class='stable_baselines3')
    env = VecMonitor(env)

    # 3. THE BRAIN
    print("Setting up PPO Model on RTX 3050...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./swarm_tensorboard/",
        learning_rate=3e-4,
        batch_size=2048,
        n_steps=2048,
        ent_coef=0.01,
        device="cuda"
    )

    # 4. TRAIN
    print("---------------------------------------")
    print("STARTING TRAINING... (Press Ctrl+C to stop early)")
    print("---------------------------------------")
    
    try:
        model.learn(total_timesteps=2_000_000)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current progress...")

    # 5. SAVE
    model_path = "swarm_model_v1"
    model.save(model_path)
    print(f"Model saved to: {model_path}.zip")

if __name__ == "__main__":
    train()