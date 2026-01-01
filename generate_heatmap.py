import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pettingzoo.mpe import simple_tag_v3
import supersuit as ss
from stable_baselines3 import PPO
from tqdm import tqdm
import os

# --- Configuration ---
MODEL_PATH = "swarm_model_v2" # Use your best model
EPISODES_TO_WATCH = 50        # 50 games
ARENA_BOUNDS = 1.5            # Arena size

def generate_heatmap():
    print(f"--- Starting Data Collection ({EPISODES_TO_WATCH} Episodes) ---")
    
    # 1. SETUP (Standard Parallel Env - No Vectorization Wrappers)
    env = simple_tag_v3.parallel_env(
        num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=50, render_mode=None
    )
    # Apply standard processing wrappers (Must match training)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.frame_stack_v1(env, 3)
    
    # 2. LOAD MODEL
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"Error: {MODEL_PATH}.zip not found. Please train the model first.")
        return

    model = PPO.load(MODEL_PATH)

    # 3. DATA STORAGE
    predator_positions = []
    prey_positions = []

    # 4. DATA COLLECTION LOOP
    for _ in tqdm(range(EPISODES_TO_WATCH), desc="Collecting data"):
        
        # --- THE FIX IS HERE ---
        # env.reset() returns (observations, infos). We only need observations.
        obs_dict, _ = env.reset() 
        
        # PettingZoo Parallel API Loop
        while env.agents:
            actions = {}
            
            for agent in env.agents:
                # Grab observation for this specific agent
                agent_obs = obs_dict[agent]
                
                # Reshape for the model (Batch of 1)
                agent_obs_batch = agent_obs.reshape(1, -1)
                
                # Predict
                action, _ = model.predict(agent_obs_batch, deterministic=True)
                actions[agent] = action[0]

            # Step the environment
            obs_dict, rewards, terminations, truncations, infos = env.step(actions)
            
            # --- EXTRACT POSITIONS ---
            raw_world = env.unwrapped.world
            
            for agent in raw_world.agents:
                pos = agent.state.p_pos
                if agent.adversary:
                    predator_positions.append(pos)
                else:
                    prey_positions.append(pos)
            
    print("Generating Heatmap plot...")

    # 5. CONVERT TO NUMPY
    pred_pos_np = np.array(predator_positions)
    prey_pos_np = np.array(prey_positions)

    # 6. VISUALIZATION
    sns.set_theme(style="white", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    fig.suptitle(f'Spatial Occupancy Heatmap (N={EPISODES_TO_WATCH})', fontsize=18, y=1.02)

    extent = [-ARENA_BOUNDS, ARENA_BOUNDS, -ARENA_BOUNDS, ARENA_BOUNDS]
    bins = 40 

    # Plot Predators
    h1 = axes[0].hist2d(pred_pos_np[:, 0], pred_pos_np[:, 1], bins=bins, range=[extent[:2], extent[2:]], cmap="Reds", density=True)
    axes[0].set_title("Predator (Wolf) Density", fontsize=16, color="darkred")
    axes[0].set_aspect('equal')
    fig.colorbar(h1[3], ax=axes[0], label="Normalized frequency")

    # Plot Prey
    h2 = axes[1].hist2d(prey_pos_np[:, 0], prey_pos_np[:, 1], bins=bins, range=[extent[:2], extent[2:]], cmap="Greens", density=True)
    axes[1].set_title("Prey (Sheep) Density", fontsize=16, color="darkgreen")
    axes[1].set_aspect('equal')
    fig.colorbar(h2[3], ax=axes[1], label="Normalized frequency")

    fig.text(0.5, 0.04, 'X Coordinate', ha='center', fontsize=14)
    fig.text(0.08, 0.5, 'Y Coordinate', va='center', rotation='vertical', fontsize=14)

    # 7. SAVE
    output_filename = "thesis_swarm_heatmap.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {output_filename}")
    plt.show()

if __name__ == "__main__":
    generate_heatmap()