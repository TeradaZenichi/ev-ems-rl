import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from opt.env import EnergyEnv
from opt.DQN.model import DQN, LSTMDQN

def test_model(START_IDX, END_IDX):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load parameters from parameters.json
    with open("data/parameters.json", "r") as f:
        params = json.load(f)
    
    # Select testing parameters based on MODEL_TYPE
    model_type = params.get("MODEL_TYPE", "DQN")
    if model_type == "DQN":
        test_params = params["TRAIN"]["DQN"]
    elif model_type == "LSTMDQN":
        test_params = params["TRAIN"]["LSTMDQN"]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    discrete_action_size = test_params["discrete_action_size"]
    data_dir = "data"
    
    # Calculate episode length based on provided start and end indices
    episode_length = END_IDX - START_IDX

    # Create the environment using the specified indices
    env = EnergyEnv(data_dir=data_dir, start_idx=START_IDX, episode_length=episode_length)
    state_dim = env.observation_space.shape[0]
    
    # Define discrete action space based on environment's action bounds
    action_low = env.action_space.low[0]
    action_high = env.action_space.high[0]
    discrete_actions = np.linspace(action_low, action_high, discrete_action_size)
    action_dim = len(discrete_actions)
    
    # Instantiate the model based on MODEL_TYPE
    if model_type == "LSTMDQN":
        model = LSTMDQN(state_dim, action_dim).to(device)
    else:
        model = DQN(state_dim, action_dim).to(device)
    
    model_save_name = test_params.get("model_save_name", "dqn_energy.pth")
    model_path = os.path.join("models", model_save_name)
    
    # Load the trained model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    state = env.reset()
    total_reward = 0
    step = 0
    done = False
    
    # Lists for collecting data for plotting
    steps_list = []
    bess_power_list = []
    pv_power_list = []
    load_power_list = []
    grid_power_list = []
    soc_list = []
    
    while not done:
        current_index = env.current_idx
        p_pv_norm = env.pv_series.iloc[current_index]
        p_load_norm = env.load_series.iloc[current_index]
        p_pv = p_pv_norm * env.PVmax
        p_load = p_load_norm * env.Loadmax
        
        if model_type == "LSTMDQN":
            state_input = np.expand_dims(state, axis=0)   # (1, state_dim)
            state_input = np.expand_dims(state_input, axis=1)  # (1, 1, state_dim)
            state_tensor = torch.FloatTensor(state_input).to(device)
            q_values, _ = model(state_tensor)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = model(state_tensor)
        
        action_idx = q_values.argmax().item()
        action = np.array([discrete_actions[action_idx]])
        
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        steps_list.append(step)
        bess_power_list.append(info["p_bess"])
        pv_power_list.append(p_pv)
        load_power_list.append(p_load)
        grid_power_list.append(info["p_grid"])
        soc_list.append(env.soc)
        
        state = next_state
        step += 1

    # Plot power values
    plt.figure(figsize=(10, 6))
    plt.plot(steps_list, bess_power_list, label="BESS Power (kW)", marker="o")
    plt.plot(steps_list, pv_power_list, label="PV Power (kW)", marker="s")
    plt.plot(steps_list, load_power_list, label="Load Power (kW)", marker="^")
    plt.plot(steps_list, grid_power_list, label="EDS/Grid Power (kW)", marker="d")
    plt.xlabel("Step")
    plt.ylabel("Power (kW)")
    plt.title("Test: Power Values of BESS, PV, Load and EDS")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot Battery SoC over steps
    plt.figure(figsize=(10, 6))
    plt.plot(steps_list, soc_list, label="Battery SoC", color="tab:blue", marker="o")
    plt.xlabel("Step")
    plt.ylabel("Battery SoC")
    plt.title("Battery SoC Over Steps")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"Total Reward for test episode: {total_reward:.2f}")

if __name__ == "__main__":
    # Example test indices (modify as needed)
    START_IDX = 288 * 30  # e.g., start at day 30 (assuming 288 steps per day)
    END_IDX = START_IDX + 288  # one day of data
    test_model(START_IDX, END_IDX)
