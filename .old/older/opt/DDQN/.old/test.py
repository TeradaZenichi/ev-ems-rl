import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from opt.env import EnergyEnv
from opt.DDQN.model import DDQN, CDDQN, MHADDQN

def test_model(START_IDX, END_IDX):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join("data", "parameters.json"), "r") as f:
        global_params = json.load(f)
    with open(os.path.join("opt/DDQN", "models.json"), "r") as f:
        model_data = json.load(f)

    model_type = global_params["MODEL"]["MODEL_TYPE"].upper()
    model_params = model_data[model_type]

    episode_length = END_IDX - START_IDX
    obs_keys = model_params.get("observations")
    env = EnergyEnv(data_dir="data", observations=obs_keys, start_idx=START_IDX, episode_length=episode_length, test=True)
    a_low, a_high = env.action_space.low[0], env.action_space.high[0]
    discrete_actions = np.linspace(a_low, a_high, model_params["discrete_action_size"])
    action_dim = len(discrete_actions)

    hl_number = model_params.get("hl_number", 5)
    hl_size = model_params.get("hl_size", 128)
    model_save_name = model_params.get("model_save_name")

    if model_type == "CDDQN":
        main_obs_keys = model_params["main_observations"]
        cond_obs_keys = model_params["conditional_observations"]
        main_dim = len(main_obs_keys)
        cond_dim = len(cond_obs_keys)
        dropout_rate = model_params.get("dropout_rate", 0.0)
        model = CDDQN(main_dim, cond_dim, action_dim, hl_number=hl_number,
                      hl_size=hl_size, dropout_rate=dropout_rate).to(device)
    elif model_type == "MHADDQN":
        state_dim = env.observation_space.shape[0]
        num_heads = model_params.get("num_heads", 4)
        ff_dim = model_params.get("ff_dim", 128)
        model = MHADDQN(state_dim, action_dim, hl_size=hl_size, num_heads=num_heads, ff_dim=ff_dim).to(device)
    else:
        state_dim = env.observation_space.shape[0]
        model = DDQN(state_dim, action_dim, hl_number=hl_number, hl_size=hl_size).to(device)

    model.load_state_dict(torch.load(os.path.join("models", model_save_name), map_location=device))
    model.eval()

    state = env.reset()
    full_obs = {k: state[i] for i, k in enumerate(obs_keys)}

    if model_type == "CDDQN":
        state = (
            np.array([full_obs[k] for k in main_obs_keys]),
            np.array([full_obs[k] for k in cond_obs_keys])
        )

    total_reward, step, done = 0, 0, False
    time_list, bess_power_list, pv_power_list = [], [], []
    load_power_list, grid_power_list, soc_list = [], [], []

    while not done:
        idx = env.current_idx
        p_pv = env.pv_series.iloc[idx] * env.PVmax
        p_load = env.load_series.iloc[idx] * env.Loadmax

        if model_type == "CDDQN":
            main_tensor = torch.FloatTensor(state[0]).unsqueeze(0).to(device)
            cond_tensor = torch.FloatTensor(state[1]).unsqueeze(0).to(device)
            q_values = model(main_tensor, cond_tensor)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = model(state_tensor)

        action_idx = q_values.argmax().item()
        action = np.array([discrete_actions[action_idx]])
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        state = next_state
        if model_type == "CDDQN":
            full_obs = {k: state[i] for i, k in enumerate(obs_keys)}
            state = (
                np.array([full_obs[k] for k in main_obs_keys]),
                np.array([full_obs[k] for k in cond_obs_keys])
            )

        time_list.append(info["time"])
        bess_power_list.append(info["p_bess"])
        pv_power_list.append(p_pv)
        load_power_list.append(p_load)
        grid_power_list.append(info["p_grid"])
        soc_list.append(env.soc)
        step += 1

    x = np.arange(len(time_list))
    tick_labels = [t.strftime("%H:%M") for t in time_list[::12]]

    plt.figure(figsize=(10, 6))
    plt.bar(x, bess_power_list, label="BESS Power (kW)", alpha=0.5)
    plt.plot(x, pv_power_list, label="PV Power (kW)", marker="s")
    plt.plot(x, load_power_list, label="Load Power (kW)", marker="o")
    plt.plot(x, grid_power_list, label="Grid Power (kW)", marker="d")
    plt.xticks(x[::12], tick_labels, rotation=45)
    plt.ylabel("Power (kW)")
    plt.title(f"{model_type} Test: Power Flow")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x, soc_list, label="Battery SoC", marker="o")
    plt.xticks(x[::12], tick_labels, rotation=45)
    plt.ylabel("SoC")
    plt.title("Battery SoC Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    START_IDX = 288 * 30
    END_IDX = START_IDX + 288
    test_model(START_IDX, END_IDX)
