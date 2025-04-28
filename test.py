import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from opt.env import EnergyEnv
from opt.D3QN.model import NHMHADDQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_history_buffer(history_len, state_dim, first_state):
    buf = deque(maxlen=history_len)
    zero = np.zeros(state_dim, dtype=np.float32)
    for _ in range(history_len - 1):
        buf.append(zero.copy())
    buf.append(first_state)
    return buf


def stack_history(history_deque):
    return np.stack(history_deque, axis=0)


def test_nhmhaddqn(start_idx, end_idx, train_data=None, ckpt_path="checkpoints/NHMHADDQN/hmhaddqn_energy_day1.pth"):
    # load parameters
    with open("data/parameters.json", "r") as f:
        global_params = json.load(f)
    with open("data/models.json", "r") as f:
        model_data = json.load(f)

    # only NHMHADDQN
    model_key = "NHMHADDQN"
    params = model_data[model_key]

    # environment
    episode_len = end_idx - start_idx
    env = EnergyEnv(
        data_dir="data",
        observations=params["observations"],
        start_idx=start_idx,
        episode_length=episode_len,
        test=True,
        data=train_data
    )

    # discrete actions
    a_low, a_high = env.action_space.low[0], env.action_space.high[0]
    discrete_actions = np.linspace(a_low, a_high, params["discrete_action_size"])

    # load model
    state_dim = env.observation_space.shape[0]
    history_len = params.get("history_len", 4)
    hl_size = params.get("hl_size", 128)

    model = NHMHADDQN(
        state_dim,
        discrete_actions.size,
        history_len=history_len,
        d_model=hl_size,
        num_heads=params.get("num_heads", 4),
        d_ff=params.get("ff_dim", 128)
    ).to(device)

    # ckpt_name = params["model_save_name"].replace(".pth", "_day1.pth")
    # ckpt_path = os.path.join("checkpoints", model_key, ckpt_name)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # initial state with history
    flat = env.reset()
    buffer = build_history_buffer(history_len, state_dim, flat)
    state = stack_history(buffer)

    # run
    total_reward = 0.0
    time_list, bess_list, pv_list, load_list, grid_list, soc_list = [], [], [], [], [], []
    done = False

    while not done:
        inp = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_vals = model(inp)
        act_idx = q_vals.argmax().item()
        action = np.array([discrete_actions[act_idx]], dtype=np.float32)

        next_flat, reward, done, info = env.step(action)
        total_reward += reward

        buffer.append(next_flat)
        state = stack_history(buffer)

        idx = env.current_idx - 1
        time_list.append(info["time"])
        pv_list.append(env.pv_series.iloc[idx] * env.PVmax)
        load_list.append(env.load_series.iloc[idx] * env.Loadmax)
        bess_list.append(info["p_bess"])
        grid_list.append(info["p_grid"])
        soc_list.append(env.soc)

    # plot results
    x = np.arange(len(time_list))
    ticks = [t.strftime("%H:%M") for t in time_list[::12]]

    plt.figure(figsize=(10, 6))
    plt.bar(x, bess_list, label="BESS (kW)", alpha=0.5)
    plt.plot(x, pv_list,   label="PV (kW)")
    plt.plot(x, load_list, label="Load (kW)")
    plt.plot(x, grid_list, label="Grid (kW)")
    plt.xticks(x[::12], ticks, rotation=45)
    plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(x, soc_list, label="SoC", marker="o")
    plt.xticks(x[::12], ticks, rotation=45)
    plt.legend(); plt.tight_layout(); plt.show()

    print(f"Total Reward: {total_reward:.2f}")


if __name__ == "__main__":
    model_path = "checkpoints/NHMHADDQN/hmhaddqn_energy_day2.pth"
    start_idx = 0
    end_idx = 288
    train_data = None
    test_nhmhaddqn(start_idx, end_idx, train_data, model_path)
