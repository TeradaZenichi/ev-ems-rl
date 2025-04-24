import os
import glob
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..env import EnergyEnv
from .model import DDQN, CDDQN, MHADDQN, HMHADDQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(model_type, state_dim, action_dim, cfg):
    hl = cfg.get("hl_size", 128)
    if model_type == "CDDQN":
        md = len(cfg["main_observations"])
        cd = len(cfg["conditional_observations"])
        net = CDDQN(md, cd, action_dim, hl_size=hl).to(device)
    elif model_type == "HMHADDQN":
        hist = cfg.get("history_len", 4)
        nh   = cfg.get("num_heads", 4)
        ff   = cfg.get("ff_dim", 128)
        net = HMHADDQN(state_dim, action_dim, hist, hl, nh, ff).to(device)
    elif model_type == "NHMHADDQN":
        hist = cfg.get("history_len", 4)
        nh   = cfg.get("num_heads", 4)
        ff   = cfg.get("ff_dim", 128)
        net = HMHADDQN(state_dim, action_dim, hist, hl, nh, ff).to(device)
    elif model_type == "MHADDQN":
        nh = cfg.get("num_heads", 4)
        ff = cfg.get("ff_dim", 128)
        net = MHADDQN(state_dim, action_dim, hl_size=hl, num_heads=nh, ff_dim=ff).to(device)
    else:
        net = DDQN(state_dim, action_dim, hl_size=hl).to(device)
    return net

def select_action(state, net, model_type, actions):
    tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    if model_type == "CDDQN":
        main, cond = state
        q = net(torch.FloatTensor(main).unsqueeze(0).to(device),
                torch.FloatTensor(cond).unsqueeze(0).to(device))
    elif model_type == "HMHADDQN":
        q = net(tensor)
    else:
        q = net(tensor)
    return q.argmax().item()

def evaluate_checkpoints():
    # load configs
    with open("data/parameters.json") as f:
        gp = json.load(f)
    with open("opt/D3QN/models.json") as f:
        mc = json.load(f)

    model_type = gp["MODEL"]["MODEL_TYPE"].upper()
    ndays      = gp.get("NUM_DAYS", 30)
    cfg        = mc[model_type]
    disc_size  = cfg["discrete_action_size"]

    # dummy env for dims/actions
    env0 = EnergyEnv(
        data_dir="data",
        start_idx=0,
        episode_length=ndays*288,
        test=True,
        observations=cfg["observations"]
    )
    state_dim = env0.observation_space.shape[0]
    a_low, a_high = env0.action_space.low[0], env0.action_space.high[0]
    actions = np.linspace(a_low, a_high, disc_size)

    ckpt_dir = os.path.join("checkpoints", model_type)
    ckpt_paths = sorted(
        glob.glob(os.path.join(ckpt_dir, f"{model_type}_day*.pth")),
        key=lambda p: int(os.path.basename(p).split("day")[1].split(".")[0])
    )

    # summary of performance: model_basename -> list of total_cost per test day
    summary = {}

    for ckpt in ckpt_paths:
        basename = os.path.basename(ckpt).replace(".pth","")
        # load model
        model = build_model(model_type, state_dim, disc_size, cfg)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        # prepare output folder
        out_folder = os.path.join(ckpt_dir, basename)
        os.makedirs(out_folder, exist_ok=True)

        # init summary list
        summary[basename] = []

        # evaluate each test day
        for day in range(1, ndays+1):
            start_idx = (day-1) * 288
            env = EnergyEnv(
                data_dir="data",
                start_idx=start_idx,
                episode_length=288,
                test=True,
                observations=cfg["observations"]
            )

            bess_p, pv_p, load_p, grid_p, soc_l, ticks = [], [], [], [], [], []
            total_cost_day = 0.0

            # reset and optional history
            state = env.reset()
            if model_type == "HMHADDQN":
                hist = cfg.get("history_len",4)
                seq = np.zeros((hist, state.shape[-1]), dtype=np.float32)
                seq[-1] = state
                state = seq

            for t in range(288):
                a_idx = select_action(state, model, model_type, actions)
                action = np.array([actions[a_idx]], dtype=np.float32)
                next_s, r, done, info = env.step(action)

                # collect data
                bess_p.append(info["p_bess"])
                ts = info["time"]
                pv_p.append(env.pv_series.loc[ts] * env.PVmax)
                load_p.append(env.load_series.loc[ts] * env.Loadmax)
                grid_p.append(info["p_grid"])
                soc_l.append(env.soc)
                total_cost_day += info["cost"]

                # update state
                if model_type == "HMHADDQN":
                    seq = np.roll(state, -1, axis=0)
                    seq[-1] = next_s
                    state = seq
                else:
                    state = next_s

                if t % 12 == 0:
                    ticks.append(ts.strftime("%H:%M"))
                if done:
                    break

            # save plots
            x = np.arange(len(bess_p))
            plt.figure(figsize=(10,6))
            plt.bar(x, bess_p, label="BESS (kW)", alpha=.5)
            plt.plot(x, pv_p,   label="PV (kW)")
            plt.plot(x, load_p, label="Load (kW)")
            plt.plot(x, grid_p, label="Grid (kW)")
            plt.xticks(x[::12], ticks, rotation=45)
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_folder, f"{basename}_day{day}_power.png"))
            plt.close()

            plt.figure(figsize=(10,4))
            plt.plot(x, soc_l, label="SoC", marker="o")
            plt.xticks(x[::12], ticks, rotation=45)
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_folder, f"{basename}_day{day}_soc.png"))
            plt.close()

            # record summary
            summary[basename].append(total_cost_day)

    # after all, plot summary
    plt.figure(figsize=(12,8))
    days = np.arange(1, ndays+1)
    for model_name, costs in summary.items():
        plt.plot(days, costs, marker='o', label=model_name)
    plt.xlabel("Test Day")
    plt.ylabel("Total Test Cost (kW·h × tariff)")
    plt.title(f"Performance per Day for {model_type} Models")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    summary_path = os.path.join(ckpt_dir, "summary_performance.png")
    plt.savefig(summary_path)
    plt.show()

    # also save the summary data
    json_path = os.path.join(ckpt_dir, "summary_performance.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"All evaluations complete. Summary saved at {summary_path} and {json_path}")

    # Load evaluation JSON file
    with open("checkpoints\MHADDQN\evaluation_results.json", "r") as f:
        data = json.load(f)

    # Determine metric key (e.g., 'total_test_cost' or 'total_cost')
    # We'll check one entry to see which key exists
    sample = next(iter(data.values()))
    metric_key = "total_test_cost" if "total_test_cost" in sample else "total_cost"

    # Extract days and corresponding metric
    days = sorted(int(k.split("_")[1]) for k in data.keys())
    metrics = [data[f"day_{d}"][metric_key] for d in days]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(days, metrics, marker='o', linestyle='-')
    plt.xlabel("Dia")
    plt.ylabel(metric_key.replace("_", " ").capitalize())
    plt.title(f"Impacto de cada dia na métrica global ({metric_key})")
    plt.xticks(days)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("checkpoints/MHADDQN/performance.pdf")


    


if __name__ == "__main__":
    evaluate_checkpoints()
