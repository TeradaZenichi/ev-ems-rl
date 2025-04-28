import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from opt.env import EnergyEnv
from opt.model import NHMHADDQN

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Diretório raiz de resultados
RESULTS_ROOT = "results"
os.makedirs(RESULTS_ROOT, exist_ok=True)

# Número de dias de modelo a avaliar (model days)
NUM_MODEL_DAYS = 16  # ajuste conforme necessário

def build_model(state_dim, action_dim, cfg, model_file):
    """Instancia NHMHADDQN e carrega checkpoint específico"""
    hist   = cfg.get("history_len", 4)
    hl     = cfg.get("hl_size", 128)
    heads  = cfg.get("num_heads", 4)
    ff_dim = cfg.get("ff_dim", 128)

    net = NHMHADDQN(
        state_dim,
        action_dim,
        history_len=hist,
        d_model=hl,
        num_heads=heads,
        d_ff=ff_dim
    ).to(device)

    ckpt_path = os.path.join("checkpoints", model_file)
    print(model_file)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()
    return net


def select_action(state, net):
    """Seleciona ação via NHMHADDQN"""
    tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q = net(tensor)
    return q.argmax().item()


def evaluate_checkpoints(num_model_days):
    # Carrega configurações
    gp = json.load(open("data/parameters.json"))
    mc = json.load(open("data/models.json"))
    cfg = mc["NHMHADDQN"]
    ndays_env = gp.get("NUM_DAYS", 30)

    # Dummy env para dimensões e ações
    env0 = EnergyEnv(
        data_dir="data",
        observations=cfg["observations"],
        start_idx=0,
        episode_length=1,
        test=True,
        data="train"
    )
    state_dim = env0.observation_space.shape[0]
    a_low, a_high = env0.action_space.low[0], env0.action_space.high[0]
    actions = np.linspace(a_low, a_high, cfg["discrete_action_size"])

    # Base do nome de checkpoint
    base_name = cfg.get("model_save_name", "model.pth").replace(".pth", "")

    summary = {}
    # Loop sobre dias de modelo
    for day_model in range(1, num_model_days + 1):
        model_file = f"{base_name}_day{day_model}.pth"
        model = build_model(state_dim, len(actions), cfg, model_file)

        # Pasta para este modelo
        model_dir = os.path.join(RESULTS_ROOT, f"model_day{day_model}")
        os.makedirs(model_dir, exist_ok=True)

        costs = []
        # Loop sobre dias de ambiente
        for env_day in range(1, ndays_env + 1):
            start_idx = (env_day - 1) * 288
            env = EnergyEnv(
                data_dir="data",
                observations=cfg["observations"],
                start_idx=start_idx,
                episode_length=288,
                test=True,
                data="train"
            )
            # Histórico inicial
            hist = cfg.get("history_len", 4)
            obs = env.reset()
            seq = np.zeros((hist, state_dim), dtype=np.float32)
            seq[-1] = obs
            state = seq

            total_cost = 0.0
            trace = {"time": [], "p_bess": [], "pv": [], "load": [], "grid": [], "soc": []}
            done = False

            while not done:
                idx = select_action(state, model)
                action = np.array([actions[idx]], dtype=np.float32)
                next_obs, _, done, info = env.step(action)
                total_cost += info["cost"]

                seq = np.roll(state, -1, axis=0)
                seq[-1] = next_obs
                state = seq

                ts = info["time"]
                trace["time"].append(ts)
                trace["p_bess"].append(info["p_bess"])
                trace["pv"].append(env.pv_series.loc[ts] * env.PVmax)
                trace["load"].append(env.load_series.loc[ts] * env.Loadmax)
                trace["grid"].append(info["p_grid"])
                trace["soc"].append(env.soc)

            costs.append(total_cost)

            # Salva plots diretamente em model_dir
            x = np.arange(len(trace["time"]))
            ticks = [t.strftime("%H:%M") for t in trace["time"][::12]]

            # Power plot
            plt.figure(figsize=(10, 6))
            plt.bar(x, trace["p_bess"], label="BESS (kW)", alpha=0.5)
            plt.plot(x, trace["pv"],   label="PV (kW)")
            plt.plot(x, trace["load"], label="Load (kW)")
            plt.plot(x, trace["grid"], label="Grid (kW)")
            plt.xticks(x[::12], ticks, rotation=45)
            plt.legend(); plt.tight_layout()
            fname_p = f"model_{day_model}_day_{env_day}_power.pdf"
            plt.savefig(os.path.join(model_dir, fname_p))
            plt.close()

            # SoC plot
            plt.figure(figsize=(10, 4))
            plt.plot(x, trace["soc"], label="SoC", marker="o")
            plt.xticks(x[::12], ticks, rotation=45)
            plt.legend(); plt.tight_layout()
            fname_s = f"model_{day_model}_day_{env_day}_soc.pdf"
            plt.savefig(os.path.join(model_dir, fname_s))
            plt.close()

        summary[f"model_day{day_model}"] = costs

    # Salvar resumo geral JSON
    summary_path = os.path.join(RESULTS_ROOT, "summary_performance.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot resumo geral (PDF)
    plt.figure(figsize=(12, 8))
    for key, costs in summary.items():
        days = np.arange(1, len(costs) + 1)
        plt.plot(days, costs, marker="o", label=key)
    plt.xlabel("Env Day")
    plt.ylabel("Total Cost")
    plt.title("Performance per Env Day for each Model Day")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_ROOT, "summary_performance.pdf"))
    plt.show()

    # Plotar custo total por model_day
    total_costs = [sum(costs) for costs in summary.values()]
    model_days = np.arange(1, len(total_costs) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(model_days, total_costs, marker="o")
    plt.xlabel("Model Day")
    plt.ylabel("Total Cost (sum over env days)")
    plt.title("Total Cost per Model Day")
    plt.xticks(model_days)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_ROOT, "total_cost_per_model_day.pdf"))
    plt.show()

if __name__ == "__main__":
    evaluate_checkpoints(NUM_MODEL_DAYS)
