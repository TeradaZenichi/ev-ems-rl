import os, json, torch, numpy as np, matplotlib.pyplot as plt
from collections import deque
from opt.env import EnergyEnv
from opt.DDQN.model import DDQN, CDDQN, MHADDQN, HMHADDQN      # ①  novo import

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_history_buffer(history_len, state_dim, first_state):
    """Cria deque com padding‑zero para as primeiras chamadas."""
    buf = deque(maxlen=history_len)
    zero_pad = np.zeros(state_dim, dtype=np.float32)
    for _ in range(history_len - 1):
        buf.append(zero_pad.copy())
    buf.append(first_state)
    return buf


def stack_history(history_deque):
    """Converte deque (hist_len itens) em array (hist_len, state_dim)."""
    return np.stack(history_deque, axis=0)


def test_model(start_idx, end_idx):
    # ---------- carregamento de configurações ----------
    with open("data/parameters.json", "r") as f:
        global_params = json.load(f)
    with open("opt/DDQN/models.json", "r") as f:
        model_data = json.load(f)

    model_type = global_params["MODEL"]["MODEL_TYPE"].upper()
    params = model_data[model_type]

    # ---------- ambiente ----------
    episode_len = end_idx - start_idx
    env = EnergyEnv(data_dir="data",
                    observations=params["observations"],
                    start_idx=start_idx,
                    episode_length=episode_len,
                    test=True)

    # ---------- espaço de ações ----------
    a_low, a_high = env.action_space.low[0], env.action_space.high[0]
    discrete_actions = np.linspace(a_low, a_high, params["discrete_action_size"])
    action_dim = len(discrete_actions)

    # ---------- criação do modelo ----------
    state_dim = env.observation_space.shape[0]
    hl_size   = params.get("hl_size", 128)

    if model_type == "CDDQN":
        main_k, cond_k = params["main_observations"], params["conditional_observations"]
        model = CDDQN(len(main_k), len(cond_k), action_dim,
                      hl_number=params.get("hl_number", 3),
                      hl_size=hl_size,
                      dropout_rate=params.get("dropout_rate", 0.0)).to(device)
    elif model_type == "MHADDQN":
        model = MHADDQN(state_dim, action_dim,
                        hl_size=hl_size,
                        num_heads=params.get("num_heads", 4),
                        ff_dim=params.get("ff_dim", 128)).to(device)
    elif model_type == "HMHADDQN":                                                        # ②  novo caso
        model = HMHADDQN(state_dim, action_dim,
                         history_len=params.get("history_len", 4),
                         hl_size=hl_size,
                         num_heads=params.get("num_heads", 4),
                         ff_dim=params.get("ff_dim", 128)).to(device)
    else:  # DDQN
        model = DDQN(state_dim, action_dim,
                     hl_number=params.get("hl_number", 3),
                     hl_size=hl_size).to(device)

    # model.load_state_dict(torch.load(os.path.join("models", params["model_save_name"]),
    #                                  map_location=device))
    
    #load checkpoints\MHADDQN\MHADDQN_day1.pth
    model.load_state_dict(torch.load(os.path.join("checkpoints", model_type, "MHADDQN_day1.pth"),
                                     map_location=device))


    model.eval()

    # ---------- inicia episódio ----------
    state_flat = env.reset()
    full_obs   = {k: state_flat[i] for i, k in enumerate(params["observations"])}

    # estado de entrada específico por tipo ---------------------------------
    if model_type == "CDDQN":
        state = (np.array([full_obs[k] for k in main_k]),
                 np.array([full_obs[k] for k in cond_k]))
    elif model_type == "HMHADDQN":                                        # ③  cria histórico
        hist_len  = params.get("history_len", 4)
        hist_buf  = build_history_buffer(hist_len, state_dim, state_flat)
        state_hist = stack_history(hist_buf)              # (hist_len, state_dim)
        state = state_hist
    else:
        state = state_flat
    # -----------------------------------------------------------------------

    # métricas para plot
    total_reward = 0
    t_list = []; bess_p = []; pv_p = []; load_p = []; grid_p = []; soc_l = []

    done = False
    while not done:
        # -------------- forward --------------
        if model_type == "CDDQN":
            inp0 = torch.FloatTensor(state[0]).unsqueeze(0).to(device)
            inp1 = torch.FloatTensor(state[1]).unsqueeze(0).to(device)
            q_vals = model(inp0, inp1)
        elif model_type == "HMHADDQN":
            inp = torch.FloatTensor(state).unsqueeze(0).to(device)          # (1, H, dim)
            q_vals = model(inp)
        else:
            inp = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_vals = model(inp)

        act_idx = q_vals.argmax().item()
        action  = np.array([discrete_actions[act_idx]])

        # -------------- ambiente --------------
        next_state_flat, reward, done, info = env.step(action)
        total_reward += reward

        # -------------- próximo estado formatado --------------
        if model_type == "CDDQN":
            full_n = {k: next_state_flat[i] for i, k in enumerate(params["observations"])}
            state = (np.array([full_n[k] for k in main_k]),
                     np.array([full_n[k] for k in cond_k]))
        elif model_type == "HMHADDQN":
            hist_buf.append(next_state_flat)
            state = stack_history(hist_buf)
        else:
            state = next_state_flat

        # -------------- logs --------------
        idx = env.current_idx - 1
        t_list.append(info["time"])
        pv_p.append(env.pv_series.iloc[idx] * env.PVmax)
        load_p.append(env.load_series.iloc[idx] * env.Loadmax)
        bess_p.append(info["p_bess"]); grid_p.append(info["p_grid"]); soc_l.append(env.soc)

    # ---------- plots ----------
    x = np.arange(len(t_list)); ticks = [t.strftime("%H:%M") for t in t_list[::12]]

    plt.figure(figsize=(10, 6))
    plt.bar(x, bess_p, label="BESS (kW)", alpha=.5)
    plt.plot(x, pv_p,   label="PV (kW)"); plt.plot(x, load_p, label="Load (kW)")
    plt.plot(x, grid_p, label="Grid (kW)")
    plt.xticks(x[::12], ticks, rotation=45); plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(x, soc_l, label="SoC", marker="o")
    plt.xticks(x[::12], ticks, rotation=45); plt.legend(); plt.tight_layout(); plt.show()

    print(f"Total Reward: {total_reward:.2f}")


if __name__ == "__main__":
    START = 288 * 30
    test_model(START, START + 288)
