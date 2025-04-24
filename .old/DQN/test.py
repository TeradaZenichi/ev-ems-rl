import os, json
import torch, numpy as np, matplotlib.pyplot as plt
from opt.env import EnergyEnv
from opt.DQN.model import DQN, LSTMDQN, SelfAttentionDQN

def test_model(START_IDX, END_IDX):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carrega as configurações
    with open("data/parameters.json", "r") as f:
        params = json.load(f)
    model_path_base = params["MODEL"]["MODEL_PATH"]
    model_type = params["MODEL"]["MODEL_TYPE"]
    with open(os.path.join(model_path_base, "models.json"), "r") as f:
        model_data = json.load(f)
    test_params = model_data[model_type]
    
    discrete_action_size = test_params["discrete_action_size"]
    episode_length = END_IDX - START_IDX
    data_dir = "data"
    
    # Cria o ambiente com as observações definidas no arquivo de treino
    obs_keys = test_params.get("observations", None)
    env = EnergyEnv(data_dir=data_dir, observations=obs_keys,
                    start_idx=START_IDX, episode_length=episode_length, test=True)
    state_dim = env.observation_space.shape[0]
    
    # Define o espaço de ação
    action_low, action_high = env.action_space.low[0], env.action_space.high[0]
    discrete_actions = np.linspace(action_low, action_high, discrete_action_size)
    action_dim = len(discrete_actions)
    
    # Instancia o modelo conforme o tipo
    if model_type == "LSTMDQN":
        model = LSTMDQN(state_dim, action_dim).to(device)
        save_name = test_params.get("model_save_name", "lstm_dqn_energy.pth")
    elif model_type == "SelfAttentionDQN":
        embed_dim  = test_params.get("embed_dim", 128)
        num_heads  = test_params.get("num_heads", 4)
        num_layers = test_params.get("num_layers", 1)
        seq_len    = test_params.get("seq_len", 1)
        model = SelfAttentionDQN(input_dim=state_dim, action_dim=action_dim,
                                 embed_dim=embed_dim, num_heads=num_heads,
                                 num_layers=num_layers, seq_len=seq_len).to(device)
        save_name = test_params.get("model_save_name", "selfattention_energy.pth")
    else:
        hidden_layers = test_params.get("hidden_layers", [128, 128, 128, 128, 128])
        model = DQN(state_dim, action_dim, hidden_layers=hidden_layers).to(device)
        save_name = test_params.get("model_save_name", "dqn_energy.pth")

    
    model.load_state_dict(torch.load(os.path.join("models", save_name), map_location=device))
    model.eval()
    
    state, total_reward, step, done = env.reset(), 0, 0, False
    # Listas para plotagem
    steps_list, time_list = [], []
    bess_power_list, pv_power_list = [], []
    load_power_list, grid_power_list, soc_list = [], [], []
    
    while not done:
        # Calcula as potências absolutas
        current_index = env.current_idx
        p_pv = env.pv_series.iloc[current_index] * env.PVmax
        p_load = env.load_series.iloc[current_index] * env.Loadmax
        
        # Prepara a entrada conforme o modelo
        if model_type in ["LSTMDQN", "SelfAttentionDQN"]:
            state_tensor = torch.FloatTensor(np.expand_dims(np.expand_dims(state, axis=0), axis=1)).to(device)
            q_values = model(state_tensor) if model_type == "SelfAttentionDQN" else model(state_tensor)[0]
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = model(state_tensor)
        
        action_idx = q_values.argmax().item()
        action = np.array([discrete_actions[action_idx]])
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        # Armazena os dados para plotagem
        steps_list.append(step)
        time_list.append(info["time"])
        bess_power_list.append(info["p_bess"])
        pv_power_list.append(p_pv)
        load_power_list.append(p_load)
        grid_power_list.append(info["p_grid"])
        soc_list.append(env.soc)
        step += 1
    
    # Configura os eixos para plotagem
    time_str = [t.strftime("%H:%M") for t in time_list]
    x = np.arange(len(time_str))
    tick_interval = 12
    tick_positions = x[::tick_interval]
    tick_labels = [time_str[i] for i in tick_positions]
    
    # Plot dos valores de potência
    plt.figure(figsize=(10, 6))
    plt.bar(x, bess_power_list, label="BESS Power (kW)", color="blue", alpha=0.5)
    plt.plot(x, pv_power_list, label="PV Power (kW)", color="orange", marker="s")
    plt.plot(x, load_power_list, label="Load Power (kW)", color="black", marker="o")
    plt.plot(x, grid_power_list, label="EDS/Grid Power (kW)", color="green", marker="d")
    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    plt.title("Test: Power Values of BESS, PV, Load and EDS")
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot do estado da bateria (SoC)
    plt.figure(figsize=(10, 6))
    plt.plot(x, soc_list, label="Battery SoC", color="tab:blue", marker="o")
    plt.xlabel("Time")
    plt.ylabel("Battery SoC")
    plt.title("Battery SoC Over Time")
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print(f"Total Reward for test episode: {total_reward:.2f}")

if __name__ == "__main__":
    START_IDX = 288 * 30  # Exemplo: início no dia 30 (288 passos por dia)
    END_IDX = START_IDX + 288  # Um dia de dados
    test_model(START_IDX, END_IDX)
