import os, random, json
from collections import deque

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm

from ..env import EnergyEnv
from .model import DQN, LSTMDQN, SelfAttentionDQN

# Set device for training (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(args)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))
    
    def __len__(self):
        return len(self.buffer)

def select_action(state, net, epsilon, actions, model_type, seq_len=1):
    # Expand dimensions for sequential models
    if model_type in ["LSTMDQN", "SelfAttentionDQN"]:
        state = np.expand_dims(np.expand_dims(state, axis=0), axis=1)
    if random.random() < epsilon:
        return random.randrange(len(actions))
    state_tensor = torch.FloatTensor(state).to(device)
    # Para LSTMDQN desempacota a tupla; para os demais, usa diretamente
    q_values = net(state_tensor) if model_type != "LSTMDQN" else net(state_tensor)[0]
    return q_values.argmax().item()

def train():
    # Carrega configurações dos arquivos JSON
    with open(os.path.join("data", "parameters.json"), "r") as f:
        params = json.load(f)
    with open(os.path.join("opt/DQN", "models.json"), "r") as f:
        model_data = json.load(f)
    
    model_type = params["MODEL"]["MODEL_TYPE"]
    train_params = model_data[model_type]

    # Extração dos hiperparâmetros
    num_episodes      = train_params["num_episodes"]
    max_steps         = train_params["episode_length"]
    batch_size        = train_params["batch_size"]
    gamma             = train_params["gamma"]
    epsilon_start     = train_params["epsilon_start"]
    epsilon_final     = train_params["epsilon_final"]
    epsilon_decay     = train_params["epsilon_decay"]
    target_update     = train_params["target_update"]
    buffer_capacity   = train_params["replay_buffer_capacity"]
    learning_rate     = train_params["learning_rate"]
    discrete_size     = train_params["discrete_action_size"]
    seq_len           = train_params.get("seq_len", 1)
    start_idx         = train_params["start_idx"]
    episode_length    = train_params["episode_length"]
    model_save_name   = train_params.get("model_save_name", "dqn_energy.pth")
    reward_json_name  = train_params.get("reward_json_name", "episode_rewards.json")
    training_txt_name = train_params.get("training_txt_name", "training_details.txt")
    
    # Cria ambiente e define espaço de ação e observação
    env = EnergyEnv(data_dir="data", observations=train_params["observations"],
                    start_idx=start_idx, episode_length=episode_length)
    state_dim = env.observation_space.shape[0]
    a_low, a_high = env.action_space.low[0], env.action_space.high[0]
    discrete_actions = np.linspace(a_low, a_high, discrete_size)
    action_dim = len(discrete_actions)
    
    # Seleciona e instancia o modelo
    if model_type == "LSTMDQN":
        policy_net = LSTMDQN(state_dim, action_dim).to(device)
        target_net = LSTMDQN(state_dim, action_dim).to(device)
    elif model_type == "SelfAttentionDQN":
        policy_net = SelfAttentionDQN(
            input_dim=state_dim,
            action_dim=action_dim,
            embed_dim=train_params.get("embed_dim", 128),
            num_heads=train_params.get("num_heads", 4),
            num_layers=train_params.get("num_layers", 1),
            seq_len=seq_len
        ).to(device)
        target_net = SelfAttentionDQN(
            input_dim=state_dim,
            action_dim=action_dim,
            embed_dim=train_params.get("embed_dim", 128),
            num_heads=train_params.get("num_heads", 4),
            num_layers=train_params.get("num_layers", 1),
            seq_len=seq_len
        ).to(device)
    else:
        hidden_layers = train_params.get("hidden_layers", [128, 128, 128, 128, 128])
        policy_net = DQN(state_dim, action_dim, hidden_layers=hidden_layers).to(device)
        target_net = DQN(state_dim, action_dim, hidden_layers=hidden_layers).to(device)
    
    policy_net.train()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_capacity)
    
    episode_rewards, training_details = [], []
    for episode in tqdm(range(num_episodes), desc="Episodes"):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps):
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-episode / epsilon_decay)
            action_idx = select_action(state, policy_net, epsilon, discrete_actions, model_type, seq_len)
            action = np.array([discrete_actions[action_idx]])
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action_idx, reward, next_state, done)
            state, total_reward = next_state, total_reward + reward
            
            if len(replay_buffer) > batch_size:
                b_state, b_action, b_reward, b_next_state, b_done = replay_buffer.sample(batch_size)
                if model_type in ["LSTMDQN", "SelfAttentionDQN"]:
                    b_state = torch.FloatTensor(b_state).view(batch_size, 1, state_dim).to(device)
                    b_next_state = torch.FloatTensor(b_next_state).view(batch_size, 1, state_dim).to(device)
                else:
                    b_state = torch.FloatTensor(b_state).to(device)
                    b_next_state = torch.FloatTensor(b_next_state).to(device)
                b_action = torch.LongTensor(b_action).unsqueeze(1).to(device)
                b_reward = torch.FloatTensor(b_reward).unsqueeze(1).to(device)
                b_done = torch.FloatTensor(b_done).unsqueeze(1).to(device)
                
                if model_type == "LSTMDQN":
                    curr_q = policy_net(b_state)[0]
                    next_q = target_net(b_next_state)[0]
                else:
                    curr_q = policy_net(b_state)
                    next_q = target_net(b_next_state)
                    
                curr_q = curr_q.gather(1, b_action)
                next_q = next_q.detach().max(1)[0].unsqueeze(1)
                exp_q = b_reward + gamma * next_q * (1 - b_done)
                
                loss = nn.MSELoss()(curr_q, exp_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break

        episode_rewards.append(total_reward)
        training_details.append({
            "episode": episode,
            "total_reward": total_reward,
            "steps": step + 1,
            "epsilon": epsilon
        })
        if episode % 10 == 0:
            tqdm.write(f"Episode: {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    # Save model and details
    os.makedirs("models", exist_ok=True)
    torch.save(policy_net.state_dict(), os.path.join("models", model_save_name))
    print(f"Training complete. Model saved to models/{model_save_name}")
    
    with open(os.path.join("models", training_txt_name), "w") as f:
        f.write("Training Details:\n")
        for d in training_details:
            f.write(f"Episode {d['episode']}: Reward {d['total_reward']:.2f}, Steps {d['steps']}, Epsilon {d['epsilon']:.2f}\n")
    print(f"Training details saved to models/{training_txt_name}")
    
    with open(os.path.join("models", reward_json_name), "w") as f:
        json.dump({"model": model_save_name, "episode_rewards": episode_rewards}, f, indent=4)
    print(f"Episode rewards saved to models/{reward_json_name}")

if __name__ == "__main__":
    train()
