import os, random, json
from collections import deque

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm

# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Relative imports: EnergyEnv is located in the parent folder (opt/env.py)
from ..env import EnergyEnv
from .model import DDQN, CDDQN, MHADDQN

# -------------------------------
# Replay Buffer and Action Selection
# -------------------------------
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

def select_action(state_input, net, epsilon, actions, model_type):
    if random.random() < epsilon:
        return random.randrange(len(actions))
    if model_type == "CDDQN":
        main_input, cond_input = state_input
        main_tensor = torch.FloatTensor(main_input).unsqueeze(0).to(device)
        cond_tensor = torch.FloatTensor(cond_input).unsqueeze(0).to(device)
        q_values = net(main_tensor, cond_tensor)
    else:
        state_tensor = torch.FloatTensor(state_input).unsqueeze(0).to(device)
        q_values = net(state_tensor)
    return q_values.argmax().item()

# -------------------------------
# Training Function
# -------------------------------
def train():
    with open(os.path.join("data", "parameters.json"), "r") as f:
        global_params = json.load(f)
    with open(os.path.join("opt/DDQN", "models.json"), "r") as f:
        model_data = json.load(f)

    model_type = global_params["MODEL"]["MODEL_TYPE"].upper()
    train_params = model_data[model_type]

    num_episodes      = train_params["num_episodes"]
    episode_length    = train_params["episode_length"]
    batch_size        = train_params["batch_size"]
    gamma             = train_params["gamma"]
    epsilon_start     = train_params["epsilon_start"]
    epsilon_final     = train_params["epsilon_final"]
    epsilon_decay     = train_params["epsilon_decay"]
    target_update     = train_params["target_update"]
    buffer_capacity   = train_params["replay_buffer_capacity"]
    learning_rate     = train_params["learning_rate"]
    discrete_size     = train_params["discrete_action_size"]
    hl_number         = train_params.get("hl_number", 5)
    hl_size           = train_params.get("hl_size", 128)
    start_idx         = train_params["start_idx"]
    model_save_name   = train_params.get("model_save_name", "ddqn_energy.pth")
    reward_json_name  = train_params.get("reward_json_name", "episode_rewards_ddqn.json")
    training_txt_name = train_params.get("training_txt_name", "training_details_ddqn.txt")

    env = EnergyEnv(data_dir="data", observations=train_params["observations"],
                    start_idx=start_idx, episode_length=episode_length)
    flat_state = env.reset()  
    obs_keys = train_params["observations"]
    full_obs = {key: flat_state[i] for i, key in enumerate(obs_keys)}

    state_dim = env.observation_space.shape[0]
    a_low, a_high = env.action_space.low[0], env.action_space.high[0]
    discrete_actions = np.linspace(a_low, a_high, discrete_size)
    action_dim = len(discrete_actions)

    if model_type == "CDDQN":
        main_obs_keys = train_params.get("main_observations")
        cond_obs_keys = train_params.get("conditional_observations")
        if main_obs_keys is None or cond_obs_keys is None:
            raise ValueError("For CDDQN, 'main_observations' and 'conditional_observations' must be defined in models.json.")
        main_dim = len(main_obs_keys)
        cond_dim = len(cond_obs_keys)
        policy_net = CDDQN(main_dim, cond_dim, action_dim, hl_number=hl_number,
                           hl_size=hl_size, dropout_rate=train_params.get("dropout_rate", 0.0)).to(device)
        target_net = CDDQN(main_dim, cond_dim, action_dim, hl_number=hl_number,
                           hl_size=hl_size, dropout_rate=train_params.get("dropout_rate", 0.0)).to(device)
    elif model_type == "MHADDQN":
        num_heads = train_params.get("num_heads", 4)
        ff_dim    = train_params.get("ff_dim", 128)
        policy_net = MHADDQN(state_dim, action_dim, hl_size=hl_size, num_heads=num_heads, ff_dim=ff_dim).to(device)
        target_net = MHADDQN(state_dim, action_dim, hl_size=hl_size, num_heads=num_heads, ff_dim=ff_dim).to(device)
    else:
        policy_net = DDQN(state_dim, action_dim, hl_number=hl_number, hl_size=hl_size).to(device)
        target_net = DDQN(state_dim, action_dim, hl_number=hl_number, hl_size=hl_size).to(device)

    policy_net.train()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_capacity)

    episode_rewards, training_details = [], []
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        flat_state = env.reset()
        # Reconstruct full_obs from the flat vector
        full_obs = {key: flat_state[i] for i, key in enumerate(obs_keys)}
        
        if model_type == "CDDQN":
            # Build state tuple for conditional model using keys from configuration.
            main_state = np.array([full_obs[k] for k in main_obs_keys])
            cond_state = np.array([full_obs[k] for k in cond_obs_keys])
            state_input = (main_state, cond_state)
        else:
            state_input = flat_state
        
        total_reward = 0
        for step in range(episode_length):
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-episode / epsilon_decay)
            action_idx = select_action(state_input, policy_net, epsilon, discrete_actions, model_type)
            action = np.array([discrete_actions[action_idx]])
            
            next_output = env.step(action)
            if isinstance(next_output, tuple):
                next_state, reward, done, _ = next_output
            else:
                next_state, reward, done, _ = next_output, 0, False, {}
            replay_buffer.push(state_input, action_idx, reward, next_state, done)
            total_reward += reward
            
            # Prepare next state for model input.
            flat_next = env._get_obs()[0]  # Use _get_obs() to obtain the new flat state.
            next_full = {key: flat_next[i] for i, key in enumerate(obs_keys)}
            if model_type == "CDDQN":
                next_main = np.array([next_full[k] for k in main_obs_keys])
                next_cond = np.array([next_full[k] for k in cond_obs_keys])
                next_state_input = (next_main, next_cond)
            else:
                next_state_input = flat_next
            
            state_input = next_state_input
            
            # Process replay buffer if enough samples exist.
            if len(replay_buffer) > batch_size:
                b_state, b_action, b_reward, b_next_state, b_done = replay_buffer.sample(batch_size)
                if model_type == "CDDQN":
                    # For each state in b_state, check if it is a flat vector.
                    b_main, b_cond = [], []
                    # Get indices for main and conditional observations based on obs_keys.
                    main_indices = [obs_keys.index(k) for k in main_obs_keys]
                    cond_indices = [obs_keys.index(k) for k in cond_obs_keys]
                    for s in b_state:
                        # s is expected to be a flat vector if reset() returned only the flat_obs.
                        if isinstance(s, (list, np.ndarray)) and s.ndim == 1 and len(s) == len(obs_keys):
                            m = s[main_indices]
                            c = s[cond_indices]
                        else:
                            m, c = s
                        b_main.append(m)
                        b_cond.append(c)
                    b_main = torch.tensor(np.array(b_main), dtype=torch.float).to(device)
                    b_cond = torch.tensor(np.array(b_cond), dtype=torch.float).to(device)

                    
                    # Process next state in a similar fashion.
                    b_next_main, b_next_cond = [], []
                    for s in b_next_state:
                        if isinstance(s, (list, np.ndarray)) and s.ndim == 1 and len(s) == len(obs_keys):
                            m = s[main_indices]
                            c = s[cond_indices]
                        else:
                            m, c = s
                        b_next_main.append(m)
                        b_next_cond.append(c)
                    b_next_main = torch.tensor(np.array(b_next_main), dtype=torch.float).to(device)
                    b_next_cond = torch.tensor(np.array(b_next_cond), dtype=torch.float).to(device)
                else:
                    b_state = torch.FloatTensor(b_state).to(device)
                    b_next_state = torch.FloatTensor(b_next_state).to(device)
                
                b_action  = torch.LongTensor(b_action).unsqueeze(1).to(device)
                b_reward  = torch.FloatTensor(b_reward).unsqueeze(1).to(device)
                b_done    = torch.FloatTensor(b_done).unsqueeze(1).to(device)
                
                if model_type == "CDDQN":
                    curr_q = policy_net(b_main, b_cond).gather(1, b_action)
                    next_q_policy = policy_net(b_next_main, b_next_cond)
                else:
                    curr_q = policy_net(b_state).gather(1, b_action)
                    next_q_policy = policy_net(b_next_state)
                
                next_actions = next_q_policy.argmax(dim=1, keepdim=True)
                if model_type == "CDDQN":
                    next_q = target_net(b_next_main, b_next_cond).gather(1, next_actions)
                else:
                    next_q = target_net(b_next_state).gather(1, next_actions)
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
            tqdm.write(f"Episode {episode}: Reward = {total_reward:.2f}, Epsilon = {epsilon:.2f}")
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", model_save_name)
    torch.save(policy_net.state_dict(), model_path)
    print(f"Training complete. Model saved to {model_path}")
    
    with open(os.path.join("models", training_txt_name), "w") as f:
        f.write("Training Details:\n")
        for d in training_details:
            f.write(f"Episode {d['episode']}: Reward = {d['total_reward']:.2f}, Steps = {d['steps']}, Epsilon = {d['epsilon']:.2f}\n")
    with open(os.path.join("models", reward_json_name), "w") as f:
        json.dump({"model": model_save_name, "episode_rewards": episode_rewards}, f, indent=4)
    print("Training logs saved.")

# -------------------------------
# Main Execution Block
# -------------------------------
if __name__ == "__main__":
    with open(os.path.join("data", "parameters.json"), "r") as f:
        global_params = json.load(f)
    mode = global_params["ENV"]["MODE"].upper()
    
    if mode == "TRAIN":
        train()
    else:
        raise NotImplementedError("Test mode not implemented in this integrated script.")
