import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from collections import deque
from tqdm import tqdm
from ..env import EnergyEnv  # Import environment from one level up
from .model import DQN      # Import DQN from the same package
from .model import LSTMDQN  # Import LSTMDQN from the same package

# Set device for training (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay Buffer class for storing experiences
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

def select_action(state, policy_net, epsilon, discrete_actions, model_type, seq_len=1):
    # If model_type is LSTMDQN, reshape the state to add sequence dimensions.
    if model_type == "LSTMDQN":
        state = np.expand_dims(state, axis=0)   # (1, state_dim)
        state = np.expand_dims(state, axis=1)    # (1, 1, state_dim)
    if random.random() < epsilon:
        return random.randrange(len(discrete_actions))
    else:
        state_tensor = torch.FloatTensor(state).to(device)
        if model_type == "LSTMDQN":
            q_values, _ = policy_net(state_tensor)
        else:
            q_values = policy_net(state_tensor)
        return q_values.argmax().item()

def train():
    # Load training hyperparameters from parameters.json
    params_path = os.path.join("data", "parameters.json")
    with open(params_path, "r") as f:
        params = json.load(f)
    
    # Select training parameters based on MODEL_TYPE
    model_type = params.get("MODEL_TYPE", "DQN")
    if model_type == "DQN":
        train_params = params["TRAIN"]["DQN"]
    elif model_type == "LSTMDQN":
        train_params = params["TRAIN"]["LSTMDQN"]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Extract training hyperparameters
    num_episodes = train_params["num_episodes"]
    max_steps = train_params["episode_length"]  # Using episode_length as max steps per episode
    batch_size = train_params["batch_size"]
    gamma = train_params["gamma"]
    epsilon_start = train_params["epsilon_start"]
    epsilon_final = train_params["epsilon_final"]
    epsilon_decay = train_params["epsilon_decay"]
    target_update = train_params["target_update"]
    replay_buffer_capacity = train_params["replay_buffer_capacity"]
    learning_rate = train_params["learning_rate"]
    discrete_action_size = train_params["discrete_action_size"]
    seq_len = train_params.get("seq_len", 1)  # Only used for LSTMDQN
    
    # Get start index and episode length from parameters
    start_idx = train_params["start_idx"]
    episode_length = train_params["episode_length"]
    
    # File names for saving model and logs
    model_save_name = train_params.get("model_save_name", "dqn_energy.pth")
    reward_json_name = train_params.get("reward_json_name", "episode_rewards.json")
    training_txt_name = train_params.get("training_txt_name", "training_details.txt")
    
    # Create the environment with defined start index and episode length
    env = EnergyEnv(data_dir="data", start_idx=start_idx, episode_length=episode_length)
    state_dim = env.observation_space.shape[0]
    
    # Define discrete action space based on environment's action bounds
    action_low = env.action_space.low[0]
    action_high = env.action_space.high[0]
    discrete_actions = np.linspace(action_low, action_high, discrete_action_size)
    action_dim = len(discrete_actions)
    
    # Initialize networks based on the model type
    if model_type == "LSTMDQN":
        policy_net = LSTMDQN(state_dim, action_dim).to(device)
        target_net = LSTMDQN(state_dim, action_dim).to(device)
    else:
        policy_net = DQN(state_dim, action_dim).to(device)
        target_net = DQN(state_dim, action_dim).to(device)
    
    # IMPORTANT: Set the policy network to training mode; target network remains in eval mode.
    policy_net.train()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(replay_buffer_capacity)
    
    episode_rewards = []
    training_details = []
    
    # Training loop over episodes
    for episode in tqdm(range(num_episodes), desc="Episodes"):
        state = env.reset()  # state is a vector of shape (state_dim,)
        total_reward = 0
        steps_in_episode = 0
        
        for step in tqdm(range(max_steps), desc="Steps", leave=False):
            # Exponential epsilon decay
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-episode / epsilon_decay)
            
            action_idx = select_action(state, policy_net, epsilon, discrete_actions, model_type, seq_len)
            action = np.array([discrete_actions[action_idx]])
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps_in_episode += 1
            
            if len(replay_buffer) > batch_size:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = replay_buffer.sample(batch_size)
                
                # Convert states to tensors; if using LSTMDQN, reshape to include sequence dimension.
                if model_type == "LSTMDQN":
                    batch_state = torch.FloatTensor(batch_state).view(batch_size, 1, state_dim).to(device)
                    batch_next_state = torch.FloatTensor(batch_next_state).view(batch_size, 1, state_dim).to(device)
                else:
                    batch_state = torch.FloatTensor(batch_state).to(device)
                    batch_next_state = torch.FloatTensor(batch_next_state).to(device)
                
                batch_action = torch.LongTensor(batch_action).unsqueeze(1).to(device)
                batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
                batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)
                
                if model_type == "LSTMDQN":
                    current_q_values, _ = policy_net(batch_state)
                    next_q_values, _ = target_net(batch_next_state)
                else:
                    current_q_values = policy_net(batch_state)
                    next_q_values = target_net(batch_next_state)
                
                # Gather Q-values for selected actions and compute target Q-values
                current_q_values = current_q_values.gather(1, batch_action)
                next_q_values = next_q_values.detach().max(1)[0].unsqueeze(1)
                expected_q_values = batch_reward + gamma * next_q_values * (1 - batch_done)
                
                loss = nn.MSELoss()(current_q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        training_details.append({
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps_in_episode,
            "epsilon": epsilon
        })
        if episode % 10 == 0:
            tqdm.write(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")
        
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    # Save the trained model using the specified file name
    os.makedirs("models", exist_ok=True)
    model_save_path = os.path.join("models", model_save_name)
    torch.save(policy_net.state_dict(), model_save_path)
    print(f"Training complete. Model saved to {model_save_path}")
    
    # Save training details as a text file
    training_txt_path = os.path.join("models", training_txt_name)
    with open(training_txt_path, "w") as txt_file:
        txt_file.write("Training Details:\n")
        for detail in training_details:
            txt_file.write(f"Episode {detail['episode']}: Reward {detail['total_reward']:.2f}, "
                           f"Steps {detail['steps']}, Epsilon {detail['epsilon']:.2f}\n")
    print(f"Training details saved to {training_txt_path}")
    
    # Save episode rewards in a JSON file, including the model name in the JSON
    reward_json_path = os.path.join("models", reward_json_name)
    reward_data = {
        "model": model_save_name,
        "episode_rewards": episode_rewards
    }
    with open(reward_json_path, "w") as json_file:
        json.dump(reward_data, json_file, indent=4)
    print(f"Episode rewards saved to {reward_json_path}")

if __name__ == "__main__":
    train()
