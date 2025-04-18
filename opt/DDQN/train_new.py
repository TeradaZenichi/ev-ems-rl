import os
import random
import json
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# local imports
from ..env import EnergyEnv
from .model import DDQN, CDDQN, MHADDQN, HMHADDQN

# ---------------------------------------------------------------------------
# Adaptive Replay Buffer (kept isolated so other models can still import the
# classic buffer if they wish)
# ---------------------------------------------------------------------------
class AdaptiveReplayBuffer:
    def __init__(self, capacity: int, history_len: int, state_dim: int):
        self.buffer = deque(maxlen=capacity)
        self.history_len = history_len
        self.state_dim = state_dim

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # helper to build rolling window
    def _window(self, idx: int, offset: int = 0):
        seq = []
        for h in range(self.history_len):
            j = idx - (self.history_len - 1 - h) + offset
            if j < 0 or j >= len(self.buffer):
                seq.append(np.zeros(self.state_dim, dtype=np.float32))
            else:
                elem = self.buffer[j][0] if offset == 0 else self.buffer[j][3]
                seq.append(elem)
        return np.stack(seq, axis=0)

    def sample(self, batch_size):
        idxs = np.random.randint(self.history_len - 1, len(self.buffer), size=batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in idxs:
            s_hist = self._window(idx, offset=0)
            ns_hist = self._window(idx, offset=1)
            _, a, r, _, d = self.buffer[idx]
            states.append(s_hist)
            next_states.append(ns_hist)
            actions.append(a)
            rewards.append(r)
            dones.append(d)
        return (np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones))

    def __len__(self):
        return len(self.buffer)

# ---------------------------------------------------------------------------
# epsilon‑greedy
# ---------------------------------------------------------------------------

def select_action(state_in, net, epsilon, actions, model_type):
    if random.random() < epsilon:
        return random.randrange(len(actions))

    if model_type == "CDDQN":
        main, cond = state_in
        q_values = net(torch.FloatTensor(main).unsqueeze(0).to(device),
                        torch.FloatTensor(cond).unsqueeze(0).to(device))
    elif model_type == "HMHADDQN":
        seq = torch.FloatTensor(state_in).unsqueeze(0).to(device)  # (1,H,D)
        q_values = net(seq)
    else:
        q_values = net(torch.FloatTensor(state_in).unsqueeze(0).to(device))
    return q_values.argmax().item()

# ---------------------------------------------------------------------------
# training entry‑point
# ---------------------------------------------------------------------------

def train():
    with open("data/parameters.json", "r") as f:
        global_params = json.load(f)
    with open("opt/DDQN/models.json", "r") as f:
        model_cfg = json.load(f)

    model_type = global_params["MODEL"]["MODEL_TYPE"].upper()
    cfg = model_cfg[model_type]

    # common hp
    num_episodes   = cfg["num_episodes"]
    max_steps      = cfg["episode_length"]
    batch_size     = cfg["batch_size"]
    gamma          = cfg["gamma"]
    eps_start      = cfg["epsilon_start"]
    eps_final      = cfg["epsilon_final"]
    eps_decay      = cfg["epsilon_decay"]
    target_update  = cfg["target_update"]
    buffer_cap     = cfg["replay_buffer_capacity"]
    lr             = cfg["learning_rate"]
    disc_size      = cfg["discrete_action_size"]
    hl_size        = cfg.get("hl_size", 128)
    start_idx      = cfg["start_idx"]
    ndays          = cfg["NUN_DAYS"]

    env = EnergyEnv(data_dir="data", observations=cfg["observations"],
                    start_idx=start_idx, episode_length=max_steps)
    flat0 = env.reset()
    state_dim = env.observation_space.shape[0]

    a_low, a_high = env.action_space.low[0], env.action_space.high[0]
    actions = np.linspace(a_low, a_high, disc_size)
    action_dim = len(actions)

    # ------------------------------------------------------------------
    # model & buffer creation
    # ------------------------------------------------------------------
    if model_type == "CDDQN":
        main_keys = cfg["main_observations"]
        cond_keys = cfg["conditional_observations"]
        main_dim  = len(main_keys)
        cond_dim  = len(cond_keys)
        policy_net = CDDQN(main_dim, cond_dim, action_dim, hl_size=hl_size).to(device)
        target_net = CDDQN(main_dim, cond_dim, action_dim, hl_size=hl_size).to(device)
        replay = deque(maxlen=buffer_cap)  # simple buffer
    elif model_type == "HMHADDQN":
        history_len = cfg.get("history_len", 4)
        num_heads   = cfg.get("num_heads", 4)
        ff_dim      = cfg.get("ff_dim", 128)
        policy_net  = HMHADDQN(state_dim, action_dim, history_len, hl_size, num_heads, ff_dim).to(device)
        target_net  = HMHADDQN(state_dim, action_dim, history_len, hl_size, num_heads, ff_dim).to(device)
        replay      = AdaptiveReplayBuffer(buffer_cap, history_len, state_dim)
    elif model_type == "MHADDQN":
        num_heads   = cfg.get("num_heads", 4)
        ff_dim      = cfg.get("ff_dim", 128)
        policy_net  = MHADDQN(state_dim, action_dim, hl_size=hl_size, num_heads=num_heads, ff_dim=ff_dim).to(device)
        target_net  = MHADDQN(state_dim, action_dim, hl_size=hl_size, num_heads=num_heads, ff_dim=ff_dim).to(device)
        replay      = deque(maxlen=buffer_cap)
    else:  # DDQN default
        policy_net = DDQN(state_dim, action_dim, hl_size=hl_size).to(device)
        target_net = DDQN(state_dim, action_dim, hl_size=hl_size).to(device)
        replay = deque(maxlen=buffer_cap)

    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # initial sequence for HMHADDQN
    if model_type == "HMHADDQN":
        history_len = cfg.get("history_len", 4)
        state_seq = np.zeros((history_len, state_dim), dtype=np.float32)
        state_seq[-1] = flat0
        state_input = state_seq
    else:
        state_input = flat0

    episode_rewards = []
    
    for day in range(ndays):
        print(f"Training day {day + 1}/{ndays}")

        new_idx = day * 288
        env.new_training_episode(new_idx)
        eps = eps_start

        for ep in tqdm(range(num_episodes), desc="episodes"):
            if model_type == "HMHADDQN":
                state_seq = np.zeros((history_len, state_dim), dtype=np.float32)
                flat = env.reset()
                state_seq[-1] = flat
                state_input = state_seq
            else:
                flat = env.reset()
                state_input = flat
            total_r = 0
            for step in range(max_steps):
                eps = eps_final + (eps_start - eps_final) * np.exp(-ep / eps_decay)
                a_idx = select_action(state_input, policy_net, eps, actions, model_type)
                a = np.array([actions[a_idx]])
                next_flat, r, done, _ = env.step(a)
                total_r += r

                if model_type == "HMHADDQN":
                    next_seq = np.roll(state_seq, -1, axis=0)
                    next_seq[-1] = next_flat
                    replay.push(state_seq, a_idx, r, next_seq, done)
                    state_seq = next_seq
                    state_input = next_seq
                else:
                    replay.append((state_input, a_idx, r, next_flat, done))
                    state_input = next_flat

                # learning
                if len(replay) > batch_size:
                    if model_type == "HMHADDQN":
                        b_s, b_a, b_r, b_ns, b_d = replay.sample(batch_size)
                        b_s = torch.FloatTensor(b_s).to(device)
                        b_ns = torch.FloatTensor(b_ns).to(device)
                    else:
                        batch = random.sample(replay, batch_size)
                        b_s, b_a, b_r, b_ns, b_d = map(np.array, zip(*batch))
                        b_s  = torch.FloatTensor(b_s).to(device)
                        b_ns = torch.FloatTensor(b_ns).to(device)
                    b_a = torch.LongTensor(b_a).unsqueeze(1).to(device)
                    b_r = torch.FloatTensor(b_r).unsqueeze(1).to(device)
                    b_d = torch.FloatTensor(b_d).unsqueeze(1).to(device)

                    if model_type == "CDDQN":
                        raise NotImplementedError  # kept short for brevity
                    else:
                        curr_q = policy_net(b_s).gather(1, b_a)
                        next_q_policy = policy_net(b_ns)
                        next_a = next_q_policy.argmax(dim=1, keepdim=True)
                        next_q = target_net(b_ns).gather(1, next_a)

                    target = b_r + gamma * next_q * (1 - b_d)
                    loss = nn.MSELoss()(curr_q, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if done:
                    break

            episode_rewards.append(total_r)
            if ep % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if ep % 10 == 0:
                tqdm.write(f"ep {ep} reward {total_r:.2f} eps {eps:.2f}")

        # save
        os.makedirs("models_MHADDQN", exist_ok=True)
        torch.save(policy_net.state_dict(), f"models_MHADDQN/model_{day}.pth")

if __name__ == "__main__":
    train()
