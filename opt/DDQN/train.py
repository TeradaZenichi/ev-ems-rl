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
# Adaptive Replay Buffer (mantida SEM MUDANÇAS)
# ---------------------------------------------------------------------------
class AdaptiveReplayBuffer:
    def __init__(self, capacity: int, history_len: int, state_dim: int):
        self.buffer = deque(maxlen=capacity)
        self.history_len = history_len
        self.state_dim = state_dim

        # será definido ao receber o primeiro push
        self.elem_shape = None

    def push(self, state, action, reward, next_state, done):
        # na primeira inserção, armazena a forma do elemento (pode ser 1D ou nD)
        if self.elem_shape is None:
            # state deve ser um numpy array
            try:
                self.elem_shape = state.shape
            except AttributeError:
                # fallback para vetor de tamanho state_dim
                self.elem_shape = (self.state_dim,)
        self.buffer.append((state, action, reward, next_state, done))

    def _window(self, idx: int, offset: int = 0):
        seq = []
        for h in range(self.history_len):
            j = idx - (self.history_len - 1 - h) + offset
            if j < 0 or j >= len(self.buffer):
                # padding com a mesma forma que os elementos reais
                pad = np.zeros(self.elem_shape, dtype=np.float32)
                seq.append(pad)
            else:
                # pegamos o estado (offset=0) ou next_state (offset=1)
                elem = self.buffer[j][0] if offset == 0 else self.buffer[j][3]
                seq.append(elem)
        # agora todos os arrays em seq têm a mesma shape
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
# Funções utilitárias
# ---------------------------------------------------------------------------
def select_action(state_in, net, epsilon, actions, model_type):
    if random.random() < epsilon:
        return random.randrange(len(actions))
    tensor = lambda x: torch.FloatTensor(x).unsqueeze(0).to(device)
    if model_type == "CDDQN":
        main, cond = state_in
        q = net(tensor(main), tensor(cond))
    else:
        q = net(tensor(state_in))
    return q.argmax().item()


def schedule_epsilon(global_ep, eps_start, eps_final, eps_decay):
    return eps_final + (eps_start - eps_final) * np.exp(-global_ep / eps_decay)


def build_networks(model_type, state_dim, action_dim, cfg):
    cap = cfg["replay_buffer_capacity"]
    hl = cfg.get("hl_size", 128)
    if model_type == "CDDQN":
        md = len(cfg["main_observations"])
        cd = len(cfg["conditional_observations"])
        policy = CDDQN(md, cd, action_dim, hl_size=hl).to(device)
        target = CDDQN(md, cd, action_dim, hl_size=hl).to(device)
        make_replay = lambda: deque(maxlen=cap)
    elif model_type == "HMHADDQN":
        hist = cfg.get("history_len", 4)
        nh = cfg.get("num_heads", 4)
        ff = cfg.get("ff_dim", 128)
        policy = HMHADDQN(state_dim, action_dim, hist, hl, nh, ff).to(device)
        target = HMHADDQN(state_dim, action_dim, hist, hl, nh, ff).to(device)
        make_replay = lambda: AdaptiveReplayBuffer(cap, hist, state_dim)
    elif model_type == "MHADDQN":
        nh = cfg.get("num_heads", 4)
        ff = cfg.get("ff_dim", 128)
        policy = MHADDQN(state_dim, action_dim, hl_size=hl, num_heads=nh, ff_dim=ff).to(device)
        target = MHADDQN(state_dim, action_dim, hl_size=hl, num_heads=nh, ff_dim=ff).to(device)
        make_replay = lambda: deque(maxlen=cap)
    else:
        policy = DDQN(state_dim, action_dim, hl_size=hl).to(device)
        target = DDQN(state_dim, action_dim, hl_size=hl).to(device)
        make_replay = lambda: deque(maxlen=cap)
    target.load_state_dict(policy.state_dict())
    return policy, target, make_replay


def train_episode(env, policy, target, replay, optimizer,
                  batch_size, gamma, actions, model_type, eps, max_steps):
    """
    Executes one episode and returns total reward.
    For HMHADDQN, the buffer stores only the most recent state vector,
    while the full history window is maintained in `state_seq`.
    """
    # reset environment and get initial flattened state
    flat0 = env.reset()
    if model_type == "HMHADDQN":
        # initialize empty history sequence and place the first state
        hist = replay.history_len
        state_seq = np.zeros((hist, env.observation_space.shape[0]), dtype=np.float32)
        state_seq[-1] = flat0
        state_input = state_seq
    else:
        # for other models, state_input is just the flat state
        state_input = flat0

    total_r = 0.0
    policy.train()
    target.eval()

    for _ in range(max_steps):
        # select action using epsilon-greedy
        a_idx = select_action(state_input, policy, eps, actions, model_type)
        action = np.array([actions[a_idx]], dtype=np.float32)

        # perform action in environment
        next_flat, r, done, _ = env.step(action)
        total_r += r

        if model_type == "HMHADDQN":
            # roll history window and append new state
            next_seq = np.roll(state_seq, -1, axis=0)
            next_seq[-1] = next_flat

            # push only the latest state vector into the buffer
            replay.push(state_seq[-1], a_idx, r, next_seq[-1], done)

            # update sequence and input for next step
            state_seq = next_seq
            state_input = next_seq
        else:
            # push transition for non-HMHADDQN models
            replay.append((state_input, a_idx, r, next_flat, done))
            state_input = next_flat

        # learning step if buffer has enough samples
        if len(replay) > batch_size:
            if model_type == "HMHADDQN":
                b_s, b_a, b_r, b_ns, b_d = replay.sample(batch_size)
                b_s  = torch.FloatTensor(b_s).to(device)
                b_ns = torch.FloatTensor(b_ns).to(device)
            else:
                batch = random.sample(replay, batch_size)
                b_s, b_a, b_r, b_ns, b_d = map(np.array, zip(*batch))
                b_s  = torch.FloatTensor(b_s).to(device)
                b_ns = torch.FloatTensor(b_ns).to(device)

            b_a = torch.LongTensor(b_a).unsqueeze(1).to(device)
            b_r = torch.FloatTensor(b_r).unsqueeze(1).to(device)
            b_d = torch.FloatTensor(b_d).unsqueeze(1).to(device)

            # compute current Q values
            curr_q = policy(b_s).gather(1, b_a)
            with torch.no_grad():
                # double Q-learning target
                next_q_policy = policy(b_ns)
                next_a = next_q_policy.argmax(dim=1, keepdim=True)
                next_q = target(b_ns).gather(1, next_a)
                target_q = b_r + gamma * next_q * (1 - b_d)

            # backpropagate loss
            loss = nn.MSELoss()(curr_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    return total_r



def train_and_evaluate():
    # load configs
    with open("data/parameters.json") as f:
        gp = json.load(f)
    with open("opt/DDQN/models.json") as f:
        mc = json.load(f)
    model_type = gp["MODEL"]["MODEL_TYPE"].upper()
    ndays = mc.get("NUM_DAYS", 30)
    cfg = mc[model_type]

    # hyperparams
    num_eps, max_steps = cfg["num_episodes"], cfg["episode_length"]
    batch_size, gamma = cfg["batch_size"], cfg["gamma"]
    eps_start, eps_final, eps_decay = cfg["epsilon_start"], cfg["epsilon_final"], cfg["epsilon_decay"]
    tgt_upd, start0 = cfg["target_update"], cfg["start_idx"]
    max_steps_test = cfg["test_episode_length"]

    #test parameters
    max_steps_test = cfg["test_episode_length"]

    # build initial env for dims
    env0 = EnergyEnv(data_dir="data", start_idx=start0,
                     episode_length=max_steps, test=False,
                     observations=cfg["observations"])
    obs_dim = env0.observation_space.shape[0]
    a_low, a_high = env0.action_space.low[0], env0.action_space.high[0]
    actions = np.linspace(a_low, a_high, cfg["discrete_action_size"])

    # build nets
    policy, target, make_replay = build_networks(model_type, obs_dim, len(actions), cfg)
    optimizer = optim.Adam(policy.parameters(), lr=cfg["learning_rate"])

    ckpt_dir = os.path.join("checkpoints", model_type)
    os.makedirs(ckpt_dir, exist_ok=True)

    for day in range(ndays):
        # TRAIN
        replay = make_replay()
        # s_idx = start0 + day * max_steps
        s_idx = start0
        env0.new_training_episode(s_idx)
        #reset epsilon
        eps = eps_start
        rewards = []
        for ep in tqdm(range(num_eps), desc=f"Day {day+1}/{ndays}"):
            # glob_ep = day * num_eps + ep
            glob_ep = ep
            eps = schedule_epsilon(glob_ep, eps_start, eps_final, eps_decay)
            r = train_episode(env0, policy, target, replay,
                              optimizer, batch_size, gamma,
                              actions, model_type, eps, max_steps)
            rewards.append(r)
            if ep % tgt_upd == 0:
                target.load_state_dict(policy.state_dict())
            if ep % 10 == 0:
                print(f"Episode {ep}: Total Reward: {r:.2f}, Epsilon: {eps:.2f}")
        total_r = float(sum(rewards))
        # save model + metadata
        model_path = os.path.join(ckpt_dir, f"{model_type}_day{day+1}.pth")
        torch.save(policy.state_dict(), model_path)
        meta = {"day": day+1, "start_idx": s_idx, "total_reward": total_r, "tariff": env0.cost_dict}
        with open(os.path.join(ckpt_dir, f"metadata_day{day+1}.json"), "w") as jf:
            json.dump(meta, jf, indent=2)

        # EVALUATION on test
        env_test = EnergyEnv(data_dir="data", start_idx=s_idx,
                             episode_length=max_steps_test, test=True,
                             observations=cfg["observations"])
    
        policy.eval()
        total_cost = 0.0
        state = env_test.reset()
        if model_type == "HMHADDQN":
            hist = cfg.get("history_len", 4)
            seq = np.zeros((hist, state.shape[-1]), dtype=np.float32)
            seq[-1] = state; state = seq
        for _ in range(max_steps_test):
            a_idx = select_action(state, policy, 0.0, actions, model_type)
            action = np.array([actions[a_idx]], dtype=np.float32)
            _, _, done, info = env_test.step(action)
            total_cost += info["cost"]
            nxt_obs = env_test._get_obs()[0]
            if model_type == "HMHADDQN":
                seq = np.roll(state, -1, axis=0); seq[-1] = nxt_obs; state = seq
            else:
                state = nxt_obs
            if done:
                break
        eval_meta = {"day": day+1, "test_start_idx": s_idx, "total_test_cost": float(total_cost)}
        with open(os.path.join(ckpt_dir, f"eval_day{day+1}.json"), "w") as jf:
            json.dump(eval_meta, jf, indent=2)

    print("Training and evaluation complete.")


if __name__ == "__main__":
    train_and_evaluate()
