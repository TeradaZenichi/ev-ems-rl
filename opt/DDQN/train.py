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

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

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
# Funções utilitárias
# ---------------------------------------------------------------------------
def select_action(state_in, net, epsilon, actions, model_type):
    if random.random() < epsilon:
        return random.randrange(len(actions))

    if model_type == "CDDQN":
        main, cond = state_in
        q_values = net(
            torch.FloatTensor(main).unsqueeze(0).to(device),
            torch.FloatTensor(cond).unsqueeze(0).to(device)
        )
    elif model_type == "HMHADDQN":
        seq = torch.FloatTensor(state_in).unsqueeze(0).to(device)
        q_values = net(seq)
    else:
        q_values = net(torch.FloatTensor(state_in).unsqueeze(0).to(device))

    return q_values.argmax().item()


def schedule_epsilon(global_ep, eps_start, eps_final, eps_decay):
    """Decaimento exponencial de epsilon."""
    return eps_final + (eps_start - eps_final) * np.exp(-global_ep / eps_decay)


def build_networks(model_type, state_dim, action_dim, cfg):
    """
    Cria policy_net, target_net e retorna fábrica de replay buffer.
    """
    cap = cfg["replay_buffer_capacity"]

    if model_type == "CDDQN":
        main_dim = len(cfg["main_observations"])
        cond_dim = len(cfg["conditional_observations"])
        policy_net = CDDQN(main_dim, cond_dim, action_dim, hl_size=cfg.get("hl_size",128)).to(device)
        target_net = CDDQN(main_dim, cond_dim, action_dim, hl_size=cfg.get("hl_size",128)).to(device)
        make_replay = lambda: deque(maxlen=cap)

    elif model_type == "HMHADDQN":
        hist = cfg.get("history_len",4)
        heads = cfg.get("num_heads",4)
        ff   = cfg.get("ff_dim",128)
        policy_net = HMHADDQN(state_dim, action_dim, hist, cfg.get("hl_size",128), heads, ff).to(device)
        target_net = HMHADDQN(state_dim, action_dim, hist, cfg.get("hl_size",128), heads, ff).to(device)
        make_replay = lambda: AdaptiveReplayBuffer(cap, hist, state_dim)

    elif model_type == "MHADDQN":
        heads = cfg.get("num_heads",4)
        ff    = cfg.get("ff_dim",128)
        policy_net = MHADDQN(state_dim, action_dim, hl_size=cfg.get("hl_size",128),
                             num_heads=heads, ff_dim=ff).to(device)
        target_net = MHADDQN(state_dim, action_dim, hl_size=cfg.get("hl_size",128),
                             num_heads=heads, ff_dim=ff).to(device)
        make_replay = lambda: deque(maxlen=cap)

    else:  # DDQN
        policy_net = DDQN(state_dim, action_dim, hl_size=cfg.get("hl_size",128)).to(device)
        target_net = DDQN(state_dim, action_dim, hl_size=cfg.get("hl_size",128)).to(device)
        make_replay = lambda: deque(maxlen=cap)

    target_net.load_state_dict(policy_net.state_dict())
    return policy_net, target_net, make_replay


def train_episode(env, policy_net, target_net, replay, optimizer,
                  batch_size, gamma, actions, model_type, eps, max_steps):
    """
    Executa um episódio completo e retorna a recompensa total.
    """
    state = env.reset()
    if model_type == "HMHADDQN":
        hist = replay.history_len
        seq = np.zeros((hist, state.shape[-1]), dtype=np.float32)
        seq[-1] = state
        state = seq

    total_r = 0.0
    policy_net.train()
    target_net.eval()

    for _ in range(max_steps):
        a_idx = select_action(state, policy_net, eps, actions, model_type)
        action = np.array([actions[a_idx]], dtype=np.float32)

        next_flat, r, done, _ = env.step(action)
        total_r += r

        if model_type == "HMHADDQN":
            next_seq = np.roll(state, -1, axis=0)
            next_seq[-1] = next_flat
            replay.push(state, a_idx, r, next_seq, done)
            state = next_seq
        else:
            replay.append((state, a_idx, r, next_flat, done))
            state = next_flat

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

            curr_q = policy_net(b_s).gather(1, b_a)
            with torch.no_grad():
                next_q_policy = policy_net(b_ns)
                next_a = next_q_policy.argmax(dim=1, keepdim=True)
                next_q = target_net(b_ns).gather(1, next_a)
                target_q = b_r + gamma * next_q * (1 - b_d)

            loss = nn.MSELoss()(curr_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    return total_r


# ---------------------------------------------------------------------------
# Função principal de treinamento
# ---------------------------------------------------------------------------
def train():
    # carrega parâmetros
    with open("data/parameters.json", "r") as f:
        gp = json.load(f)
    with open("opt/DDQN/models.json", "r") as f:
        mc = json.load(f)

    model_type = gp["MODEL"]["MODEL_TYPE"].upper()
    ndays      = gp.get("NUM_DAYS", 1)
    cfg        = mc[model_type]

    num_eps    = cfg["num_episodes"]
    max_steps  = cfg["episode_length"]
    batch_size = cfg["batch_size"]
    gamma      = cfg["gamma"]
    eps_start  = cfg["epsilon_start"]
    eps_final  = cfg["epsilon_final"]
    eps_decay  = cfg["epsilon_decay"]
    tgt_upd    = cfg["target_update"]
    start_idx0 = cfg["start_idx"]

    # ambiente para dims e tarifas
    env = EnergyEnv(data_dir="data",
                    observations=cfg["observations"],
                    start_idx=start_idx0,
                    episode_length=max_steps)
    obs_dim     = env.observation_space.shape[0]
    a_low, a_high = env.action_space.low[0], env.action_space.high[0]
    actions     = np.linspace(a_low, a_high, cfg["discrete_action_size"])

    # redes e buffer factory
    policy_net, target_net, make_replay = build_networks(model_type, obs_dim, len(actions), cfg)
    optimizer = optim.Adam(policy_net.parameters(), lr=cfg["learning_rate"])

    # prepara diretório de checkpoints
    ckpt_dir = os.path.join("checkpoints", model_type)
    os.makedirs(ckpt_dir, exist_ok=True)

    # loop de dias
    for day in range(ndays):
        replay = make_replay()
        start_idx = start_idx0 + day * max_steps
        env.new_training_episode(start_idx)

        total_rewards = []
        for ep in tqdm(range(num_eps), desc=f"Day {day+1}/{ndays}"):
            glob_ep = day * num_eps + ep
            eps = schedule_epsilon(glob_ep, eps_start, eps_final, eps_decay)

            r = train_episode(env, policy_net, target_net, replay, optimizer,
                              batch_size, gamma, actions, model_type, eps, max_steps)
            total_rewards.append(r)

            if ep % tgt_upd == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if ep % 10 == 0:
                tqdm.write(f"[Day {day+1}] Ep {ep} | R={r:.2f} | eps={eps:.3f}")

        # soma total do dia
        total_reward_day = float(sum(total_rewards))

        # salva checkpoint do modelo
        model_path = os.path.join(ckpt_dir, f"{model_type}_day{day+1}.pth")
        torch.save(policy_net.state_dict(), model_path)
        print(f"✅ Modelo salvo: {model_path}")

        # salva metadados: total_reward e tarifa
        meta = {
            "day": day + 1,
            "start_idx": start_idx,
            "total_reward": total_reward_day,
            "tariff": env.cost_dict
        }
        meta_path = os.path.join(ckpt_dir, f"metadata_day{day+1}.json")
        with open(meta_path, "w") as jf:
            json.dump(meta, jf, indent=2)
        print(f"✅ Metadados salvos: {meta_path}")

    print("Treinamento concluído.")


if __name__ == "__main__":
    train()
