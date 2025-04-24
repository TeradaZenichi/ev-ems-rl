import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..env import EnergyEnv
from .model import DDQN, CDDQN, MHADDQN, HMHADDQN, NHMHADDQN

# --- Reproducibility ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1) Prioritized Experience Replay Buffer ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity   = capacity
        self.alpha      = alpha
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        N = len(self.buffer)
        prios = self.priorities[:N]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(N, batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()
        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), weights, indices)

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + 1e-6

    def __len__(self):
        return len(self.buffer)


# --- Utilities ---
def schedule_epsilon(ep, eps_start, eps_final, eps_decay):
    return eps_final + (eps_start - eps_final) * np.exp(-ep / eps_decay)


def select_action(state, net, eps, actions, model_type):
    # if using NoisyNets, eps should be zero and noise provides exploration
    if random.random() < eps:
        return random.randrange(len(actions))
    to_tensor = lambda x: torch.FloatTensor(x).unsqueeze(0).to(DEVICE)
    if model_type == "CDDQN":
        main, cond = state
        q = net(to_tensor(main), to_tensor(cond))
    else:
        q = net(to_tensor(state))
    return q.argmax().item()


# --- Build networks & replay factory ---
def build_models(model_type, state_dim, action_dim, cfg):
    cap   = cfg["replay_buffer_capacity"]
    alpha = cfg.get("per_alpha", 0.6)

    if model_type == "CDDQN":
        policy = CDDQN(len(cfg["main_observations"]),
                       len(cfg["conditional_observations"]),
                       action_dim,
                       hl_size=cfg["hl_size"]).to(DEVICE)
        target = CDDQN(len(cfg["main_observations"]),
                       len(cfg["conditional_observations"]),
                       action_dim,
                       hl_size=cfg["hl_size"]).to(DEVICE)

    elif model_type == "HMHADDQN":
        policy = HMHADDQN(state_dim,
                          action_dim,
                          history_len=cfg["history_len"],
                          d_model=cfg["hl_size"],
                          num_heads=cfg["num_heads"],
                          d_ff=cfg["ff_dim"]).to(DEVICE)
        target = HMHADDQN(state_dim,
                          action_dim,
                          history_len=cfg["history_len"],
                          d_model=cfg["hl_size"],
                          num_heads=cfg["num_heads"],
                          d_ff=cfg["ff_dim"]).to(DEVICE)

    elif model_type == "MHADDQN":
        policy = MHADDQN(state_dim, action_dim,
                         hl_size=cfg["hl_size"],
                         num_heads=cfg["num_heads"],
                         ff_dim=cfg["ff_dim"]).to(DEVICE)
        target = MHADDQN(state_dim, action_dim,
                         hl_size=cfg["hl_size"],
                         num_heads=cfg["num_heads"],
                         ff_dim=cfg["ff_dim"]).to(DEVICE)

    elif model_type == "NHMHADDQN":
        policy = NHMHADDQN(state_dim,
                           action_dim,
                           history_len=cfg["history_len"],
                           d_model=cfg["hl_size"],
                           num_heads=cfg["num_heads"],
                           d_ff=cfg["ff_dim"]).to(DEVICE)
        target = NHMHADDQN(state_dim,
                           action_dim,
                           history_len=cfg["history_len"],
                           d_model=cfg["hl_size"],
                           num_heads=cfg["num_heads"],
                           d_ff=cfg["ff_dim"]).to(DEVICE)

    else:  # DDQN
        policy = DDQN(state_dim, action_dim,
                      hl_size=cfg["hl_size"]).to(DEVICE)
        target = DDQN(state_dim, action_dim,
                      hl_size=cfg["hl_size"]).to(DEVICE)

    target.load_state_dict(policy.state_dict())
    make_replay = lambda: PrioritizedReplayBuffer(cap, alpha=alpha)
    return policy, target, make_replay


# --- Run one episode ---
def run_episode(env, policy, target, replay, optimizer,
                cfg, model_type, training=True):
    state = env.reset()
    # for history‐based models (HMHADDQN & NHMHADDQN)
    if model_type in ("HMHADDQN", "NHMHADDQN"):
        hist = cfg["history_len"]
        seq = np.zeros((hist, state.shape[-1]), dtype=np.float32)
        seq[-1] = state
        state = seq

    total_reward = 0.0
    total_cost   = 0.0
    policy.train() if training else policy.eval()

    for ep in range(cfg["max_steps"]):
        a_idx = select_action(state, policy, cfg["eps"], cfg["actions"], model_type)
        action = np.array([cfg["actions"][a_idx]], dtype=np.float32)

        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        total_cost   += info["cost"]

        # push transition
        if model_type in ("HMHADDQN", "NHMHADDQN"):
            next_seq = np.roll(state, -1, axis=0)
            next_seq[-1] = next_obs
            replay.push(state, a_idx, reward, next_seq, done)
            state = next_seq
        else:
            replay.push(state, a_idx, reward, next_obs, done)
            state = next_obs

        # learning step
        if training and len(replay) > cfg["batch_size"]:
            bs, ba, br, bns, bd, is_w, idxs = replay.sample(
                cfg["batch_size"], beta=cfg.get("per_beta", 0.4)
            )
            bs_t   = torch.FloatTensor(bs).to(DEVICE)
            bns_t  = torch.FloatTensor(bns).to(DEVICE)
            ba_t   = torch.LongTensor(ba).unsqueeze(1).to(DEVICE)
            br_t   = torch.FloatTensor(br).unsqueeze(1).to(DEVICE)
            bd_t   = torch.FloatTensor(bd).unsqueeze(1).to(DEVICE)
            is_w_t = torch.FloatTensor(is_w).unsqueeze(1).to(DEVICE)

            # double Q-learning update
            curr_q = policy(bs_t).gather(1, ba_t)
            with torch.no_grad():
                next_q_policy = policy(bns_t)
                next_a        = next_q_policy.argmax(dim=1, keepdim=True)
                next_q_target = target(bns_t).gather(1, next_a)
                target_q      = br_t + cfg["gamma"] * next_q_target * (1 - bd_t)

            loss = (curr_q - target_q).pow(2) * is_w_t
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            td_errors = (curr_q - target_q).abs().detach().cpu().numpy().flatten()
            replay.update_priorities(idxs, td_errors)

        if done:
            break

    return total_reward, total_cost


# --- Main Training & Evaluation ---
def train_and_evaluate():
    gp     = json.load(open("data/parameters.json"))
    mc_all = json.load(open("opt/D3QN/models.json"))     # ← corrected path
    mtype  = gp["MODEL"]["MODEL_TYPE"].upper()
    cfg    = mc_all[mtype]

    ndays               = cfg.get("NUM_DAYS", 1)
    episode_length      = cfg["episode_length"]
    test_episode_length = cfg.get("test_episode_length", episode_length)

    env0 = EnergyEnv(
        data_dir="data",
        observations=cfg["observations"],
        start_idx=cfg["start_idx"],
        episode_length=episode_length
    )
    state_dim = env0.observation_space.shape[0]
    actions   = np.linspace(
        env0.action_space.low[0],
        env0.action_space.high[0],
        cfg["discrete_action_size"]
    )

    policy, target, make_replay = build_models(mtype, state_dim, len(actions), cfg)
    optimizer = optim.Adam(policy.parameters(), lr=cfg["learning_rate"])

    ckpt_dir = os.path.join("checkpoints", mtype)
    os.makedirs(ckpt_dir, exist_ok=True)

    for day in range(ndays):
        replay = make_replay()
        env0.new_training_episode(cfg["start_idx"] + day * episode_length)

        model_name = cfg["model_save_name"].replace(".pth", f"_day{day+1}.pth")
        model_path = os.path.join(ckpt_dir, model_name)

        train_cfg = {
            "max_steps":  episode_length,
            "batch_size": cfg["batch_size"],
            "gamma":      cfg["gamma"],
            "eps":        cfg["epsilon_start"],
            "actions":    actions,
            "per_beta":   gp.get("PER_BETA", 0.4),
            "history_len": cfg.get("history_len", 1)
        }

        rewards, costs = [], []

        for ep in tqdm(range(cfg["num_episodes"]), desc=f"Day {day+1}/{ndays}"):
            # 2) curriculum-driven epsilon schedule (noop if ε_start=ε_final=0)
            train_cfg["eps"] = schedule_epsilon(
                ep,
                cfg["epsilon_start"],
                cfg["epsilon_final"],
                cfg["epsilon_decay"]
            )
            r, c = run_episode(env0, policy, target, replay,
                               optimizer, train_cfg, mtype, training=True)
            rewards.append(r)
            costs.append(c)

            # 4) target network periodic sync
            if ep % cfg["target_update"] == 0:
                target.load_state_dict(policy.state_dict())

            # 6) noisy‐net: still print & save every 10 eps
            if ep % 10 == 0:
                print(f"Day {day+1} Ep {ep:4d} → Reward: {r:8.2f}, Cost: {c:8.2f}")
                torch.save(policy.state_dict(), model_path)

        total_r, total_c = sum(rewards), sum(costs)
        print(f"=== Day {day+1} Summary → Total Reward = {total_r:.2f}, Total Cost = {total_c:.2f} ===")

        # final save for the day
        torch.save(policy.state_dict(), model_path)
        with open(os.path.join(ckpt_dir, f"metadata_day{day+1}.json"), "w") as jf:
            json.dump({
                "day":          day+1,
                "start_idx":    cfg["start_idx"] + day * episode_length,
                "total_reward": total_r,
                "total_cost":   total_c
            }, jf, indent=2)

    print("Training and evaluation complete.")


if __name__ == "__main__":
    train_and_evaluate()
