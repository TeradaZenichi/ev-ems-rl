import os
import random
import json
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from opt.env import EnergyEnv
from opt.model import NHMHADDQN

# --- Reproducibility ---
seed = 42
random.seed(seed)
numpy_seed = seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def schedule_epsilon(ep, eps_start, eps_final, eps_decay):
    """
    Exponential epsilon decay schedule.
    """
    return eps_final + (eps_start - eps_final) * np.exp(-ep / eps_decay)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
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


def select_action(state, net, eps, actions):
    """Epsilon-greedy selection with NoisyNet exploration."""
    if random.random() < eps:
        return random.randrange(len(actions))
    tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        q = net(tensor)
    return q.argmax().item()

# --- Build NHMHADDQN model & replay factory ---
def build_models(state_dim, action_dim, cfg):
    """Instantiate policy and target networks plus PER buffer factory."""
    cap = cfg["replay_buffer_capacity"]
    alpha = cfg.get("per_alpha", 0.6)
    policy = NHMHADDQN(
        state_dim,
        action_dim,
        history_len=cfg["history_len"],
        d_model=cfg["hl_size"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["ff_dim"]
    ).to(DEVICE)
    target = NHMHADDQN(
        state_dim,
        action_dim,
        history_len=cfg["history_len"],
        d_model=cfg["hl_size"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["ff_dim"]
    ).to(DEVICE)
    target.load_state_dict(policy.state_dict())
    make_replay = lambda: PrioritizedReplayBuffer(cap, alpha=alpha)
    return policy, target, make_replay

# --- Run one episode ---
def run_episode(env, policy, target, replay, optimizer, cfg, training=True):
    """Execute one full episode, return total reward and cost."""
    obs = env.reset()
    hist = cfg["history_len"]
    # Initialize history buffer from real data or repeat obs
    seq = np.zeros((hist, obs.shape[-1]), dtype=np.float32)
    start_idx = env.current_idx
    if start_idx >= hist - 1:
        orig_idx = env.current_idx
        for h in range(hist):
            idx = start_idx - (hist - 1) + h
            env.current_idx = idx
            flat_obs, _ = env._get_obs()
            seq[h] = flat_obs
        env.current_idx = orig_idx
    else:
        seq = np.repeat(obs[np.newaxis, :], hist, axis=0)
    state = seq

    total_reward = 0.0
    total_cost = 0.0
    policy.train() if training else policy.eval()

    for step in range(cfg["max_steps"]):
        a_idx = select_action(state, policy, cfg["eps"], cfg["actions"])
        action = np.array([cfg["actions"][a_idx]], dtype=np.float32)
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        total_cost += info["cost"]
        next_seq = np.roll(state, -1, axis=0)
        next_seq[-1] = next_obs
        replay.push(state, a_idx, reward, next_seq, done)
        state = next_seq

        if training and len(replay) > cfg["batch_size"]:
            bs, ba, br, bns, bd, is_w, idxs = replay.sample(cfg["batch_size"], beta=cfg.get("per_beta", 0.4))
            bs_t = torch.FloatTensor(bs).to(DEVICE)
            bns_t = torch.FloatTensor(bns).to(DEVICE)
            ba_t = torch.LongTensor(ba).unsqueeze(1).to(DEVICE)
            br_t = torch.FloatTensor(br).unsqueeze(1).to(DEVICE)
            bd_t = torch.FloatTensor(bd).unsqueeze(1).to(DEVICE)
            is_w_t = torch.FloatTensor(is_w).unsqueeze(1).to(DEVICE)

            curr_q = policy(bs_t).gather(1, ba_t)
            with torch.no_grad():
                next_q_policy = policy(bns_t)
                next_a = next_q_policy.argmax(dim=1, keepdim=True)
                next_q_target = target(bns_t).gather(1, next_a)
                target_q = br_t + cfg["gamma"] * next_q_target * (1 - bd_t)

            loss = (curr_q - target_q).pow(2) * is_w_t
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            td_errors = (curr_q - target_q).abs().detach().cpu().numpy().flatten()
            replay.update_priorities(idxs, td_errors)

        if done:
            break

    return total_reward, total_cost

# --- Main Training & CV with Sliding Window ---
def train_and_evaluate():
    # load global and model configs
    gp = json.load(open("data/parameters.json"))
    mc = json.load(open("data/models.json"))
    cfg = mc[gp["MODEL"]["MODEL_TYPE"]]

    ndays = cfg.get("NUM_DAYS", 30)
    episode_length = cfg["episode_length"]

    # dummy env for shape and actions
    env0 = EnergyEnv(data_dir="data",
                     observations=cfg["observations"],
                     start_idx=cfg["start_idx"],
                     episode_length=episode_length)
    state_dim = env0.observation_space.shape[0]
    actions = np.linspace(env0.action_space.low[0], env0.action_space.high[0],
                          cfg["discrete_action_size"])

    # add missing cfg keys
    cfg["actions"] = actions
    cfg["max_steps"] = episode_length
    cfg["eps"] = cfg.get("epsilon_start", 0.0)

    # build models, optimizer, buffer
    policy, target, make_replay = build_models(state_dim, len(actions), cfg)
    optimizer = optim.Adam(policy.parameters(), lr=cfg["learning_rate"])
    tau = cfg.get("tau", 0.005)

    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    cv_results = []
    # sliding window cross-validation
    for fold, start_day in enumerate(range(0, ndays - 2), start=1):
        # reset target network and continuous buffer
        target.load_state_dict(policy.state_dict())
        replay = make_replay()
        train_metrics = []
        val_metrics = []

        # prepare validation environment for day start_day+2
        val_env = EnergyEnv(
            data_dir="data",
            observations=cfg["observations"],
            start_idx=cfg["start_idx"] + (start_day + 2) * episode_length,
            episode_length=episode_length,
            test=True
        )

        # training on two consecutive days with in-loop validation
        for train_day in [start_day, start_day + 1]:
            env = EnergyEnv(
                data_dir="data",
                observations=cfg["observations"],
                start_idx=cfg["start_idx"] + train_day * episode_length,
                episode_length=episode_length
            )
            pbar = tqdm(range(cfg["num_episodes"]), desc=f"Fold {fold} Train Day {train_day+1}")
            for ep in pbar:
                # update epsilon
                cfg["eps"] = schedule_epsilon(
                    ep,
                    cfg.get("epsilon_start", 0.0),
                    cfg.get("epsilon_final", 0.0),
                    cfg.get("epsilon_decay", 1)
                )
                # run training episode
                r_train, c_train = run_episode(env, policy, target, replay, optimizer, cfg, training=True)
                train_metrics.append({"day": train_day+1, "episode": ep+1, "reward": r_train, "cost": c_train})
                # soft update target network
                for p, pt in zip(policy.parameters(), target.parameters()):
                    pt.data.mul_(1 - tau)
                    pt.data.add_(tau * p.data)
                # run validation episode on day+2
                r_val, c_val = run_episode(val_env, policy, target, replay, optimizer, cfg, training=False)
                val_metrics.append({"day": start_day+3, "episode": ep+1, "reward": r_val, "cost": c_val})
                # update progress bar with both metrics
                pbar.set_postfix(
                    train_reward=f"{r_train:.2f}",
                    train_cost=f"{c_train:.2f}",
                    val_reward=f"{r_val:.2f}",
                    val_cost=f"{c_val:.2f}"
                )
            pbar.close()

        # save checkpoint for this fold
        fold_name = cfg["model_save_name"].replace(".pth", f"_fold{fold}.pth")
        torch.save(policy.state_dict(), os.path.join(ckpt_dir, fold_name))

        # collect fold results
        cv_results.append({
            "fold": fold,
            "train": train_metrics,
            "validation": val_metrics
        })
        print(f"Fold {fold} complete: last val reward={r_val:.2f}, val cost={c_val:.2f}")

    # save detailed CV results
    with open(os.path.join(ckpt_dir, "cv_results.json"), 'w') as f:
        json.dump(cv_results, f, indent=2)
    print("Cross-validation complete.")

if __name__ == "__main__":
    train_and_evaluate()
