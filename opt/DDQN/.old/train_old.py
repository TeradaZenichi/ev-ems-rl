# opt/DDQN/train.py
import os, json, random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..env     import EnergyEnv
from .model    import DDQN, CDDQN, MHADDQN, HMHADDQN      #  ←  NOVO

# ---------- seeds / device ----------
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- adaptive replay ----------
class AdaptiveReplayBuffer:
    def __init__(self, cap, history_len, state_dim):
        self.buf, self.H, self.D = deque(maxlen=cap), history_len, state_dim

    def push(self, s, a, r, s2, d): self.buf.append((s, a, r, s2, d))

    def _seq(self, i, nxt=False):
        seq=[]
        offs = 3 if nxt else 0
        for h in range(self.H):
            j = i - (self.H-1-h) + (1 if nxt else 0)
            if 0<=j<len(self.buf): seq.append(self.buf[j][offs])
            else:                  seq.append(np.zeros(self.D, np.float32))
        return np.stack(seq,0)                # (H,D)

    def sample(self, n):
        idx = np.random.randint(self.H-1, len(self.buf), n)
        S,A,R,S2,D = [],[],[],[],[]
        for i in idx:
            s_hist, s2_hist = self._seq(i), self._seq(i,True)
            _,a,r,_,d = self.buf[i]
            S.append(s_hist); S2.append(s2_hist)
            A.append(a);     R.append(r);     D.append(d)
        return map(np.array,(S,A,R,S2,D))

    def __len__(self): return len(self.buf)

# ---------- ação ε‑greedy ----------
def select_action(state, net, eps, A, typ):
    if random.random() < eps: return random.randrange(len(A))

    if typ=="CDDQN":
        s_m,s_c = state
        q=net(torch.FloatTensor(s_m).unsqueeze(0).to(device),
              torch.FloatTensor(s_c).unsqueeze(0).to(device))
    elif typ=="HMHADDQN":
        q=net(torch.FloatTensor(state).unsqueeze(0).to(device))        # (1,H,D)
    else:                               # DDQN / MHADDQN
        last = state[-1] if state.ndim==2 else state                  # (D,)
        q=net(torch.FloatTensor(last).unsqueeze(0).to(device))        # (1,D)
    return q.argmax().item()

# ---------- treinamento ----------
def train():
    cfg   = json.load(open("data/parameters.json"))
    all_p = json.load(open("opt/DDQN/models.json"))
    typ   = cfg["MODEL"]["MODEL_TYPE"].upper()
    p     = all_p[typ]

    env = EnergyEnv("data", observations=p["observations"],
                    start_idx=p["start_idx"], episode_length=p["episode_length"])
    state_dim  = env.observation_space.shape[0]
    actions    = np.linspace(env.action_space.low[0], env.action_space.high[0],
                             p["discrete_action_size"])
    act_dim    = len(actions)

    # -------- modelos --------
    if typ=="CDDQN":
        main_k, cond_k = p["main_observations"], p["conditional_observations"]
        main_dim, cond_dim = len(main_k), len(cond_k)
        net = lambda: CDDQN(main_dim, cond_dim, act_dim,
                            hl_size=p.get("hl_size",128),
                            hl_number=p.get("hl_number",3),
                            dropout_rate=p.get("dropout_rate",0.))
    elif typ=="MHADDQN":
        net = lambda: MHADDQN(state_dim, act_dim,
                              hl_size=p.get("hl_size",128),
                              num_heads=p.get("num_heads",4),
                              ff_dim=p.get("ff_dim",128))
    elif typ=="HMHADDQN":
        hist = p.get("history_len",4)
        net = lambda: HMHADDQN(state_dim, act_dim, history_len=hist,
                               hl_size=p.get("hl_size",128),
                               num_heads=p.get("num_heads",4),
                               ff_dim=p.get("ff_dim",128))
    else:
        net = lambda: DDQN(state_dim, act_dim,
                           hl_size=p.get("hl_size",128),
                           hl_number=p.get("hl_number",3))

    policy, target = net().to(device), net().to(device)
    target.load_state_dict(policy.state_dict())
    opt = optim.Adam(policy.parameters(), lr=p["learning_rate"])

    hist_len = p.get("history_len",1) if typ=="HMHADDQN" else 1
    buffer   = AdaptiveReplayBuffer(p["replay_buffer_capacity"], hist_len, state_dim)

    epi_rewards=[]
    for epi in tqdm(range(p["num_episodes"]), desc="episodes"):
        s = env.reset()
        if typ=="CDDQN":
            obs={k:s[i] for i,k in enumerate(p["observations"])}
            state=(np.array([obs[k] for k in main_k]),
                   np.array([obs[k] for k in cond_k]))
        else:
            state=s

        window = deque(maxlen=hist_len)           # para HMHADDQN
        for _ in range(hist_len-1): window.append(np.zeros(state_dim))
        window.append(state if state_dim else s)

        total=0
        for step in range(p["episode_length"]):
            eps = p["epsilon_final"] + (p["epsilon_start"]-p["epsilon_final"])*np.exp(-epi/p["epsilon_decay"])

            # entrada para política
            if typ=="HMHADDQN": pol_state = np.stack(window,0)
            else:               pol_state = state

            a_idx = select_action(pol_state, policy, eps, actions, typ)
            a_val = np.array([actions[a_idx]])

            s2,r,done,_=env.step(a_val); total+=r

            buffer.push(pol_state, a_idx, r, s2, done)

            # monta próximo estado
            if typ=="CDDQN":
                obs2={k:s2[i] for i,k in enumerate(p["observations"])}
                state=(np.array([obs2[k] for k in main_k]),
                       np.array([obs2[k] for k in cond_k]))
            else:
                state=s2
            window.append(state if typ!="HMHADDQN" else s2)

            # --------  treino  --------
            if len(buffer)>p["batch_size"]:
                bs,ba,br,bs2,bd = buffer.sample(p["batch_size"])
                ba=torch.LongTensor(ba).unsqueeze(1).to(device)
                br=torch.FloatTensor(br).unsqueeze(1).to(device)
                bd=torch.FloatTensor(bd).unsqueeze(1).to(device)

                if typ=="CDDQN":
                    idx_m=[p["observations"].index(k) for k in main_k]
                    idx_c=[p["observations"].index(k) for k in cond_k]
                    m  = torch.FloatTensor(bs [:,-1,idx_m]).to(device)
                    c  = torch.FloatTensor(bs [:,-1,idx_c]).to(device)
                    m2 = torch.FloatTensor(bs2[:,-1,idx_m]).to(device)
                    c2 = torch.FloatTensor(bs2[:,-1,idx_c]).to(device)
                    q     = policy(m ,c ).gather(1,ba)
                    a_st  = policy(m2,c2).argmax(1,keepdim=True)
                    q_next= target(m2,c2).gather(1,a_st)
                elif typ=="HMHADDQN":
                    s_t  = torch.FloatTensor(bs ).to(device)
                    s2_t = torch.FloatTensor(bs2).to(device)
                    q     = policy(s_t ).gather(1,ba)
                    a_st  = policy(s2_t).argmax(1,keepdim=True)
                    q_next= target(s2_t).gather(1,a_st)
                else:  # DDQN / MHADDQN
                    s_t  = torch.FloatTensor(bs [:,-1,:]).to(device)
                    s2_t = torch.FloatTensor(bs2[:,-1,:]).to(device)
                    q     = policy(s_t ).gather(1,ba)
                    a_st  = policy(s2_t).argmax(1,keepdim=True)
                    q_next= target(s2_t).gather(1,a_st)

                y = br + p["gamma"]*q_next*(1-bd)
                loss = nn.MSELoss()(q,y.detach())
                opt.zero_grad(); loss.backward(); opt.step()

            if done: break

        epi_rewards.append(total)
        if epi%p["target_update"]==0: target.load_state_dict(policy.state_dict())
        if epi%10==0: tqdm.write(f"epi {epi}  reward {total:.1f}  ε {eps:.2f}")

    os.makedirs("models",exist_ok=True)
    torch.save(policy.state_dict(),f"models/{p['model_save_name']}")
    json.dump({"rewards":epi_rewards},open(f"models/{p['reward_json_name']}","w"),indent=2)
    print("Treinamento concluído.")

if __name__=="__main__":
    if json.load(open("data/parameters.json"))["ENV"]["MODE"].upper()=="TRAIN":
        train()
