import gym
import numpy as np
import pandas as pd
import json
import os
from gym import spaces

class EnergyEnv(gym.Env):
    def __init__(self,
                 data_dir='data',
                 timestep_min=5,
                 start_idx=0,
                 episode_length=288,
                 test=False,
                 observations=None,
                 data=None):
        super(EnergyEnv, self).__init__()

        # carrega parâmetros gerais
        with open(os.path.join(data_dir, 'parameters.json'), 'r') as f:
            self.params = json.load(f)

        self.Pnom = self.params.get("Pnom", 1)
        self.bonus = self.params["RL"].get("Bonus", 0)

        self.obs_keys = observations if observations else ["balance_ratio", "soc_ratio", "hour_sin"]
        self.bess_params = self.params["BESS"]
        self.initial_soc = self.bess_params["SoC0"]
        self.Emax = self.bess_params["Emax"]
        self.Pmax_charge = self.bess_params["Pmax_c"]
        self.Pmax_discharge = self.bess_params["Pmax_d"]
        self.eff = self.bess_params["eff"]
        self.dt = self.params["timestep"] / 60
        self.PEDS_max = self.params["EDS"]["Pmax"]
        self.PEDS_min = self.params["EDS"]["Pmin"]
        self.PVmax = self.params["PV"]["Pmax"]
        self.Loadmax = self.params["Load"]["Pmax"]
        self.cost_dict = self.params["EDS"]["cost"]
        self.difficulty = float(self.params["ENV"]["difficulty"])

        # modo de operação definido em __init__
        self.test_mode = test

        # leitura condicional dos dados de PV e carga
        mode = 'test' if self.test_mode else 'train'
        pv_file   = f'pv_5min_{mode}.csv'
        load_file = f'load_5min_{mode}.csv'

        if data is not None:
            pv_file   = f'pv_5min_{data}.csv'
            load_file = f'load_5min_{data}.csv'

        self.pv_data = pd.read_csv(
            os.path.join(data_dir, pv_file),
            index_col='timestamp',
            parse_dates=['timestamp']
        )
        self.load_data = pd.read_csv(
            os.path.join(data_dir, load_file),
            index_col='timestamp',
            parse_dates=['timestamp']
        )

        self.pv_series = self.pv_data['p_norm']
        self.load_series = self.load_data['p_norm']


        self.start_idx = start_idx
        self.episode_length = episode_length
        self.current_idx = self.start_idx
        self.end_idx = self.start_idx + self.episode_length

        self.action_space = spaces.Box(
            low=-self.Pmax_discharge,
            high=self.Pmax_charge,
            shape=(1,),
            dtype=np.float32
        )

        self.soc = self.initial_soc
        self.done = False

        flat_obs, _ = self._get_obs()
        obs_dim = flat_obs.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.reset()

    def new_training_episode(self, start_idx):
        """
        Reinicia índice e SoC para novo episódio de treinamento.
        """
        self.start_idx = start_idx
        self.current_idx = self.start_idx
        self.end_idx = self.start_idx + self.episode_length
        self.soc = self.initial_soc
        self.done = False
        self.difficulty = float(self.params["ENV"]["difficulty"])
        self.episode_counter = 0
        return self.reset()

    def reset(self):
        """
        Reseta o ambiente:
        - Se self.test_mode for True, mantém SoC e índices fixos (modo TEST).
        - Caso contrário, aplica randomização e currículo (modo TRAIN).
        """
        if self.test_mode:
            # TEST mode: mantém SoC e índices fixos
            self.soc = self.initial_soc
            self.current_idx = self.start_idx
            self.end_idx = self.start_idx + self.episode_length
            self.done = False

        else:
            # TRAIN mode: randomização e currículo
            if not hasattr(self, "episode_counter"):
                self.episode_counter = 0
            self.episode_counter += 1

            # currículo
            if (str(self.params["ENV"]["curriculum"]).upper() == "TRUE"
                and self.episode_counter % int(self.params["ENV"]["curriculum_steps"]) == 0):
                inc = float(self.params["ENV"]["curriculum_increment"])
                mx  = float(self.params["ENV"]["curriculum_max"])
                self.difficulty = min(self.difficulty + inc, mx)

            rand_obs = self.params["ENV"].get("randomize_observations", {})

            # randomização SoC
            if (rand_obs.get("soc","False").upper()=="TRUE"
                and str(self.params["ENV"]["randomize"]).upper()=="TRUE"):
                soc_range = 0.05 + self.difficulty * 0.95
                low = max(0, 0.5 - soc_range/2)
                high = min(1, 0.5 + soc_range/2)
                self.soc = np.random.uniform(low, high)
            else:
                self.soc = self.initial_soc

            # randomização EDS
            if (rand_obs.get("eds","False").upper()=="TRUE"
                and str(self.params["ENV"]["randomize"]).upper()=="TRUE"):
                scale = 0.05 + self.difficulty
                factor = 1 + np.random.uniform(-scale, scale)
                self.PEDS_max = max(0, self.params["EDS"]["Pmax"] * factor)
                self.PEDS_min = max(0, self.params["EDS"]["Pmin"] * factor)
            else:
                self.PEDS_max = self.params["EDS"]["Pmax"]
                self.PEDS_min = self.params["EDS"]["Pmin"]

            # randomização de índice
            if (rand_obs.get("idx","False").upper()=="TRUE"
                and str(self.params["ENV"]["randomize"]).upper()=="TRUE"):
                limit = int((0.2 + 0.6*self.difficulty) * 0.1 * len(self.pv_series))
                self.start_idx = np.random.randint(0, max(1, limit - self.episode_length))

            self.current_idx = self.start_idx
            self.end_idx = self.start_idx + self.episode_length
            self.done = False

            #print after 10 episodes
            if self.episode_counter % 10 == 0:
                print(f"Episode: {self.episode_counter}, Difficulty: {self.difficulty:.2f}, "
                      f"SoC: {self.soc:.2f}, EDS: [{self.PEDS_min:.2f}, {self.PEDS_max:.2f}]")

        flat_obs, _ = self._get_obs()
        return flat_obs

    def _update_soc(self, p_bess):
        delta = (p_bess * self.eff * self.dt) / self.Emax if p_bess >= 0 else (p_bess / self.eff * self.dt) / self.Emax
        penalty = 0.0
        self.soc += delta
        if self.soc > 1.0:
            overflow = self.soc - 1.0
            penalty += overflow * abs(p_bess) * self.params["RL"].get("bess_penalty", 10.0)
            self.soc = 1.0
        if self.soc < 0.0:
            under = -self.soc
            penalty += under * abs(p_bess) * self.params["RL"].get("bess_penalty", 10.0)
            self.soc = 0.0
        return penalty * self.dt

    def _compute_cost(self, p_grid, hour_str):
        tarif = self.cost_dict.get(hour_str, 0.4)
        cost = (p_grid * self.dt if p_grid > 0 else 0) * tarif
        return cost, tarif

    def _compute_penalty(self, p_grid):
        pen = self.params["RL"]["Penalty"]
        if p_grid > self.PEDS_max:
            return pen * (p_grid - self.PEDS_max) * self.dt
        if p_grid < -self.PEDS_min:
            return pen * (-self.PEDS_min - p_grid) * self.dt
        return 0.0

    def _compute_alignment_bonus(self, p_bess, p_pv, p_load):
        expected = max(0, p_load - p_pv)
        mis = 0
        if p_bess < 0:
            excess = abs(p_bess) - expected
            mis = max(0, excess)
        return -self.params["RL"].get("misalignment_penalty", 0.0) * mis * self.dt
    
    def _charge_bonus(self, p_bess, p_pv, p_load):
        """
        Positive bonus proportional to how much PV surplus is actually used for charging.
        """
        if p_bess <= 0:
            return 0.0
        surplus_pv = max(0.0, p_pv - p_load)
        used_pv    = min(p_bess, surplus_pv)
        rate       = self.params["RL"].get("charge_bonus", 1.0)
        return rate * used_pv * self.dt

    def _match_penalty(self, p_bess, p_pv, p_load):
        """
        Negative penalty proportional to the absolute error between actual discharge
        and ideal discharge (load minus PV).
        """
        if p_bess >= 0:
            return 0.0
        expected_discharge = max(0.0, p_load - p_pv)
        actual_discharge   = abs(p_bess)
        mismatch           = abs(actual_discharge - expected_discharge)
        rate               = self.params["RL"].get("match_penalty", 1.0)
        return - rate * mismatch * self.dt

    def step(self, action):
        # Clip BESS power to allowed range
        p_bess = np.clip(action[0], -self.Pmax_discharge, self.Pmax_charge)
        if p_bess > 0:
            # Limit charging by remaining capacity
            max_charge = ((1 - self.soc) * self.Emax) / (self.eff * self.dt)
            p_bess = min(p_bess, max_charge)
        else:
            # Limit discharging by current SOC
            max_discharge = (self.soc * self.Emax * self.eff) / self.dt
            p_bess = -min(abs(p_bess), max_discharge)

        # Retrieve current timestamp and generation/load values
        t       = self.pv_series.index[self.current_idx]
        p_pv    = self.pv_series.iloc[self.current_idx] * self.PVmax
        p_load  = self.load_series.iloc[self.current_idx] * self.Loadmax

        # Update SOC and compute overflow/underflow penalty
        overflow_penalty = self._update_soc(p_bess)

        # Compute grid usage cost and constraint penalties
        p_grid       = p_load - p_pv + p_bess
        cost, tariff = self._compute_cost(p_grid, t.strftime("%H:00"))
        grid_pen     = self._compute_penalty(p_grid)
        misalign_pen = self._compute_alignment_bonus(p_bess, p_pv, p_load)

        # Base reward (negative of cost and penalties, plus misalignment bonus)
        reward = - cost - grid_pen - overflow_penalty + misalign_pen

        # Add PV‐charging bonus
        reward += self._charge_bonus(p_bess, p_pv, p_load)

        # Subtract mismatch penalty for discharge errors
        reward += self._match_penalty(p_bess, p_pv, p_load)

        # Advance time step
        self.current_idx += 1
        if self.current_idx >= self.end_idx:
            self.done = True

        # Build observation and info dict
        flat_obs, info = self._get_obs()
        info.update({
            "p_grid":            p_grid,
            "p_bess":            p_bess,
            "cost":              cost,
            "tariff":            tariff,
            "overflow_penalty":  overflow_penalty,
            "grid_penalty":      grid_pen,
            "misalignment_penalty": misalign_pen,
            # include the new terms for logging
            "charge_bonus":      self._charge_bonus(p_bess, p_pv, p_load),
            "match_penalty":     self._match_penalty(p_bess, p_pv, p_load),
            "time":              t
        })

        return flat_obs, reward, self.done, info

    def _get_obs(self):
        if self.current_idx >= len(self.pv_series):
            return np.zeros(len(self.obs_keys), dtype=np.float32), {}

        t = pd.to_datetime(self.pv_series.index[self.current_idx])
        pv_raw = self.pv_series.iloc[self.current_idx]
        load_raw = self.load_series.iloc[self.current_idx]

        obs = {
            "pv": pv_raw * self.PVmax / self.Pnom,
            "load": load_raw * self.Loadmax / self.Pnom,
            "pmax_norm": self.PEDS_max / self.Pnom,
            "pmin_norm": self.PEDS_min / self.Pnom,
            "soc": self.soc * self.Emax / self.Pnom,
            "hour_sin": np.sin(2 * np.pi * (t.hour / 24.0)),
            "day_sin": np.sin(2 * np.pi * (t.day / 31.0)),
            "month_sin": np.sin(2 * np.pi * (t.month / 12.0)),
            "weekday": t.weekday() / 6.0
,
        }
        obs["balance_ratio"] = obs["pv"] - obs["load"]

        hist = 288
        if self.current_idx >= hist:
            pv_hist = self.pv_series.iloc[self.current_idx - hist:self.current_idx].values
            load_hist = self.load_series.iloc[self.current_idx - hist:self.current_idx].values
        else:
            pv_hist = self.pv_series.iloc[:self.current_idx].values
            load_hist = self.load_series.iloc[:self.current_idx].values
            pad = hist - len(pv_hist)
            pv_hist = np.concatenate((np.zeros(pad), pv_hist))
            load_hist = np.concatenate((np.zeros(pad), load_hist))

        obs["pv_hist"] = (pv_hist * self.PVmax / self.Pnom).astype(np.float32)
        obs["load_hist"] = (load_hist * self.Loadmax / self.Pnom).astype(np.float32)
        obs["pmax"] = self.PEDS_max
        obs["pmin"] = self.PEDS_min

        flat_obs = np.concatenate([
            np.atleast_1d(obs[k]).astype(np.float32).flatten()
            for k in self.obs_keys if k in obs
        ])

        return flat_obs, obs

    def render(self, mode='human'):
        print(f"SoC: {self.soc:.2f}, PV(norm): {self.pv_series.iloc[self.current_idx]:.3f}, "
              f"Load(norm): {self.load_series.iloc[self.current_idx]:.3f}")


if __name__ == "__main__":
    # exemplo de uso
    env = EnergyEnv(data_dir='data', test=True)
    obs = env.reset()
    print("Obs inicial:", obs)
    for action in [np.array([2.0]), np.array([-1.0]), np.array([0.0])]:
        o, r, done, info = env.step(action)
        print("Obs:", o, "Rew:", r, "Done:", done)
