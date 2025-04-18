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
                 observations=None):
        super(EnergyEnv, self).__init__()

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
        self.test_mode = test

        self.mode = self.params["ENV"]["MODE"]
        self.randomize = self.params["ENV"]["randomize"]
        self.curriculum = self.params["ENV"]["curriculum"]
        self.curriculum_steps = int(self.params["ENV"]["curriculum_steps"])
        self.curriculum_increment = float(self.params["ENV"]["curriculum_increment"])
        self.curriculum_max = float(self.params["ENV"]["curriculum_max"])
        self.difficulty = float(self.params["ENV"]["difficulty"])
        self.randomize_observations = self.params["ENV"]["randomize_observations"]

        self.pv_data = pd.read_csv(os.path.join(data_dir, 'pv_5min.csv'), index_col='timestamp', parse_dates=['timestamp'])
        self.load_data = pd.read_csv(os.path.join(data_dir, 'load_5min.csv'), index_col='timestamp', parse_dates=['timestamp'])
        self.pv_series = self.pv_data['p_norm']
        self.load_series = self.load_data['p_norm']
        
        self.start_idx = start_idx
        self.episode_length = episode_length
        self.current_idx = self.start_idx
        self.end_idx = self.start_idx + self.episode_length

        self.action_space = spaces.Box(low=-self.Pmax_discharge, high=self.Pmax_charge, shape=(1,), dtype=np.float32)

        self.soc = self.initial_soc
        self.done = False

        flat_obs, _ = self._get_obs()
        obs_dim = flat_obs.shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.reset()

    def new_training_episode(self, start_idx):
        """
        Starts a new training episode with the specified start index and episode length.
        """
        self.start_idx = start_idx
        self.current_idx = self.start_idx
        self.end_idx = self.start_idx + self.episode_length
        self.soc = self.initial_soc
        self.done = False
        self.difficulty = float(self.params["ENV"]["difficulty"])
        self.episode_counter = 0
        self.reset()
        return self._get_obs()[0]  # Return the flattened observation

    
    def reset(self):
        """
        Resets the episode.
        - In test mode, resets with fixed parameters.
        - In train mode, if randomization is enabled, then for each parameter listed in
        `self.params["ENV"]["randomize_observations"]` the value is randomized according to the current difficulty.
        """
        if str(self.mode).upper() == "TEST":
            self.soc = self.initial_soc
            self.current_idx = self.start_idx
            self.end_idx = self.start_idx + self.episode_length
            self.done = False

        elif str(self.mode).upper() == "TRAIN":

            # Initialize or increment the episode counter.
            if not hasattr(self, "episode_counter"):
                self.episode_counter = 0
            self.episode_counter += 1

            # Update difficulty every `curriculum_steps` episodes.
            if str(self.curriculum).upper() == "TRUE" and (self.episode_counter % self.curriculum_steps == 0):
                self.difficulty = min(self.difficulty + self.curriculum_increment, self.curriculum_max)
                print(f"Curriculum update: new difficulty = {self.difficulty}")

            rand_obs = self.params["ENV"].get("randomize_observations", {})

            # ---- SoC randomization with expanding range centered at 0.5 ----
            if rand_obs.get("soc", "False").upper() == "TRUE" and str(self.randomize).upper() == "TRUE":
                soc_range = 0.05 + self.difficulty * 0.45  # max range ±0.225 around 0.5
                lower_bound = max(0, 0.5 - soc_range / 2)
                upper_bound = min(1, 0.5 + soc_range / 2)
                self.soc = np.random.uniform(lower_bound, upper_bound)
            else:
                self.soc = self.initial_soc

            # ---- PEDS max/min randomization with scaled variability ----
            if rand_obs.get("eds", "False").upper() == "TRUE" and str(self.randomize).upper() == "TRUE":
                scale = 0.05 + 0.15 * self.difficulty  # from ±5% to ±20%
                factor = 1 + np.random.uniform(-scale, scale)
                self.PEDS_max = self.params["EDS"]["Pmax"] * factor
                self.PEDS_min = self.params["EDS"]["Pmin"] * factor
            else:
                self.PEDS_max = self.params["EDS"]["Pmax"]
                self.PEDS_min = self.params["EDS"]["Pmin"]

            # ---- Randomize start index progressively along the time base ----
            if rand_obs.get("idx", "False").upper() == "TRUE" and str(self.randomize).upper() == "TRUE":
                limit_idx = int((0.2 + 0.6 * self.difficulty) * 0.1*len(self.pv_series))
                self.start_idx = np.random.randint(0, max(1, limit_idx - self.episode_length))
                self.current_idx = self.start_idx
                self.end_idx = self.start_idx + self.episode_length
            else:
                self.current_idx = self.start_idx
                self.end_idx = self.start_idx + self.episode_length

            self.done = False

        else:
            self.soc = self.initial_soc
            self.current_idx = self.start_idx
            self.end_idx = self.start_idx + self.episode_length
            self.done = False

        return self._get_obs()[0]  # Return the flattened observation

    def _update_soc(self, p_bess):
        delta_soc = (p_bess * self.eff * self.dt) / self.Emax if p_bess >= 0 else (p_bess / self.eff * self.dt) / self.Emax
        soc_before = self.soc
        self.soc += delta_soc
        bess_penalty = 0.0
        if self.soc > 1.0:
            overflow = self.soc - 1.0
            bess_penalty += overflow * abs(p_bess) * self.params["RL"].get("bess_penalty", 10.0)
            self.soc = 1.0
        elif self.soc < 0.0:
            underflow = -self.soc
            bess_penalty += underflow * abs(p_bess) * self.params["RL"].get("bess_penalty", 10.0)
            self.soc = 0.0
        return bess_penalty * self.dt

    def _compute_cost(self, p_grid, hour_str):
        tariff = self.cost_dict.get(hour_str, 0.4)
        return (p_grid * self.dt if p_grid > 0 else 0) * tariff, tariff

    def _compute_penalty(self, p_grid):
        penalty = self.params["RL"]["Penalty"]
        if p_grid > self.PEDS_max:
            return penalty * (p_grid - self.PEDS_max) * self.dt
        elif p_grid < -self.PEDS_min:
            return penalty * (-self.PEDS_min - p_grid) * self.dt
        return 0.0

    def _compute_alignment_bonus(self, p_bess, p_pv, p_load):
        expected_discharge = max(0, p_load - p_pv)
        # expected_charge = max(0, p_pv - p_load)
        misalignment = 0

        if p_bess < 0:
            excess_discharge = abs(p_bess) - expected_discharge
            misalignment = max(0, excess_discharge)

        return -self.params["RL"].get("misalignment_penalty", 0.0) * misalignment * self.dt

    def step(self, action):
        p_bess = np.clip(action[0], -self.Pmax_discharge, self.Pmax_charge)
        if p_bess > 0:
            max_allowed = ((1.0 - self.soc) * self.Emax) / (self.eff * self.dt)
            p_bess = min(p_bess, max_allowed)
        elif p_bess < 0:
            max_allowed = (self.soc * self.Emax * self.eff) / self.dt
            p_bess = -min(abs(p_bess), max_allowed)

        t_current = self.pv_series.index[self.current_idx]
        p_pv = self.pv_series.iloc[self.current_idx] * self.PVmax
        p_load = self.load_series.iloc[self.current_idx] * self.Loadmax

        overflow_penalty = self._update_soc(p_bess)
        p_grid = p_load - p_pv + p_bess
        cost, tariff = self._compute_cost(p_grid, t_current.strftime("%H:00"))
        penalty_value = self._compute_penalty(p_grid)
        misalignment_penalty = self._compute_alignment_bonus(p_bess, p_pv, p_load)

        reward = -cost - penalty_value - overflow_penalty + misalignment_penalty
        balance_ratio = p_pv - p_load
        bonus_value = 0
        if balance_ratio > 0 and p_bess > 0:
            bonus_value = self.bonus * balance_ratio
        elif balance_ratio < 0 and p_bess < 0:
            bonus_value = self.bonus * abs(balance_ratio)
        reward += bonus_value * self.dt

        self.current_idx += 1
        if self.current_idx >= self.end_idx:
            self.done = True

        flat_obs, full_obs = self._get_obs()
        if full_obs is None:
            full_obs = {}
        full_obs.update({"p_grid": p_grid, "p_bess": p_bess, "cost": cost, "tariff": tariff,
                         "penalty_applied": penalty_value, "bonus_applied": bonus_value,
                         "overflow_penalty": overflow_penalty, "misalignment_penalty": misalignment_penalty,
                         "time": t_current})
        return flat_obs, reward, self.done, full_obs

    def _get_obs(self, mode="normalized"):
        obs = {}
        if self.current_idx >= len(self.pv_series):
            return np.zeros(len(self.obs_keys), dtype=np.float32), None

        t_current = pd.to_datetime(self.pv_series.index[self.current_idx])
        p_pv_raw = self.pv_series.iloc[self.current_idx]
        p_load_raw = self.load_series.iloc[self.current_idx]

        if mode == "normalized":
            obs["pv"] = p_pv_raw * self.PVmax / self.Pnom
            obs["load"] = p_load_raw * self.Loadmax / self.Pnom
            obs["pmax_norm"] = self.PEDS_max / self.Pnom
            obs["pmin_norm"] = self.PEDS_min / self.Pnom
            obs["soc"] = self.soc * self.Emax / self.Pnom
        else:
            obs["pv"] = p_pv_raw * self.PVmax
            obs["load"] = p_load_raw * self.Loadmax
            obs["pmax_norm"] = self.PEDS_max
            obs["pmin_norm"] = self.PEDS_min
            obs["soc"] = self.soc

        obs["hour_sin"] = np.sin(2 * np.pi * (t_current.hour / 24.0))
        obs["day_sin"] = np.sin(2 * np.pi * (t_current.day / 31.0))
        obs["month_sin"] = np.sin(2 * np.pi * (t_current.month / 12.0))
        obs["weekday"] = float(t_current.weekday())

        obs["balance_ratio"] = obs["pv"] - obs["load"]
        obs["soc_ratio"] = obs["soc"] * self.Emax / self.dt

        history_length = 288
        if self.current_idx >= history_length:
            pv_hist = self.pv_series.iloc[self.current_idx - history_length:self.current_idx].values
            load_hist = self.load_series.iloc[self.current_idx - history_length:self.current_idx].values
        else:
            pv_hist = self.pv_series.iloc[:self.current_idx].values
            load_hist = self.load_series.iloc[:self.current_idx].values
            if len(pv_hist) < history_length:
                pad = np.zeros(history_length - len(pv_hist))
                pv_hist = np.concatenate((pad, pv_hist))
                load_hist = np.concatenate((pad, load_hist))

        if mode == "normalized":
            obs["pv_hist"] = np.array(pv_hist * self.PVmax / self.Pnom).flatten()
            obs["load_hist"] = np.array(load_hist * self.Loadmax / self.Pnom).flatten()
        else:
            obs["pv_hist"] = np.array(pv_hist * self.PVmax).flatten()
            obs["load_hist"] = np.array(load_hist * self.Loadmax).flatten()

        obs["pmax"] = self.PEDS_max
        obs["pmin"] = self.PEDS_min

        flat_obs = np.concatenate([
            np.array(obs[k], dtype=np.float32).flatten() for k in self.obs_keys if k in obs
        ])

        return flat_obs, obs

    def render(self, mode='human'):
        print(f"Battery SoC: {self.soc:.2f}")
        print(f"Current PV (normalized): {self.pv_series.iloc[self.current_idx]}")
        print(f"Current Load (normalized): {self.load_series.iloc[self.current_idx]}")

if __name__ == "__main__":
    env = EnergyEnv(data_dir='data')
    obs = env.reset()
    print("Initial observation:", obs)
    test_actions = [
        np.array([2.0]),
        np.array([-1.0]),
        np.array([0.0]),
        np.array([3.0]),
        np.array([-2.0])
    ]
    for i, action in enumerate(test_actions):
        print(f"\nStep {i+1} with action: {action}")
        obs, reward, done, info = env.step(action)
        print("Observation:", obs)
        print("Reward:", reward)
        print("Info:", info)
        if done:
            print("Episode finished.")
            break
