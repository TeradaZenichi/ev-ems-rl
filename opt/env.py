import gym
import numpy as np
import pandas as pd
import json
import os
from gym import spaces

class EnergyEnv(gym.Env):
    def __init__(
        self,
        data_dir='data',
        timestep_min=5,
        start_idx=0,
        episode_length=288,
        test=False,
        observations=None,
        data=None
    ):
        super(EnergyEnv, self).__init__()

        # load global parameters
        with open(os.path.join(data_dir, 'parameters.json'), 'r') as f:
            self.params = json.load(f)

        self.Pnom  = self.params.get("Pnom", 1)
        self.bonus = self.params["RL"].get("Bonus", 0)

        # observation keys
        self.obs_keys = observations or ["pv", "load", "pmax_norm", "pmin_norm", "soc", "hour_sin", "day_sin", "month_sin", "weekday"]
        self.bess_params = self.params["BESS"]
        self.initial_soc = self.bess_params["SoC0"]
        self.Emax         = self.bess_params["Emax"]
        self.Pmax_charge  = self.bess_params["Pmax_c"]
        self.Pmax_discharge = self.bess_params["Pmax_d"]
        self.eff          = self.bess_params["eff"]
        self.dt           = self.params["timestep"] / 60.0
        self.PEDS_max     = self.params["EDS"]["Pmax"]
        self.PEDS_min     = self.params["EDS"]["Pmin"]
        self.PVmax        = self.params["PV"]["Pmax"]
        self.Loadmax      = self.params["Load"]["Pmax"]
        self.cost_dict    = self.params["EDS"]["cost"]
        self.difficulty   = float(self.params["ENV"]["difficulty"])

        self.test_mode = test

        # select train/test file suffix
        mode = 'test' if self.test_mode else 'train'
        if data is not None:
            mode = data

        pv_file   = f'pv_5min_{mode}.csv'
        load_file = f'load_5min_{mode}.csv'

        self.pv_data   = pd.read_csv(os.path.join(data_dir, pv_file),
                                     index_col='timestamp', parse_dates=['timestamp'])
        self.load_data = pd.read_csv(os.path.join(data_dir, load_file),
                                     index_col='timestamp', parse_dates=['timestamp'])

        self.pv_series   = self.pv_data['p_norm']
        self.load_series = self.load_data['p_norm']

        self.start_idx      = start_idx
        self.episode_length = episode_length
        self.current_idx    = self.start_idx
        self.end_idx        = self.start_idx + self.episode_length

        self.action_space = spaces.Box(
            low=-self.Pmax_discharge,
            high=self.Pmax_charge,
            shape=(1,),
            dtype=np.float32
        )

        self.soc  = self.initial_soc
        self.done = False

        flat_obs, _ = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=flat_obs.shape,
            dtype=np.float32
        )

        self.reset()

    def new_training_episode(self, start_idx):
        """
        Reset indices and SoC for a new training episode.
        """
        self.start_idx      = start_idx
        self.current_idx    = self.start_idx
        self.end_idx        = self.start_idx + self.episode_length
        self.soc            = self.initial_soc
        self.done           = False
        self.difficulty     = float(self.params["ENV"]["difficulty"])
        self.episode_counter = 0
        return self.reset()

    def reset(self):
        """
        Reset environment.
        In test mode: keep SoC and indices fixed.
        In train mode: apply curriculum and randomization.
        """
        if self.test_mode:
            self.soc         = self.initial_soc
            self.current_idx = self.start_idx
            self.end_idx     = self.start_idx + self.episode_length
            self.done        = False
        else:
            if not hasattr(self, "episode_counter"):
                self.episode_counter = 0
            self.episode_counter += 1

            # curriculum update
            env_cfg = self.params["ENV"]
            if env_cfg.get("curriculum", "False").upper() == "TRUE" and \
               self.episode_counter % int(env_cfg["curriculum_steps"]) == 0:
                inc = float(env_cfg["curriculum_increment"])
                mx  = float(env_cfg["curriculum_max"])
                self.difficulty = min(self.difficulty + inc, mx)

            rand_obs = env_cfg.get("randomize_observations", {})

            # randomize SoC
            if rand_obs.get("soc", "False").upper() == "TRUE" and env_cfg.get("randomize", "False").upper() == "TRUE":
                soc_range = 0.05 + self.difficulty * 0.95
                low  = max(0, 0.5 - soc_range/2)
                high = min(1, 0.5 + soc_range/2)
                self.soc = np.random.uniform(low, high)
            else:
                self.soc = self.initial_soc

            # randomize EDS capacity
            if rand_obs.get("eds", "False").upper() == "TRUE" and env_cfg.get("randomize", "False").upper() == "TRUE":
                scale = 0.05 + self.difficulty
                factor = 1 + np.random.uniform(-scale, scale)
                self.PEDS_max = max(0, self.params["EDS"]["Pmax"] * factor)
                self.PEDS_min = max(0, self.params["EDS"]["Pmin"] * factor)
            else:
                self.PEDS_max = self.params["EDS"]["Pmax"]
                self.PEDS_min = self.params["EDS"]["Pmin"]

            # randomize start index
            if rand_obs.get("idx", "False").upper() == "TRUE" and env_cfg.get("randomize", "False").upper() == "TRUE":
                limit = int((0.2 + 0.6*self.difficulty) * 0.1 * len(self.pv_series))
                self.start_idx = np.random.randint(0, max(1, limit - self.episode_length))

            self.current_idx = self.start_idx
            self.end_idx     = self.start_idx + self.episode_length
            self.done        = False

            if self.episode_counter % 10 == 0:
                print(f"Episode {self.episode_counter}: difficulty={self.difficulty:.2f}, SoC={self.soc:.2f}")

        flat_obs, _ = self._get_obs()
        return flat_obs

    def _update_soc(self, p_bess):
        """
        Update SoC based on charge/discharge and compute overflow penalty.
        """
        if p_bess >= 0:
            delta = (p_bess * self.eff * self.dt) / self.Emax
        else:
            delta = (p_bess / self.eff * self.dt) / self.Emax

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
        cost  = (p_grid * self.dt if p_grid > 0 else 0.0) * tarif
        return cost, tarif

    def _compute_penalty(self, p_grid):
        pen = self.params["RL"]["Penalty"]
        if p_grid > self.PEDS_max:
            return pen * (p_grid - self.PEDS_max) * self.dt
        if p_grid < -self.PEDS_min:
            return pen * (-self.PEDS_min - p_grid) * self.dt
        return 0.0

    def _compute_alignment_bonus(self, p_bess, p_pv, p_load):
        expected = max(0.0, p_load - p_pv)
        mis = 0.0
        if p_bess < 0:
            excess = abs(p_bess) - expected
            mis = max(0.0, excess)
        return - self.params["RL"].get("misalignment_penalty", 0.0) * mis * self.dt

    def _charge_bonus(self, p_bess, p_pv, p_load):
        """
        Reward for using PV surplus to charge the battery.
        """
        if p_bess <= 0:
            return 0.0
        surplus = max(0.0, p_pv - p_load)
        used    = min(p_bess, surplus)
        rate    = self.params["RL"].get("charge_bonus", 1.0)
        return rate * used * self.dt

    def _match_penalty(self, p_bess, p_pv, p_load):
        """
        Penalize only the mismatch beyond a tolerance delta between discharge
        and the ideal (load - PV).
        """
        if p_bess >= 0:
            return 0.0
        expected    = max(0.0, p_load - p_pv)
        actual      = abs(p_bess)
        base_error  = abs(actual - expected)
        tol         = self.params["RL"].get("match_tol", 0.0)
        excess      = max(0.0, base_error - tol)
        rate        = self.params["RL"].get("match_penalty", 1.0)
        return - rate * excess * self.dt

    def step(self, action):
        # Clip action
        p_bess = np.clip(action[0], -self.Pmax_discharge, self.Pmax_charge)
        # enforce capacity limits
        if p_bess > 0:
            max_c = ((1 - self.soc) * self.Emax) / (self.eff * self.dt)
            p_bess = min(p_bess, max_c)
        else:
            max_d = (self.soc * self.Emax * self.eff) / self.dt
            p_bess = -min(abs(p_bess), max_d)

        # observations
        t      = self.pv_series.index[self.current_idx]
        p_pv   = self.pv_series.iloc[self.current_idx] * self.PVmax
        p_load = self.load_series.iloc[self.current_idx] * self.Loadmax

        # compute costs & penalties
        overflow_penalty = self._update_soc(p_bess)
        p_grid, tarif     = self._compute_cost(p_load - p_pv + p_bess, t.strftime("%H:00"))
        grid_penalty      = self._compute_penalty(p_load - p_pv + p_bess)
        align_bonus       = self._compute_alignment_bonus(p_bess, p_pv, p_load)

        # base reward
        reward  = - p_grid - grid_penalty - overflow_penalty + align_bonus
        # add new bonuses/penalties
        # reward += self._charge_bonus(p_bess, p_pv, p_load)
        reward += self._match_penalty(p_bess, p_pv, p_load)

        # advance
        self.current_idx += 1
        if self.current_idx >= self.end_idx:
            self.done = True

        flat_obs, info = self._get_obs()
        info.update({
            "p_grid":             p_load - p_pv + p_bess,
            "p_bess":             p_bess,
            "cost":               p_grid,
            "tariff":             tarif,
            "overflow_penalty":   overflow_penalty,
            "grid_penalty":       grid_penalty,
            "misalignment_penalty": align_bonus,
            "charge_bonus":       self._charge_bonus(p_bess, p_pv, p_load),
            "match_penalty":      self._match_penalty(p_bess, p_pv, p_load),
            "time":               t
        })

        return flat_obs, reward, self.done, info

    def _get_obs(self):
        if self.current_idx >= len(self.pv_series):
            return np.zeros(len(self.obs_keys), dtype=np.float32), {}

        t        = pd.to_datetime(self.pv_series.index[self.current_idx])
        pv_raw   = self.pv_series.iloc[self.current_idx]
        load_raw = self.load_series.iloc[self.current_idx]

        obs = {
            "pv":        pv_raw * self.PVmax / self.Pnom,
            "load":      load_raw * self.Loadmax / self.Pnom,
            "pmax": self.PEDS_max / self.Pnom,
            "pmin": self.PEDS_min / self.Pnom,
            "soc":       self.soc * self.Emax / self.Pnom,
            "hour_sin":  np.sin(2*np.pi*(t.hour/24.0)),
            "day_sin":   np.sin(2*np.pi*(t.day/31.0)),
            "month_sin": np.sin(2*np.pi*(t.month/12.0)),
            "weekday":   t.weekday() / 6.0,
        }
        obs["balance_ratio"] = obs["pv"] - obs["load"]

        flat_obs = np.concatenate([
            np.atleast_1d(obs[k]).astype(np.float32).flatten()
            for k in self.obs_keys
            if k in obs
        ])
        return flat_obs, obs

    def render(self, mode='human'):
        print(f"SoC={self.soc:.2f}, PV(norm)={self.pv_series.iloc[self.current_idx]:.3f}, "
              f"Load(norm)={self.load_series.iloc[self.current_idx]:.3f}")


if __name__ == "__main__":
    # exemplo de uso
    env = EnergyEnv(data_dir='data', test=True)
    obs = env.reset()
    print("Obs inicial:", obs)
    for action in [np.array([2.0]), np.array([-1.0]), np.array([0.0])]:
        o, r, done, info = env.step(action)
        print("Obs:", o, "Rew:", r, "Done:", done)
