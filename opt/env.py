import gym
import numpy as np
import pandas as pd
import json
import os
from gym import spaces

class EnergyEnv(gym.Env):
    """
    Simulation environment for a system with:
      - Photovoltaic Generation (PV)
      - Load (demand)
      - Battery (BESS)
      - Grid energy cost
      
    The agent decides the charging/discharging power of the BESS, where:
      - A positive value indicates charging (energy consumption to charge the battery).
      - A negative value indicates discharging (energy injected by the battery).
      
    The power balance is given by:
      p_grid = p_load - p_pv + p_bess
      
    Thus, when the BESS discharges (p_bess negative), the grid demand is reduced.
    """
    def __init__(self,
                 data_dir='data',
                 timestep_min=5,
                 start_idx=0,
                 episode_length=288):
        """
        Parameters:
            data_dir (str): path to the folder containing CSV files and parameters.json.
            timestep_min (int): duration of each time step in minutes (e.g., 5 min).
            start_idx (int): starting index in the dataset for the episode.
            episode_length (int): number of time steps in the episode (e.g., 288 steps of 5 min = 24h).
        """
        super(EnergyEnv, self).__init__()
        
        # --- Load parameters ---
        with open(os.path.join(data_dir, 'parameters.json'), 'r') as f:
            self.params = json.load(f)

        # BESS parameters
        self.bess_params = self.params["BESS"]
        self.initial_soc = self.bess_params["SoC0"]       # Initial SoC (fraction [0,1])
        self.Emax = self.bess_params["Emax"]              # Battery capacity (kWh)
        self.Pmax_charge = self.bess_params["Pmax_c"]       # Maximum charging power (kW)
        self.Pmax_discharge = self.bess_params["Pmax_d"]    # Maximum discharging power (kW)
        self.eff = self.bess_params["eff"]                # Charging/discharging efficiency
        self.dt = self.params["timestep"]/60

        # PV and Load parameters (normalized data)
        self.PVmax = self.params["PV"]["Pmax"]
        self.Loadmax = self.params["Load"]["Pmax"]

        # Grid parameters: hourly tariff and limits for grid power
        self.cost_dict = self.params["EDS"]["cost"]

        # --- Load PV and Load data ---
        # Read CSV files with date parsing
        self.pv_data = pd.read_csv(
            os.path.join(data_dir, 'pv_5min.csv'), parse_dates=True)
        self.load_data = pd.read_csv(os.path.join(data_dir, 'load_5min.csv'), parse_dates=True)

        # Check if the CSVs have a 'timestamp' column. If not, create a date range.
        if 'timestamp' not in self.pv_data.columns:
            self.pv_data['timestamp'] = pd.date_range(start='2021-01-01', periods=len(self.pv_data), freq='5min')
        if 'timestamp' not in self.load_data.columns:
            self.load_data['timestamp'] = pd.date_range(start='2021-01-01', periods=len(self.load_data), freq='5min')

        self.pv_data.set_index('timestamp', inplace=True)
        self.pv_data.index = pd.to_datetime(self.pv_data.index)

        self.load_data.set_index('timestamp', inplace=True)
        self.load_data.index = pd.to_datetime(self.load_data.index)

        # Assume the normalized data is in the 'p_norm' column
        self.pv_series = self.pv_data['p_norm']
        self.load_series = self.load_data['p_norm']

        # Episode control
        self.start_idx = start_idx
        self.episode_length = episode_length
        self.current_idx = self.start_idx
        self.end_idx = self.start_idx + self.episode_length

        # --- Define action and observation spaces ---
        # Action: BESS power (kW) between -Pmax_discharge (discharge) and +Pmax_charge (charge)
        self.action_space = spaces.Box(low=-self.Pmax_discharge, high=self.Pmax_charge, shape=(1,), dtype=np.float32)

        # Observation: [SoC, PV (normalized), Load (normalized), hour_sin]
        # SoC ∈ [0,1], PV and Load ∈ [0,1], hour_sin ∈ [-1,1]
        obs_low  = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0, 1.0,  1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(4,), dtype=np.float32)

        # Internal state
        self.soc = None
        self.done = False

        self.reset()

    def reset(self):
        """
        Resets the episode.
        """
        self.current_idx = self.start_idx
        self.end_idx = self.start_idx + self.episode_length
        self.done = False
        self.soc = self.initial_soc  # Initial SoC
        return self._get_obs()

    def step(self, action):
        """
        Executes one simulation step.
        
        Parameters:
            action (array): BESS charging/discharging power (kW). Positive for charging, negative for discharging.
        
        Returns:
            obs (np.array): next state
            reward (float): reward (negative of the imported energy cost and any penalty)
            done (bool): True if the episode has ended
            info (dict): additional information
        """
        # Limit the action to the defined bounds
        p_bess = np.clip(action[0], -self.Pmax_discharge, self.Pmax_charge)
        
        # Energy-based constraints:
        if p_bess > 0:
            # Charging: battery gains energy = p_bess * eff * dt (kWh)
            available_energy = (1.0 - self.soc) * self.Emax
            max_power_allowed = available_energy / (self.eff * self.dt)
            if p_bess > max_power_allowed:
                p_bess = max_power_allowed
        elif p_bess < 0:
            # Discharging: battery loses energy = |p_bess|/eff * dt (kWh)
            available_energy = self.soc * self.Emax
            max_power_allowed = available_energy * self.eff / self.dt
            if abs(p_bess) > max_power_allowed:
                p_bess = -max_power_allowed

        # Get current PV and Load values
        t_current = self.pv_series.index[self.current_idx]
        p_pv_norm = self.pv_series.iloc[self.current_idx]
        p_load_norm = self.load_series.iloc[self.current_idx]

        # Convert normalized values to kW
        p_pv = p_pv_norm * self.PVmax
        p_load = p_load_norm * self.Loadmax

        # --- Update BESS SoC ---
        if p_bess >= 0:
            # Charging: increase SoC
            delta_soc = (p_bess * self.eff * self.dt) / self.Emax
        else:
            # Discharging: decrease SoC
            delta_soc = (p_bess / self.eff * self.dt) / self.Emax
        
        new_soc = self.soc + delta_soc
        self.soc = np.clip(new_soc, 0.0, 1.0)

        # --- Calculate grid power ---
        # p_grid = p_load - p_pv + p_bess
        # If p_bess is negative (discharging), it reduces grid demand.
        p_grid = p_load - p_pv + p_bess

        # --- Calculate energy cost ---
        # Cost is considered only if p_grid > 0 (energy import)
        hour_str = t_current.strftime("%H:00")
        tariff = self.cost_dict.get(hour_str, 0.4)
        energy_import = p_grid * self.dt if p_grid > 0 else 0
        cost = energy_import * tariff

        # Reward is the negative cost
        reward = -cost

        # --- Penalize if grid power exceeds limits ---
        eds_max = self.params["EDS"]["Pmax"]
        eds_min = self.params["EDS"]["Pmin"]
        penalty = self.params["RL"]["Penalty"]
        if p_grid > eds_max:
            reward -= penalty * (p_grid - eds_max)
        elif p_grid < -eds_min:
            reward -= penalty * (-eds_min - p_grid)

        # Advance the index
        self.current_idx += 1
        if self.current_idx >= self.end_idx:
            self.done = True

        obs = self._get_obs()
        info = {
            "p_grid": p_grid,
            "p_bess": p_bess,
            "cost": cost,
            "time": t_current
        }
        return obs, reward, self.done, info

    def _get_obs(self):
        """
        Returns the current state of the environment.
        
        State: [SoC, normalized PV, normalized Load, hour_sin]
        """
        t_current = pd.to_datetime(self.pv_series.index[self.current_idx])
        p_pv_norm = self.pv_series.iloc[self.current_idx]
        p_load_norm = self.load_series.iloc[self.current_idx]

        # Encode the time of day: use the sine of the hour to capture periodicity
        hour = t_current.hour
        angle = 2 * np.pi * (hour / 24.0)
        hour_sin = np.sin(angle)

        obs = np.array([
            self.soc,
            p_pv_norm,
            p_load_norm,
            hour_sin
        ], dtype=np.float32)
        return obs

    def render(self, mode='human'):
        # Simple example: printing the main states of the environment
        print(f"Battery SoC: {self.soc:.2f}")
        print(f"Current PV (normalized): {self.pv_series.iloc[self.current_idx]}")
        print(f"Current Load (normalized): {self.load_series.iloc[self.current_idx]}")

if __name__ == "__main__":
    # Debug block to manually test the environment.
    # We define a list of actions to simulate, where:
    #   - Positive values indicate charging.
    #   - Negative values indicate discharging.
    env = EnergyEnv(data_dir='data')
    obs = env.reset()
    print("Initial observation:", obs)

    # List of test actions (in kW)
    test_actions = [
        np.array([2.0]),   # Charge with 2 kW
        np.array([-1.0]),  # Discharge with 1 kW
        np.array([0.0]),   # No action
        np.array([3.0]),   # Charge with 3 kW
        np.array([-2.0])   # Discharge with 2 kW
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
