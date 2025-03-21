import pandas as pd
import numpy as np
import json
import gym
from gym import spaces
from datetime import datetime, timedelta

class EVChargingSimulationEnv(gym.Env):
    """
    Ambiente de simulação para recarga de veículos com base em um CSV.

    Regras:
      - O CSV possui a coluna "timestamp" e colunas para cada carregador.
      - Quando há transição de 0 para 1 (início da sessão), sorteamos:
          * SoC inicial (por exemplo, Uniform(10, 30)%)
          * Capacidade da bateria (por exemplo, Uniform(50, 100) kWh)
          * Inicialmente, tempo de carregamento = 0 minutos
      - Enquanto o carregador estiver ativo (valor 1), o agente escolhe uma potência
        de recarga (ação) que incrementa o SoC do veículo e o tempo de carregamento é incrementado.
      - Quando há transição de 1 para 0 (último 1 antes de desligar) e o SoC não estiver em 100%,
        aplica-se uma penalidade: penalty_rate * (100 - SoC). Ao fim da sessão, os dados do tempo de carregamento são resetados.
      - Cada passo da simulação representa 5 minutos.
    """
    def __init__(self, csv_path, parameters_path):
        super(EVChargingSimulationEnv, self).__init__()
        
        # Carregar parâmetros do arquivo JSON
        with open(parameters_path, 'r') as f:
            params = json.load(f)
        
        # Parâmetros do EDS
        self.Pmax = params["EDS"]["Pmax"]
        self.Pmin = params["EDS"]["Pmin"]
        self.cost = params["EDS"]["cost"]
        # Parâmetro do RL (penalidade)
        self.penalty_rate = params["RL"]["Penalty"]
        
        # Lista de carregadores de interesse (deve dar match com as colunas do CSV)
        self.chargers = params["EVCS"]
        
        # Carregar CSV e ordenar pela coluna timestamp
        self.df = pd.read_csv(csv_path, sep=",", encoding="utf-8")
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        self.df.sort_values("timestamp", inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        # Definir o timestamp inicial e final da simulação
        self.start_time = self.df["timestamp"].min()
        self.end_time = self.df["timestamp"].max()
        
        # Índice de simulação (cada linha representa 5 minutos)
        self.current_step = 0
        self.num_steps = len(self.df)
        
        # Espaço de ação: para cada carregador, o agente escolhe a potência (kW) no intervalo [Pmin, Pmax]
        self.action_space = spaces.Box(low=self.Pmin, high=self.Pmax, 
                                       shape=(len(self.chargers),), dtype=np.float32)
        
        # Espaço de observação: agora, para cada carregador, retornaremos [SoC, battery_capacity, charging_time].
        # Se não houver sessão ativa, os valores serão 0.
        self.observation_space = spaces.Box(low=0.0, high=100.0, 
                                            shape=(3 * len(self.chargers),), dtype=np.float32)
        
        # Dicionário para manter o estado das sessões de carregamento para cada carregador.
        # Cada entrada será None (sem sessão ativa) ou um dicionário com:
        #   "SoC": estado de carga atual (%)
        #   "battery_capacity": capacidade total da bateria (kWh)
        #   "charging_time": tempo de carregamento acumulado (minutos)
        self.sessions = {charger: None for charger in self.chargers}
    
    def reset(self):
        self.current_step = 0
        self.sessions = {charger: None for charger in self.chargers}
        return self._get_obs()
    
    def _get_obs(self):
        """
        Constrói o vetor de observação concatenando, para cada carregador, os valores:
        [SoC, battery_capacity, charging_time].
        Se não houver sessão ativa, todos os valores serão 0.
        """
        obs = []
        for charger in self.chargers:
            session = self.sessions.get(charger)
            if session is not None:
                obs.extend([
                    session["SoC"], 
                    session["battery_capacity"], 
                    session["charging_time"]
                ])
            else:
                obs.extend([0.0, 0.0, 0.0])
        return np.array(obs, dtype=np.float32)
    
    def step(self, actions):
        """
        Executa um passo da simulação (5 minutos).
        Os actions são um array com uma ação para cada carregador (na ordem de self.chargers),
        representando a potência de recarga escolhida (kW).
        """
        if self.current_step >= self.num_steps - 1:
            done = True
            return self._get_obs(), 0.0, done, {}
        
        # Obter a linha atual e a próxima linha do CSV
        current_row = self.df.iloc[self.current_step]
        next_row = self.df.iloc[self.current_step + 1]
        
        total_reward = 0.0
        
        # Para cada carregador definido em parameters.json
        for i, charger in enumerate(self.chargers):
            current_val = current_row.get(charger, 0)
            next_val = next_row.get(charger, 0)
            
            # Se houver transição de 0 para 1: início de sessão
            if (current_val == 0) and (next_val == 1):
                initial_SoC = np.random.uniform(10, 30)
                battery_capacity = np.random.uniform(50, 100)
                # Inicializa a sessão com charging_time = 0
                self.sessions[charger] = {
                    "SoC": initial_SoC, 
                    "battery_capacity": battery_capacity,
                    "charging_time": 0.0
                }
            
            # Se a sessão estiver ativa no próximo passo (valor 1)
            if next_val == 1 and self.sessions.get(charger) is not None:
                power = np.clip(actions[i], self.Pmin, self.Pmax)
                # Energia entregue em 5 minutos (kWh)
                energy_delivered = power * (5/60)
                # Atualiza o SoC
                increment = (energy_delivered / self.sessions[charger]["battery_capacity"]) * 100
                new_SoC = min(100.0, self.sessions[charger]["SoC"] + increment)
                self.sessions[charger]["SoC"] = new_SoC
                # Incrementa o tempo de carregamento (em minutos)
                self.sessions[charger]["charging_time"] += 5
                # Custo associado à energia entregue
                cost = energy_delivered * self.cost
                total_reward -= cost
            
            # Se houver transição de 1 para 0: fim da sessão
            if (current_val == 1) and (next_val == 0) and self.sessions.get(charger) is not None:
                final_SoC = self.sessions[charger]["SoC"]
                if final_SoC < 100.0:
                    missing = 100.0 - final_SoC
                    penalty = self.penalty_rate * missing
                    total_reward -= penalty
                # Finaliza a sessão e reseta o tempo de carregamento
                self.sessions[charger] = None
        
        # Penalização global: se a soma das potências (ações) for maior que self.Pmax, aplica penalidade
        total_power = np.sum(actions)
        if total_power > self.Pmax:
            penalty_global = self.penalty_rate * (total_power - self.Pmax)
            total_reward -= penalty_global
        
        self.current_step += 1
        done = self.current_step >= self.num_steps - 1
        
        obs = self._get_obs()
        # Inclui informação adicional: número de carregadores ativos (em sessão)
        active_chargers = sum(1 for s in self.sessions.values() if s is not None)
        info = {
            "timestamp": self.df.iloc[self.current_step]["timestamp"],
            "active_chargers": active_chargers
        }
        return obs, total_reward, done, info
    
    def render(self, mode="human"):
        current_time = self.df.iloc[self.current_step]["timestamp"]
        print(f"Timestamp: {current_time}")
        for charger in self.chargers:
            session = self.sessions.get(charger)
            if session is not None:
                print(f"{charger} - SoC: {session['SoC']:.2f}%, Capacidade: {session['battery_capacity']:.2f} kWh, Tempo carregando: {session['charging_time']} min")
            else:
                print(f"{charger} - Sem sessão ativa.")

if __name__ == "__main__":
    env = EVChargingSimulationEnv(csv_path="data/EV-charging.csv", parameters_path="data/parameters.json")
    env.reset()
    
    start_time = env.start_time
    end_time = env.end_time
    print("Horário inicial:", start_time)
    print("Horário final:", end_time)
    
    current_time = start_time
    counter = 0
    total_reward = 0.0
    
    while current_time <= end_time:
        print(f"\nPasso {counter} - Tempo: {current_time}")
        actions = np.array([env.Pmin] * len(env.chargers), dtype=np.float32)
        obs, reward, done, info = env.step(actions)
        total_reward += reward
        
        env.render()
        
        current_time += timedelta(minutes=5)
        counter += 1
        
        if done:
            break

    print("\nRecompensa total da simulação:", total_reward)
