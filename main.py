import numpy as np
import matplotlib.pyplot as plt

# Importa o ambiente (script disponível em opt/environment.py)
from opt.environment import EVChargingSimulationEnv
# Importa o agente Q-learning implementado com PyTorch
from opt.models.qlearning import QLearningAgent

def main():
    # Caminhos para os dados e parâmetros
    csv_path = "data/EV-charging_train.csv"  # Utiliza os dados de treinamento
    parameters_path = "data/parameters.json"
    
    # Inicializa o ambiente
    env = EVChargingSimulationEnv(csv_path, parameters_path)
    
    # Dimensão do estado: (2 valores por carregador: SoC e battery_capacity)
    state_dim = env.observation_space.shape[0]
    
    # Define o conjunto de ações discretas:
    # Por exemplo, potência mínima, média e máxima
    action_values = [env.Pmin, (env.Pmin + env.Pmax) / 2, env.Pmax]
    action_dim = len(action_values)
    
    # Inicializa o agente Q-learning
    agent = QLearningAgent(input_dim=state_dim, output_dim=action_dim, lr=1e-3)
    
    episodes = 100
    max_steps = env.num_steps  # Cada passo corresponde a 5 minutos conforme o CSV
    rewards_per_episode = []
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps):
            # Seleciona uma ação (índice) com base na política epsilon-greedy
            action_idx = agent.select_action(state)
            # Mapeia o índice para o valor da potência
            action_value = action_values[action_idx]
            # Aplica a mesma ação para todos os carregadores (simplificação)
            actions = np.array([action_value] * len(env.chargers), dtype=np.float32)
            
            next_state, reward, done, info = env.step(actions)
            loss = agent.train_step(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
        rewards_per_episode.append(total_reward)
        print(f"Episódio {ep+1}: Recompensa Total = {total_reward:.2f}, Epsilon = {agent.epsilon:.3f}")
    
    # Plota a recompensa total por episódio
    plt.plot(rewards_per_episode)
    plt.xlabel("Episódios")
    plt.ylabel("Recompensa Total")
    plt.title("Treinamento Q-learning")
    plt.show()

if __name__ == "__main__":
    main()
