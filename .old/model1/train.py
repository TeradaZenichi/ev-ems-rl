import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from tqdm import tqdm  # Importa tqdm

# Importa o ambiente e o agente Q-learning (DQN)
from opt.environment import EVChargingSimulationEnv
from opt.models.qlearning import QLearningAgent

def main():
    # Define o dispositivo: GPU se disponível, caso contrário CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)
    
    # Caminhos para os dados e parâmetros
    train_csv_path = "data/EV-charging_train.csv"  # Dados de treinamento
    test_csv_path = "data/EV-charging_test.csv"      # Dados de teste
    parameters_path = "data/parameters.json"
    
    # Inicializa o ambiente de treinamento
    env = EVChargingSimulationEnv(train_csv_path, parameters_path)
    
    ## Limita o tamanho do dataframe para usar apenas 25% das linhas
    fraction = 0.25  
    env.df = env.df.iloc[:int(fraction * len(env.df))].reset_index(drop=True)
    env.num_steps = len(env.df)
    print("Número de passos limitado a:", env.num_steps)
    
    # Para o DQN atuando em um único carregador, usamos os 3 atributos:
    # [SoC, battery_capacity, charging_time] do primeiro carregador.
    state_dim = 3  
    
    # MODIFICAÇÃO: Definir um conjunto discreto maior de ações.
    num_actions = 10  # Exemplo: 10 ações
    action_values = np.linspace(env.Pmin, env.Pmax, num=num_actions)
    action_dim = len(action_values)
    
    # Inicializa o agente DQN e move o modelo para o dispositivo
    agent = QLearningAgent(input_dim=state_dim, output_dim=action_dim, lr=1e-3)
    agent.model.to(device)
    
    episodes = 100
    max_steps = env.num_steps  # Cada passo representa 5 minutos conforme o CSV limitado
    rewards_per_episode = []
    episode_times = []  # Lista para salvar o tempo de cada episódio
    
    best_reward = -np.inf
    best_model_path = "best_model.pth"
    
    previous_ep_end_time = time.time()
    # Uso do tqdm para o loop de treinamento
    for ep in tqdm(range(episodes), desc="Treinamento"):
        current_time = time.time()
        delay = current_time - previous_ep_end_time
        if delay > 5:  # se demorar mais que 5 segundos entre episódios
            print(f"Demora excessiva entre episódios: {delay:.2f} s")
        
        print(f"\nIniciando episódio {ep+1}")
        ep_start_time = time.time()  # Início do episódio
        
        state = env.reset()
        if len(state) > 3:
            state = state[:3]
        total_reward = 0
        
        for step in range(max_steps):
            action_idx = agent.select_action(state)
            action_value = action_values[action_idx]
            # Cria o vetor de ações:
            actions = np.array([action_value] + [env.Pmin] * (len(env.chargers) - 1), dtype=np.float32)
            
            next_state, reward, done, info = env.step(actions)
            if len(next_state) > 3:
                next_state = next_state[:3]
            loss = agent.train_step(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        ep_end_time = time.time()
        ep_duration = ep_end_time - ep_start_time
        episode_times.append(ep_duration)
        rewards_per_episode.append(total_reward)
        print(f"Episódio {ep+1}: Recompensa Total = {total_reward:.2f}, Epsilon = {agent.epsilon:.3f}, Tempo = {ep_duration:.2f} s")
        
        previous_ep_end_time = ep_end_time
        
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.model.state_dict(), best_model_path)
            print(f"Melhor modelo salvo com recompensa {best_reward:.2f}")
    
    # Plota os resultados do treinamento
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_per_episode)
    plt.xlabel("Episódios")
    plt.ylabel("Recompensa Total")
    plt.title("Treinamento DQN - Único Carregador")
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_times)
    plt.xlabel("Episódios")
    plt.ylabel("Tempo (s)")
    plt.title("Tempo por Episódio")
    plt.tight_layout()
    plt.show()
    
    # Avaliação no conjunto de teste
    print("\nAplicando o melhor modelo no conjunto de teste...")
    agent.model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_env = EVChargingSimulationEnv(test_csv_path, parameters_path)
    test_state = test_env.reset()
    if len(test_state) > 3:
        test_state = test_state[:3]
    
    agent.epsilon = 0.0  # Atuação determinística
    test_total_reward = 0
    steps = 0
    
    t0 = time.time()
    while True:
        action_idx = agent.select_action(test_state)
        action_value = action_values[action_idx]
        actions = np.array([action_value] + [test_env.Pmin] * (len(test_env.chargers) - 1), dtype=np.float32)
        next_state, reward, done, info = test_env.step(actions)
        if len(next_state) > 3:
            next_state = next_state[:3]
        test_total_reward += reward
        steps += 1
        
        test_env.render()
        test_state = next_state
        if done:
            break
    eval_time = time.time() - t0
    print(f"\nAvaliação concluída em {eval_time:.2f} s")
    print(f"Recompensa total no conjunto de teste: {test_total_reward:.2f} em {steps} passos")
    
if __name__ == "__main__":
    main()
