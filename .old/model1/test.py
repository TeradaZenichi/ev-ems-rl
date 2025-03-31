import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import time

# Importa o ambiente e o agente Q-learning (DQN)
from opt.environment import EVChargingSimulationEnv
from opt.models.qlearning import QLearningAgent

def main():
    # Define o dispositivo: GPU se disponível, caso contrário CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)
    
    # Caminhos para os dados e parâmetros
    test_csv_path = "data/EV-charging_test.csv"      # Dados de teste
    parameters_path = "data/parameters.json"
    best_model_path = "best_model.pth"  # Modelo previamente salvo
    
    # Inicializa o ambiente de teste
    env = EVChargingSimulationEnv(test_csv_path, parameters_path)
    
    # Para o DQN atuando em um único carregador, usamos os 3 atributos:
    # [SoC, battery_capacity, charging_time] do primeiro carregador.
    state_dim = 3  
    # Define o conjunto de ações discretas (ex.: potência mínima, média e máxima)
    action_values = [env.Pmin, (env.Pmin + env.Pmax) / 2, env.Pmax]
    action_dim = len(action_values)
    
    # Inicializa o agente DQN e carrega o melhor modelo salvo
    agent = QLearningAgent(input_dim=state_dim, output_dim=action_dim, lr=1e-3)
    agent.model.load_state_dict(torch.load(best_model_path, map_location=device))
    agent.model.to(device)
    # Força atuação determinística (sem exploração)
    agent.epsilon = 0.0
    
    # Reseta o ambiente e extrai o estado do primeiro carregador
    state = env.reset()
    if len(state) > 3:
        state = state[:3]
    
    # Cria a pasta 'Results/' se não existir
    results_dir = "Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    test_total_reward = 0
    steps = 0
    saved_week = None  ## Variável para monitorar a semana atual

    t0 = time.time()
    while True:
        action_idx = agent.select_action(state)
        action_value = action_values[action_idx]
        # Cria o vetor de ações: o primeiro carregador é controlado pelo DQN,
        # os demais recebem a ação padrão (env.Pmin)
        actions = np.array([action_value] + [env.Pmin] * (len(env.chargers) - 1), dtype=np.float32)
        
        next_state, reward, done, info = env.step(actions)
        if len(next_state) > 3:
            next_state = next_state[:3]
        test_total_reward += reward
        steps += 1
        
        # MODIFICAÇÃO: Salva imagem para cada semana progressivamente
        current_week = info["timestamp"].isocalendar().week
        if saved_week is None:
            saved_week = current_week
        elif current_week != saved_week:
            # Cria texto com os detalhes da operação para a semana encerrada
            text = f"Operation details for Week {saved_week}\nTimestamp: {info['timestamp']}\n"
            for charger in env.chargers:
                session = env.sessions.get(charger)
                if session is not None:
                    text += f"{charger} - SoC: {session['SoC']:.2f}%, Capacidade: {session['battery_capacity']:.2f} kWh, Tempo: {session['charging_time']} min\n"
                else:
                    text += f"{charger} - Sem sessão ativa.\n"
            plt.figure(figsize=(8, 6))
            plt.axis('off')
            plt.text(0.5, 0.5, text, ha='center', va='center', fontsize=12)
            plt.title(f"Week {saved_week} Operation")
            plt.savefig(os.path.join(results_dir, f"week_{saved_week}.png"))
            plt.close()
            saved_week = current_week  # Atualiza para a nova semana
        
        # Opcional: renderiza o ambiente no console para debug
        env.render()
        
        state = next_state
        if done:
            break

    # Após o loop, salva imagem para a última semana, se ainda não foi salva
    if saved_week is not None:
        text = f"Operation details for Week {saved_week}\nTimestamp: {info['timestamp']}\n"
        for charger in env.chargers:
            session = env.sessions.get(charger)
            if session is not None:
                text += f"{charger} - SoC: {session['SoC']:.2f}%, Capacidade: {session['battery_capacity']:.2f} kWh, Tempo: {session['charging_time']} min\n"
            else:
                text += f"{charger} - Sem sessão ativa.\n"
        plt.figure(figsize=(8, 6))
        plt.axis('off')
        plt.text(0.5, 0.5, text, ha='center', va='center', fontsize=12)
        plt.title(f"Week {saved_week} Operation")
        plt.savefig(os.path.join(results_dir, f"week_{saved_week}.png"))
        plt.close()
    
    eval_time = time.time() - t0
    print(f"\nAvaliação concluída em {eval_time:.2f} s")
    print(f"Recompensa total no conjunto de teste: {test_total_reward:.2f} em {steps} passos")

if __name__ == "__main__":
    main()
