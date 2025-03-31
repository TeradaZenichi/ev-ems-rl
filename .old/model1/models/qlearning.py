import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Rede neural para aproximar os Q-values.
        Para um único carregador, input_dim deve ser 3 (ex.: [SoC, battery_capacity, charging_time])
        e output_dim é o número de ações discretas (por exemplo, 3 para potências mínima, média e máxima).
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class QLearningAgent:
    def __init__(self, input_dim, output_dim, lr=1e-3, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Inicializa o agente de Q-learning para um único carregador.
        
        Parâmetros:
          - input_dim (int): Dimensão da entrada; para um carregador, deve ser 3.
          - output_dim (int): Número de ações discretas (ex.: 3).
          - lr (float): Taxa de aprendizado.
          - gamma (float): Fator de desconto.
          - epsilon (float): Taxa de exploração inicial.
          - epsilon_decay (float): Fator de decaimento do epsilon.
          - epsilon_min (float): Valor mínimo para epsilon.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.output_dim = output_dim
        
        ## MODIFICAÇÃO GPU: Define o dispositivo (GPU se disponível, senão CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = QNetwork(input_dim, output_dim)
        self.model.to(self.device)  # MODIFICAÇÃO GPU: Move o modelo para o dispositivo
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def select_action(self, state):
        """
        Seleciona uma ação usando a política epsilon-greedy.
        
        Se uma ação for escolhida aleatoriamente (exploração), retorna um índice entre 0 e output_dim-1.
        Caso contrário, utiliza o modelo para prever os Q-values e escolhe a ação com maior valor.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.output_dim)
        else:
            # MODIFICAÇÃO GPU: Envia o estado para o dispositivo
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values, dim=1).item()
    
    def train_step(self, state, action, reward, next_state, done):
        # MODIFICAÇÃO GPU: Converte os dados para tensores e os envia para o dispositivo
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([float(done)]).to(self.device)
        
        # Previsão atual: Q(s, a)
        q_values = self.model(state_tensor)
        # Aqui, garantimos que q_value tenha shape [1]
        q_value = q_values[0, action].unsqueeze(0)
        
        # Cálculo do target: r + γ * max Q(s', a')
        with torch.no_grad():
            next_q_values = self.model(next_state_tensor)
            next_q_value = torch.max(next_q_values)
            target = reward_tensor + self.gamma * next_q_value * (1 - done_tensor)
        
        # Perda e backpropagation
        loss = self.criterion(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Atualiza o epsilon se o episódio terminou
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
