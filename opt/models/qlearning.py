import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
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
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.output_dim = output_dim
        
        self.model = QNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def select_action(self, state):
        """
        Seleciona uma ação usando a política epsilon-greedy.
        Se a ação for escolhida aleatoriamente, retorna um índice entre 0 e output_dim-1.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.output_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values, dim=1).item()
    
    def train_step(self, state, action, reward, next_state, done):
        """
        Executa um passo de treinamento utilizando a atualização do Q-learning.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward])
        done_tensor = torch.FloatTensor([float(done)])
        
        # Previsão atual: Q(s, a)
        q_values = self.model(state_tensor)
        q_value = q_values[0, action]
        
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
