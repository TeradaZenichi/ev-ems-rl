import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LSTMDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64, num_layers=1):
        """
        Parameters:
          state_dim (int): Dimension of the state (e.g., 4)
          action_dim (int): Number of discrete actions (e.g., 10)
          hidden_size (int): Size of the hidden state for the LSTM
          num_layers (int): Number of stacked LSTM layers
        """
        super(LSTMDQN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer to process the sequence of states
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        # Final fully connected layer to map the last LSTM output to Q-values for each action
        self.fc = nn.Linear(hidden_size, action_dim)

    def forward(self, x, hidden=None):
        """
        Parameters:
          x (torch.Tensor): Input tensor of shape (batch, seq_len, state_dim)
          hidden (tuple, optional): Tuple (h_0, c_0) of initial hidden and cell states.
        
        Returns:
          q_values (torch.Tensor): Q-values for the last state in the sequence, shape (batch, action_dim)
          hidden (tuple): The updated hidden and cell states of the LSTM.
        """
        # Process the input sequence through the LSTM.
        out, hidden = self.lstm(x, hidden)
        # Extract the output corresponding to the last time step
        last_out = out[:, -1, :]  # shape: (batch, hidden_size)
        q_values = self.fc(last_out)
        return q_values, hidden