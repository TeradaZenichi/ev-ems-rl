import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=[128, 128, 128, 128, 128]):
        """
        Parameters:
          state_dim: Dimensão da entrada.
          action_dim: Número de ações (dimensão da saída).
          hidden_layers: Lista com os tamanhos das camadas ocultas.
        """
        super(DQN, self).__init__()
        layers = []
        input_dim = state_dim
        # Cria as camadas ocultas dinamicamente
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        # Camada de saída
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


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
    

class SelfAttentionDQN(nn.Module):
    def __init__(self, input_dim, action_dim, embed_dim=128, num_heads=4, num_layers=1, seq_len=1):
        """
        Parâmetros:
          input_dim: Dimensão de cada token da entrada.
          action_dim: Número de ações (dimensão da saída dos Q-values).
          embed_dim: Dimensão do espaço de embedding.
          num_heads: Número de cabeças na atenção multi-cabeça.
          num_layers: Número de camadas do Transformer Encoder.
          seq_len: Comprimento da sequência de entrada.
        """
        super(SelfAttentionDQN, self).__init__()
        self.seq_len = seq_len
        
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, action_dim)
    
    def forward(self, x):
        """
        x: tensor de entrada com shape (batch_size, seq_len, input_dim)
        
        Retorna:
          q_values: tensor com shape (batch_size, action_dim)
        """
        x = self.input_projection(x)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)  # (batch_size, seq_len, embed_dim)
        x = x.mean(dim=1)  # (batch_size, embed_dim)
        q_values = self.fc_out(x)  # (batch_size, action_dim)
        return q_values

# Exemplo de uso:
if __name__ == "__main__":
    # Suponha que cada estado seja representado por 20 features e queremos uma sequência de 5 tokens
    batch_size = 8
    seq_len = 5
    input_dim = 20
    action_dim = 4  # Número de ações discretas

    # Cria uma entrada aleatória com shape (batch_size, seq_len, input_dim)
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    
    model = SelfAttentionDQN(input_dim=input_dim, action_dim=action_dim, seq_len=seq_len)
    q_vals = model(dummy_input)
    print("Q-values shape:", q_vals.shape)  # Esperado: (8, 4)
