import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hl_number=3, hl_size=128):
        """
        Parameters:
            state_dim: Dimension of the input (number of state features).
            action_dim: Number of actions (dimension of Q-value output).
            hl_number: Number of hidden layers in the shared feature extractor.
            hl_size: Number of neurons in each hidden layer.
        """
        super(DDQN, self).__init__()
        
        # Create the shared feature extractor with hl_number layers, each with hl_size neurons.
        layers = []
        input_dim = state_dim
        for _ in range(hl_number):
            layers.append(nn.Linear(input_dim, hl_size))
            layers.append(nn.ReLU())
            input_dim = hl_size
        self.feature = nn.Sequential(*layers)
        
        # Value stream: estimate the state value V(s)
        self.value_fc = nn.Sequential(
            nn.Linear(input_dim, hl_size // 2),
            nn.ReLU(),
            nn.Linear(hl_size // 2, 1)
        )
        
        # Advantage stream: estimate the advantage A(s, a) for each action
        self.advantage_fc = nn.Sequential(
            nn.Linear(input_dim, hl_size // 2),
            nn.ReLU(),
            nn.Linear(hl_size // 2, action_dim)
        )
    
    def forward(self, x):
        """
        Forward pass for the network.
        
        Args:
            x: Input tensor with shape (batch_size, state_dim)
            
        Returns:
            Tensor of Q-values with shape (batch_size, action_dim)
        """
        # Pass through the shared feature extractor
        feature = self.feature(x)
        # Compute state value and advantage streams
        value = self.value_fc(feature)            # (batch_size, 1)
        advantage = self.advantage_fc(feature)      # (batch_size, action_dim)
        # Combine the two streams to get the Q-values using the dueling formula
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class CDDQN(nn.Module):
    def __init__(self, main_dim, cond_dim, action_dim, hl_number=3, hl_size=128, dropout_rate=0.0):
        """
        Parameters:
            main_dim (int): Dimension of main features (e.g., PV, load, hour_sin).
            cond_dim (int): Dimension of conditional features (e.g., soc, pmax, pmin).
            action_dim (int): Number of actions (output Q-values dimension).
            hl_number (int): Number of hidden layers in the main feature extractor.
            hl_size (int): Number of neurons in each hidden layer.
            dropout_rate (float): Dropout rate to be applied in the network.
        """
        super(CDDQN, self).__init__()
        
        # Build main feature extractor: dynamic MLP with hl_number layers
        layers_main = []
        input_dim = main_dim
        for _ in range(hl_number):
            layers_main.append(nn.Linear(input_dim, hl_size))
            layers_main.append(nn.ReLU())
            if dropout_rate > 0:
                layers_main.append(nn.Dropout(dropout_rate))
            input_dim = hl_size
        self.main_extractor = nn.Sequential(*layers_main)
        
        # Build conditional feature extractor: simple MLP for conditional input
        layers_cond = [
            nn.Linear(cond_dim, hl_size),
            nn.ReLU()
        ]
        if dropout_rate > 0:
            layers_cond.append(nn.Dropout(dropout_rate))
        self.cond_extractor = nn.Sequential(*layers_cond)
        
        # Fusion layer: combine main and conditional features
        self.fusion = nn.Sequential(
            nn.Linear(hl_size * 2, hl_size),
            nn.ReLU()
        )
        
        # Dueling streams
        
        # Value stream: estimates V(s)
        self.value_fc = nn.Sequential(
            nn.Linear(hl_size, hl_size // 2),
            nn.ReLU(),
            nn.Linear(hl_size // 2, 1)
        )
        
        # Advantage stream: estimates A(s,a) for each action
        self.advantage_fc = nn.Sequential(
            nn.Linear(hl_size, hl_size // 2),
            nn.ReLU(),
            nn.Linear(hl_size // 2, action_dim)
        )
        
    def forward(self, main_input, cond_input):
        """
        Forward pass for the Conditional Dueling DDQN.
        
        Args:
            main_input (torch.Tensor): Main feature input with shape (batch_size, main_dim).
            cond_input (torch.Tensor): Conditional feature input with shape (batch_size, cond_dim).
        
        Returns:
            torch.Tensor: Q-values for each action, shape (batch_size, action_dim).
        """
        # Process main features
        main_feat = self.main_extractor(main_input)  # (batch_size, hl_size)
        # Process conditional features
        cond_feat = self.cond_extractor(cond_input)    # (batch_size, hl_size)
        # Concatenate the features along the feature dimension
        combined = torch.cat((main_feat, cond_feat), dim=1)  # (batch_size, 2*hl_size)
        # Fuse the combined features into a single representation
        fused = self.fusion(combined)  # (batch_size, hl_size)
        
        # Compute value and advantage streams
        value = self.value_fc(fused)            # (batch_size, 1)
        advantage = self.advantage_fc(fused)      # (batch_size, action_dim)
        
        # Dueling aggregation: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class MHADDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hl_size=128, num_heads=4, mha_dim=64, ff_dim=128):
        """
        Multihead Attention Dueling Deep Q-Network (MHADDQN).

        Args:
            state_dim (int): Dimensão da entrada de estado.
            action_dim (int): Número de ações discretas.
            hl_size (int): Dimensão do embedding após projeção inicial.
            num_heads (int): Número de cabeças de atenção.
            mha_dim (int): Dimensão do espaço de atenção.
            ff_dim (int): Dimensão da feedforward após atenção.
        """
        super(MHADDQN, self).__init__()

        self.state_dim = state_dim

        # Projeção linear inicial da entrada para o embedding
        self.embedding = nn.Linear(state_dim, hl_size)

        # Multihead attention: usamos 1 "token" (batch, 1, hl_size) como input
        self.attention = nn.MultiheadAttention(embed_dim=hl_size, num_heads=num_heads, batch_first=True)

        # Feedforward após atenção
        self.ff = nn.Sequential(
            nn.Linear(hl_size, ff_dim),
            nn.ReLU()
        )

        # Dueling architecture
        self.value_stream = nn.Sequential(
            nn.Linear(ff_dim, ff_dim // 2),
            nn.ReLU(),
            nn.Linear(ff_dim // 2, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(ff_dim, ff_dim // 2),
            nn.ReLU(),
            nn.Linear(ff_dim // 2, action_dim)
        )

    def forward(self, x):
        """
        Forward pass do MHADDQN.

        Args:
            x (Tensor): Tensor de entrada (batch_size, state_dim)

        Returns:
            Tensor de Q-valores (batch_size, action_dim)
        """
        x = self.embedding(x).unsqueeze(1)  # (B, 1, hl_size)
        attn_output, _ = self.attention(x, x, x)  # Autoatenção
        attn_output = attn_output.squeeze(1)  # (B, hl_size)
        x = self.ff(attn_output)

        value = self.value_stream(x)  # (B, 1)
        advantage = self.advantage_stream(x)  # (B, A)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)  # Dueling

        return q



class HMHADDQN(nn.Module):
    """
    Historical Multi‑Head Attention Dueling DQN
    com Positional Encoding Aprendível (Learnable Encoding)
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        history_len: int = 4,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim   = state_dim
        self.history_len = history_len
        self.d_model     = d_model     # == d_key == d_value

        # 1.  Embedding do vetor de observação de cada passo
        self.input_proj = nn.Linear(state_dim, d_model)

        # 2.  Positional encodings **aprendíveis**
        #     Shape: (history_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(history_len, d_model) * 0.02)

        # 3.  Multi‑head self‑attention (batch_first=True ⇒ entradas (B, T, D))
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        # 4.  Feed‑forward pós‑atenção
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 5.  Dueling heads
        hidden_v = d_ff // 2
        hidden_a = d_ff // 2

        self.value_stream = nn.Sequential(
            nn.Linear(d_ff, hidden_v),
            nn.ReLU(),
            nn.Linear(hidden_v, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(d_ff, hidden_a),
            nn.ReLU(),
            nn.Linear(hidden_a, action_dim)
        )

        self._reset_parameters()

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : (B, T, state_dim)  — T == history_len

        Returns
        -------
        q_values : (B, action_dim)
        """
        B, T, _ = x.shape
        assert T == self.history_len, "input seq len must equal history_len"

        x = self.input_proj(x) + self.pos_embed   # broadcast PE → (T,d) + (B,T,d)
        attn_out, _ = self.attn(x, x, x)          # (B, T, d_model)
        context = attn_out.mean(dim=1)            # (B, d_model)
        feat = self.ff(context)                   # (B, d_ff)

        # 5) dueling
        value      = self.value_stream(feat)      # (B, 1)
        advantage  = self.adv_stream(feat)        # (B, A)
        q_values   = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values.squeeze(-1)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        # pos_embed já inicializado ~N(0,0.02)



class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_f, self.out_f = in_features, out_features
        self.mu_W = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_b = nn.Parameter(torch.empty(out_features))
        self.sigma_W = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.sigma_b = nn.Parameter(torch.full((out_features,), sigma_init))
        # já não precisamos mais desses buffers para a forward
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_f)
        nn.init.uniform_(self.mu_W, -bound, bound)
        nn.init.uniform_(self.mu_b, -bound, bound)

    def forward(self, x):
        if self.training:
            # amostragem OUT-OF-PLACE
            eps_W = torch.randn_like(self.sigma_W)
            eps_b = torch.randn_like(self.sigma_b)
            W = self.mu_W + self.sigma_W * eps_W
            b = self.mu_b + self.sigma_b * eps_b
        else:
            W = self.mu_W
            b = self.mu_b
        return F.linear(x, W, b)



class NHMHADDQN(nn.Module):
    """
    Noisy Historical Multi-Head Attention Dueling DQN
    with Learnable Positional Encoding and Noisy Dueling Heads.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        history_len: int = 4,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.history_len = history_len

        # 1) input projection
        self.input_proj = nn.Linear(state_dim, d_model)

        # 2) learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(history_len, d_model) * 0.02)

        # 3) multi-head self-attention
        self.attn = nn.MultiheadAttention(embed_dim=d_model,
                                          num_heads=num_heads,
                                          batch_first=True,
                                          dropout=dropout)

        # 4) feed-forward após atenção
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 5) dueling heads now with NoisyLinear
        hidden_v = d_ff // 2
        hidden_a = d_ff // 2

        self.value_stream = nn.Sequential(
            NoisyLinear(d_ff, hidden_v),
            nn.ReLU(),
            NoisyLinear(hidden_v, 1),
        )
        self.adv_stream = nn.Sequential(
            NoisyLinear(d_ff, hidden_a),
            nn.ReLU(),
            NoisyLinear(hidden_a, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        assert T == self.history_len

        # embedding + positional
        x = self.input_proj(x) + self.pos_embed
        # self-attention
        attn_out, _ = self.attn(x, x, x)
        # temporal pooling
        context = attn_out.mean(dim=1)
        # feed-forward
        feat = self.ff(context)
        # dueling with noise
        v = self.value_stream(feat)      # (B,1)
        a = self.adv_stream(feat)        # (B,A)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q.squeeze(-1)
