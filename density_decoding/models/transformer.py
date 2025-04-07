import math
import torch
import torch.nn as nn
from tqdm import tqdm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Create a set of frequencies using a logarithmic scale
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, n_c, n_t, d_model=64, nhead=8, num_layers=3, dropout=0.1):
        """
        Args:
            input_dim: Dimension of the input behavior at each time step (often 1).
            n_c: Number of components (output channels).
            n_t: Number of time steps.
            d_model: Dimension of the model embeddings.
            nhead: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dropout: Dropout probability.
        """
        super(TransformerModel, self).__init__()
        # Project the input behavior to the model's embedding dimension.
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=n_t)
        # Create a transformer encoder with the specified number of layers and heads.
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        # Final linear layer to predict n_c components at each time step.
        self.decoder = nn.Linear(d_model, n_c)
        
    def forward(self, y):
        # Expect y to have shape [batch, n_t, input_dim]
        x = self.embedding(y)                # -> [batch, n_t, d_model]
        x = self.pos_encoder(x)              # -> [batch, n_t, d_model]
        x = x.transpose(0, 1)                # -> [n_t, batch, d_model] for transformer
        x = self.transformer_encoder(x)      # -> [n_t, batch, d_model]
        x = x.transpose(0, 1)                # -> [batch, n_t, d_model]
        x_pred = self.decoder(x)             # -> [batch, n_t, n_c]
        x_pred = x_pred.transpose(1, 2)       # -> [batch, n_c, n_t] to match target shape
        return x_pred

def train_transformer(X, Y, train, test, input_dim=1, d_model=64, nhead=8, num_layers=3, dropout=0.1, learning_rate=1e-3, n_epochs=10000):
    """
    Trains the transformer-based model.
    
    Args:
        X: Target data of shape [num_samples, n_c, n_t]
        Y: Input behavior data of shape [num_samples, n_t] or [num_samples, n_t, input_dim]
        train: Indices for training samples.
        test: Indices for testing samples.
        input_dim: Dimensionality of the behavior input (default 1).
        d_model, nhead, num_layers, dropout: Transformer hyperparameters.
        learning_rate: Learning rate for the optimizer.
        n_epochs: Number of training epochs.
    
    Returns:
        model: Trained transformer model.
        losses: List of training loss values.
    """
    _, n_c, n_t = X.shape
    model = TransformerModel(input_dim=input_dim, n_c=n_c, n_t=n_t, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.PoissonNLLLoss()
    
    X = torch.tensor(X, dtype=torch.float)
    # Ensure Y has shape [num_samples, n_t, input_dim]
    if len(Y.shape) == 2:
        Y = Y.unsqueeze(-1)
    elif len(Y.shape) == 3:
        pass
    else:
        raise ValueError("Unexpected shape for Y")
    Y = torch.tensor(Y, dtype=torch.float)
    
    train_x, test_x = X[train], X[test]
    train_y, test_y = Y[train], Y[test]
    
    losses = []
    for epoch in tqdm(range(n_epochs), desc="Train Transformer:"):
        optimizer.zero_grad()
        x_pred = model(train_y)  # x_pred shape: [batch, n_c, n_t]
        loss = criterion(x_pred, train_x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, losses
