# transformer.py

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
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, n_c, n_t, d_model=64, nhead=8, num_layers=3, dropout=0.1, n_r=2):
        super(TransformerModel, self).__init__()
        # Transformer components
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=n_t)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, n_c)
        
        # Parameters to be integrated into the model output and later used as priors for ADVI.
        self.n_r = n_r
        self.U = nn.Parameter(torch.randn(n_c, n_r))
        self.V = nn.Parameter(torch.randn(n_r, n_t))
        self.b = nn.Parameter(torch.randn(1, n_c, 1))
        
        # Optional: learnable weight to balance transformer vs. GLM branch (or fix a constant weight)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # alpha in [0,1]
        
    def forward(self, y):
        # Transformer branch
        x = self.embedding(y)                 # [batch, n_t, d_model]
        x = self.pos_encoder(x)               # [batch, n_t, d_model]
        x = x.transpose(0, 1)                 # [n_t, batch, d_model]
        x = self.transformer_encoder(x)       # [n_t, batch, d_model]
        x = x.transpose(0, 1)                 # [batch, n_t, d_model]
        transformer_out = self.decoder(x)     # [batch, n_t, n_c]
        transformer_out = transformer_out.transpose(1, 2)  # [batch, n_c, n_t]
        
        # GLM-style branch using U, V, and b
        beta = torch.einsum("cr,rt->ct", self.U, self.V)  # [n_c, n_t]
        glm_out = beta.unsqueeze(0) + self.b               # [1, n_c, n_t]
        # glm_out = beta.unsqueeze(0)
        # Combine both outputs; here, alpha controls the contribution from the transformer branch.
        final_output = self.alpha * transformer_out + (1 - self.alpha) * glm_out
        # final_output = transformer_out*glm_out + self.b

        print("y.shape: ",y.shape)
        print("transformer_out.shape: ",transformer_out.shape)
        print("self.U.shape: ",self.U.shape)
        print("self.V.shape: ",self.V.shape)
        print("glm_out.shape: ",glm_out.shape)
        print("self.alpha.shape: ",self.alpha.shape)
        print("final output.shape: ",final_output.shape)
        return final_output

def train_transformer(X, Y, train, test, input_dim=1, d_model=64, nhead=8, num_layers=3, dropout=0.1, learning_rate=1e-3, n_epochs=10000, n_r=2):
    _, n_c, n_t = X.shape
    model = TransformerModel(input_dim=input_dim, n_c=n_c, n_t=n_t, d_model=d_model, 
                             nhead=nhead, num_layers=num_layers, dropout=dropout, n_r=n_r).float()  # ensure model is float32
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.PoissonNLLLoss()
    
    X = torch.tensor(X, dtype=torch.float)
    Y = torch.tensor(Y, dtype=torch.float)
    if len(Y.shape) == 2:
        Y = Y.unsqueeze(-1)
    elif len(Y.shape) != 3:
        raise ValueError("Unexpected shape for Y")
    
    train_x, test_x = X[train], X[test]
    train_y, test_y = Y[train], Y[test]
    
    losses = []
    for epoch in tqdm(range(n_epochs), desc="Train Transformer:"):
        optimizer.zero_grad()
        x_pred = model(train_y)
        loss = criterion(x_pred, train_x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, losses
