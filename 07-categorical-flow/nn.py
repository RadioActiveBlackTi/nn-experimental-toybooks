import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class FLMDenoiser(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=16, num_layers=3):
        super().__init__()
        self.vocab_size = vocab_size

        self.input_proj = nn.Linear(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 4, embed_dim))  # seq_len = 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, 
            batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out_proj = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, It, t):
        # It: (batch_size, seq_len, vocab_size)
        x = self.input_proj(It)  # (batch_size, seq_len, embed_dim)

        time_emb = self.time_mlp(t).unsqueeze(1)  # (batch_size, 1, embed_dim)
        x = x + self.pos_embedding + time_emb 

        hidden = self.transformer(x)
        logits = self.out_proj(hidden)

        return logits