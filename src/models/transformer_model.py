import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerNextLocPredictor(nn.Module):
    """
    Transformer-based next location predictor.
    Inspired by BERT and GPT architectures adapted for trajectory prediction.
    """
    def __init__(
        self,
        num_locations=1187,
        num_users=46,
        embedding_dim=128,
        num_heads=4,
        num_layers=3,
        dropout=0.2,
        max_seq_len=60
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Embeddings
        self.loc_embedding = nn.Embedding(num_locations, embedding_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users, embedding_dim // 4, padding_idx=0)
        self.weekday_embedding = nn.Embedding(8, embedding_dim // 8, padding_idx=0)
        
        # Temporal feature projections
        self.time_projection = nn.Linear(1, embedding_dim // 8)
        self.duration_projection = nn.Linear(1, embedding_dim // 8)
        self.diff_projection = nn.Linear(1, embedding_dim // 8)
        
        # Combine all embeddings
        feature_dim = embedding_dim + embedding_dim // 4 + embedding_dim // 8 * 4
        self.input_projection = nn.Linear(feature_dim, embedding_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(embedding_dim, num_locations)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len=None):
        batch_size, seq_length = loc_seq.size()
        
        # Create embeddings
        loc_emb = self.loc_embedding(loc_seq)  # [B, L, D]
        user_emb = self.user_embedding(user_seq)  # [B, L, D/4]
        weekday_emb = self.weekday_embedding(weekday_seq)  # [B, L, D/8]
        
        # Temporal features
        time_emb = self.time_projection(start_min_seq.unsqueeze(-1).float() / 1440.0)  # Normalize to [0,1]
        dur_emb = self.duration_projection(dur_seq.unsqueeze(-1) / 500.0)  # Normalize duration
        diff_emb = self.diff_projection(diff_seq.unsqueeze(-1).float() / 7.0)  # Normalize diff
        
        # Concatenate all features
        combined = torch.cat([loc_emb, user_emb, weekday_emb, time_emb, dur_emb, diff_emb], dim=-1)
        
        # Project to model dimension
        x = self.input_projection(combined)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create attention mask (mask padding)
        mask = (loc_seq == 0)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Use the last non-padded token for prediction
        if seq_len is not None:
            # Gather last real token for each sequence
            indices = (seq_len - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.embedding_dim)
            x = torch.gather(x, 1, indices).squeeze(1)
        else:
            x = x[:, -1, :]  # Use last token
        
        x = self.dropout(x)
        logits = self.output_layer(x)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
