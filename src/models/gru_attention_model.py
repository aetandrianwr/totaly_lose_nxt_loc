import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUWithAttention(nn.Module):
    """
    GRU with multi-head attention for next location prediction.
    Optimized for parameter efficiency while maintaining performance.
    """
    def __init__(
        self,
        num_locations=1187,
        num_users=46,
        embedding_dim=96,
        hidden_dim=192,
        num_layers=2,
        num_heads=3,
        dropout=0.25,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embeddings with tied weights for efficiency
        self.loc_embedding = nn.Embedding(num_locations, embedding_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users, 24, padding_idx=0)
        self.weekday_embedding = nn.Embedding(8, 12, padding_idx=0)
        
        # Compact temporal encodings
        self.time_projection = nn.Linear(3, 24)  # Combined temporal features
        
        input_dim = embedding_dim + 24 + 12 + 24
        
        # GRU (more parameter efficient than LSTM)
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Multi-head attention
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer norm for stability
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, num_locations)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    nn.init.orthogonal_(param)
                elif len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def multi_head_attention(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Project
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        return self.out_proj(context)
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len=None):
        # Embeddings
        loc_emb = self.loc_embedding(loc_seq)
        user_emb = self.user_embedding(user_seq)
        weekday_emb = self.weekday_embedding(weekday_seq)
        
        # Combined temporal features
        temporal = torch.stack([
            start_min_seq.float() / 1440.0,
            dur_seq / 500.0,
            diff_seq.float() / 7.0
        ], dim=-1)
        temporal_emb = self.time_projection(temporal)
        
        # Combine all features
        x = torch.cat([loc_emb, user_emb, weekday_emb, temporal_emb], dim=-1)
        
        # GRU
        gru_out, h_n = self.gru(x)
        
        # Multi-head attention with residual
        attn_out = self.multi_head_attention(gru_out)
        gru_out = self.norm1(gru_out + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(gru_out)
        gru_out = self.norm2(gru_out + self.dropout(ffn_out))
        
        # Use last hidden state
        output = gru_out[:, -1, :]
        
        # Predict
        logits = self.output_layer(output)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
