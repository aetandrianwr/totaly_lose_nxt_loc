import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EnhancedNextLocPredictor(nn.Module):
    """
    Enhanced next location predictor combining:
    - Multi-scale temporal convolutions (inspired by TCN)
    - Self-attention mechanism
    - User and location embeddings with factorization
    - Residual connections
    
    Target: <500K params, >40% Acc@1
    """
    def __init__(
        self,
        num_locations=1187,
        num_users=46,
        embedding_dim=64,
        hidden_dim=96,
        num_heads=4,
        dropout=0.2,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Location embedding with better initialization
        self.loc_embedding = nn.Embedding(num_locations, embedding_dim, padding_idx=0)
        
        # User embedding
        self.user_embedding = nn.Embedding(num_users, 32, padding_idx=0)
        
        # Temporal embeddings
        self.weekday_embedding = nn.Embedding(8, 16, padding_idx=0)
        self.hour_embedding = nn.Embedding(24, 16)  # Extract hour from minutes
        
        # Continuous temporal features
        self.temporal_mlp = nn.Sequential(
            nn.Linear(3, 32),  # time_of_day, duration, diff
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Total input dimension
        input_dim = embedding_dim + 32 + 16 + 16 + 32  # loc + user + weekday + hour + temporal
        
        # Multi-scale temporal convolutions (reduced to 2 scales)
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        
        self.conv_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Self-attention
        self.num_heads = num_heads
        assert (hidden_dim * 2) % num_heads == 0
        self.attention_dim = hidden_dim * 2
        self.head_dim = self.attention_dim // num_heads
        
        self.q_proj = nn.Linear(self.attention_dim, self.attention_dim)
        self.k_proj = nn.Linear(self.attention_dim, self.attention_dim)
        self.v_proj = nn.Linear(self.attention_dim, self.attention_dim)
        self.attn_out = nn.Linear(self.attention_dim, self.attention_dim)
        
        self.attn_norm = nn.LayerNorm(self.attention_dim)
        
        # Feed-forward network (smaller)
        self.ffn = nn.Sequential(
            nn.Linear(self.attention_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.attention_dim)
        )
        
        self.ffn_norm = nn.LayerNorm(self.attention_dim)
        
        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(self.attention_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_locations)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        # Better initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'embedding' in name:
                    nn.init.normal_(param, mean=0, std=0.02)
                elif len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def self_attention(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Multi-head projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.attention_dim)
        
        return self.attn_out(context)
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len=None):
        batch_size, seq_length = loc_seq.size()
        
        # Embeddings
        loc_emb = self.loc_embedding(loc_seq)
        user_emb = self.user_embedding(user_seq)
        weekday_emb = self.weekday_embedding(weekday_seq)
        
        # Extract hour from start_min
        hour = (start_min_seq.float() / 60).long() % 24
        hour_emb = self.hour_embedding(hour)
        
        # Continuous temporal features (normalized)
        temporal_features = torch.stack([
            start_min_seq.float() / 1440.0,  # Time of day [0, 1]
            torch.clamp(dur_seq / 500.0, 0, 1),  # Duration normalized
            diff_seq.float() / 7.0  # Day difference
        ], dim=-1)
        temporal_emb = self.temporal_mlp(temporal_features)
        
        # Combine all embeddings
        x = torch.cat([loc_emb, user_emb, weekday_emb, hour_emb, temporal_emb], dim=-1)
        
        # Multi-scale temporal convolutions
        x_t = x.transpose(1, 2)  # [B, D, L]
        
        conv1_out = F.gelu(self.conv1(x_t))
        conv2_out = F.gelu(self.conv2(x_t))
        
        # Concatenate multi-scale features
        conv_out = torch.cat([conv1_out, conv2_out], dim=1)  # [B, 2*H, L]
        conv_out = conv_out.transpose(1, 2)  # [B, L, 2*H]
        conv_out = self.conv_norm(conv_out)
        
        # Create padding mask
        mask = (loc_seq == 0)
        
        # Self-attention with residual
        attn_out = self.self_attention(conv_out, mask)
        x = self.attn_norm(conv_out + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + self.dropout(ffn_out))
        
        # Extract final representation (last non-padded token)
        if seq_len is not None:
            indices = (seq_len - 1).clamp(0, seq_length - 1).to(x.device)
            indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.attention_dim)
            final_repr = torch.gather(x, 1, indices).squeeze(1)
        else:
            final_repr = x[:, -1, :]
        
        # Prediction
        logits = self.predictor(final_repr)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
