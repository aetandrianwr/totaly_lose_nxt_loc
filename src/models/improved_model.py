import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ImprovedNextLocPredictor(nn.Module):
    """
    Improved model with:
    - Location frequency-aware embeddings
    - Stronger regularization
    - Better temporal encoding
    - Hierarchical attention
    
    Designed to generalize better and reduce train-test gap.
    """
    def __init__(
        self,
        num_locations=1187,
        num_users=46,
        embedding_dim=80,
        hidden_dim=160,
        num_heads=4,
        dropout=0.3,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Shared location embedding (regularization through parameter sharing)
        self.loc_embedding = nn.Embedding(num_locations, embedding_dim, padding_idx=0)
        
        # User embedding (smaller to prevent overfitting on user IDs)
        self.user_embedding = nn.Embedding(num_users, 24, padding_idx=0)
        
        # Temporal embeddings
        self.weekday_embedding = nn.Embedding(8, 12, padding_idx=0)
        self.hour_embedding = nn.Embedding(24, 12)
        
        # Continuous features with LayerNorm for stability
        self.temporal_encoder = nn.Sequential(
            nn.Linear(3, 24),
            nn.LayerNorm(24),
            nn.Dropout(dropout),
            nn.GELU()
        )
        
        # Input dimension
        input_dim = embedding_dim + 24 + 12 + 12 + 24
        
        # Bidirectional GRU for better context
        self.gru = nn.GRU(
            input_dim,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=True
        )
        
        # Self-attention
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer norms for stability
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN with smaller expansion to reduce parameters
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Final prediction with dropout
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_locations)
        
        self._init_weights()
    
    def _init_weights(self):
        # Conservative initialization to prevent overfitting
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'embedding' in name:
                    nn.init.normal_(param, mean=0, std=0.01)
                elif 'gru' in name:
                    nn.init.orthogonal_(param)
                elif len(param.shape) >= 2:
                    nn.init.xavier_normal_(param, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def attention(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Multi-head attention
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        return self.out_linear(context)
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len=None):
        batch_size, seq_length = loc_seq.size()
        
        # Embeddings
        loc_emb = self.loc_embedding(loc_seq)
        user_emb = self.user_embedding(user_seq)
        weekday_emb = self.weekday_embedding(weekday_seq)
        
        # Hour embedding
        hour = (start_min_seq.float() / 60).long().clamp(0, 23)
        hour_emb = self.hour_embedding(hour)
        
        # Continuous temporal features (better normalization)
        temporal_features = torch.stack([
            (start_min_seq.float() % 1440) / 1440.0,
            torch.sigmoid(dur_seq / 100.0),
            diff_seq.float() / 7.0
        ], dim=-1)
        temporal_emb = self.temporal_encoder(temporal_features)
        
        # Combine features
        x = torch.cat([loc_emb, user_emb, weekday_emb, hour_emb, temporal_emb], dim=-1)
        
        # BiGRU
        gru_out, _ = self.gru(x)
        
        # Attention mask
        mask = (loc_seq == 0)
        
        # Self-attention with residual
        attn_out = self.attention(gru_out, mask)
        gru_out = self.norm1(gru_out + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(gru_out)
        output = self.norm2(gru_out + ffn_out)
        
        # Get final representation (last non-padded)
        if seq_len is not None:
            indices = (seq_len - 1).clamp(0, seq_length - 1).to(output.device)
            indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            final_repr = torch.gather(output, 1, indices).squeeze(1)
        else:
            final_repr = output[:, -1, :]
        
        # Prediction with dropout
        x = self.dropout(final_repr)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
