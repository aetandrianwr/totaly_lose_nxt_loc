import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedPredictor(nn.Module):
    """
    Optimized architecture balancing capacity and regularization.
    Key improvements:
    - BiGRU for better context
    - Residual connections
    - Careful dropout placement
    - Label smoothing + dropout for regularization
    """
    def __init__(
        self,
        num_locations=1187,
        num_users=46,
        embedding_dim=80,
        hidden_dim=144,
        dropout=0.28,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Embeddings
        self.loc_embedding = nn.Embedding(num_locations, embedding_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users, 24, padding_idx=0)
        
        # Temporal embeddings
        self.weekday_emb = nn.Embedding(8, 14, padding_idx=0)
        self.hour_emb = nn.Embedding(24, 14)
        
        # Continuous features
        self.temporal_fc = nn.Sequential(
            nn.Linear(2, 24),
            nn.Dropout(dropout * 0.5),
            nn.GELU()
        )
        
        input_dim = embedding_dim + 24 + 14 + 14 + 24
        
        # BiGRU
        self.gru = nn.GRU(
            input_dim,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, num_locations)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len=None):
        batch_size, seq_length = loc_seq.size()
        
        # Embeddings
        loc_emb = self.loc_embedding(loc_seq)
        user_emb = self.user_embedding(user_seq)
        weekday_emb = self.weekday_emb(weekday_seq)
        
        # Hour
        hour = (start_min_seq.float() / 60).long().clamp(0, 23)
        hour_emb = self.hour_emb(hour)
        
        # Temporal features
        temp_feat = torch.stack([
            dur_seq / 500.0,
            diff_seq.float() / 7.0
        ], dim=-1)
        temp_emb = self.temporal_fc(temp_feat)
        
        # Combine
        x = torch.cat([loc_emb, user_emb, weekday_emb, hour_emb, temp_emb], dim=-1)
        x = self.dropout(x)
        
        # GRU
        gru_out, _ = self.gru(x)
        
        # Attention with residual
        mask = (loc_seq == 0)
        attn_out, _ = self.attention(gru_out, gru_out, gru_out, key_padding_mask=mask)
        x = self.norm1(gru_out + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        # Final representation
        if seq_len is not None:
            indices = (seq_len - 1).clamp(0, seq_length - 1).to(x.device)
            indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            final = torch.gather(x, 1, indices).squeeze(1)
        else:
            final = x[:, -1, :]
        
        # Output
        final = self.dropout(final)
        logits = self.fc_out(final)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
