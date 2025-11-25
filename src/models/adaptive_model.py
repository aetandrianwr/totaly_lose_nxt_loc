import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SequencePatternEncoder(nn.Module):
    """
    Encode sequences focusing on patterns rather than just embeddings.
    Uses n-gram style pattern detection.
    """
    def __init__(self, num_locations, pattern_dim=64):
        super().__init__()
        self.loc_emb = nn.Embedding(num_locations, pattern_dim)
        
        # Bigram and trigram pattern encoders
        self.bigram_conv = nn.Conv1d(pattern_dim, pattern_dim, kernel_size=2, padding=0)
        self.trigram_conv = nn.Conv1d(pattern_dim, pattern_dim, kernel_size=3, padding=1)
        
    def forward(self, loc_seq):
        # B, L
        emb = self.loc_emb(loc_seq)  # B, L, D
        emb_t = emb.transpose(1, 2)  # B, D, L
        
        bigram = F.gelu(self.bigram_conv(emb_t))
        trigram = F.gelu(self.trigram_conv(emb_t))
        
        # Match dimensions
        B, D, L = emb_t.shape
        bigram = F.pad(bigram, (0, L - bigram.size(2)))  # Pad to match length
        
        # Combine
        combined = (emb_t + bigram + trigram) / 3
        return combined.transpose(1, 2)  # B, L, D


class AdaptiveLocationPredictor(nn.Module):
    """
    Two-stage predictor:
    1. Pattern-based sequence encoding
    2. Adaptive prediction head that can generalize to unseen targets
    
    Key: Learn to predict based on sequence patterns, not just frequency
    """
    def __init__(
        self,
        num_locations=1187,
        num_users=46,
        embedding_dim=84,
        hidden_dim=136,
        num_heads=4,
        dropout=0.22,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Pattern-based encoding
        self.pattern_encoder = SequencePatternEncoder(num_locations, embedding_dim)
        
        # User context
        self.user_emb = nn.Embedding(num_users, 28, padding_idx=0)
        
        # Temporal
        self.weekday_emb = nn.Embedding(8, 14, padding_idx=0)
        self.hour_emb = nn.Embedding(24, 14)
        
        # Duration/diff encoding
        self.temp_mlp = nn.Sequential(
            nn.Linear(2, 24),
            nn.Dropout(dropout * 0.5),
            nn.GELU()
        )
        
        input_dim = embedding_dim + 28 + 14 + 14 + 24
        
        # Sequence encoder with local and global attention
        self.proj = nn.Linear(input_dim, hidden_dim)
        
        # Local attention (within sequence)
        self.local_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Global context
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Prediction head - learns to map patterns to locations
        # Uses deeper network to capture complex mappings
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_locations)
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.7)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.015)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len=None):
        B, L = loc_seq.shape
        
        # Pattern-based encoding
        pattern_emb = self.pattern_encoder(loc_seq)
        
        # Context
        user_emb = self.user_emb(user_seq)
        weekday_emb = self.weekday_emb(weekday_seq)
        
        hour = (start_min_seq.float() / 60).long().clamp(0, 23)
        hour_emb = self.hour_emb(hour)
        
        temp_feat = torch.stack([
            dur_seq / 300.0,
            diff_seq.float() / 5.0
        ], dim=-1)
        temp_emb = self.temp_mlp(temp_feat)
        
        # Combine all features
        x = torch.cat([pattern_emb, user_emb, weekday_emb, hour_emb, temp_emb], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        
        # Local attention
        mask = (loc_seq == 0)
        attn_out, _ = self.local_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        # Global pooling
        if seq_len is not None:
            # Mask out padding for pooling
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x_masked = x.masked_fill(mask_expanded, 0)
            seq_len_expanded = seq_len.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            
            # Average pool non-padding
            seq_len_float = seq_len.float().to(x.device)
            global_repr = x_masked.sum(dim=1) / seq_len_float.unsqueeze(-1).clamp(min=1)
        else:
            global_repr = x.mean(dim=1)
        
        global_repr = self.global_pool(global_repr)
        
        # Get last position
        if seq_len is not None:
            indices = (seq_len - 1).clamp(0, L - 1).to(x.device)
            indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            last_repr = torch.gather(x, 1, indices).squeeze(1)
        else:
            last_repr = x[:, -1, :]
        
        # Combine local and global
        combined = self.norm3(last_repr + global_repr)
        combined = self.dropout(combined)
        
        # Predict - maps sequence pattern to next location
        logits = self.predictor(combined)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
