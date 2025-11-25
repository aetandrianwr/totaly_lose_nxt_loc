import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss to handle class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LocationTransitionModel(nn.Module):
    """
    Focused model for location prediction with:
    - Location transition patterns
    - Frequency-aware embeddings
    - Simple but effective architecture
    
    Target: >40% Acc@1 with <500K params
    """
    def __init__(
        self,
        num_locations=1187,
        num_users=46,
        embedding_dim=80,
        hidden_dim=144,
        dropout=0.25,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Location embedding - main signal
        self.loc_embedding = nn.Embedding(num_locations, embedding_dim, padding_idx=0)
        
        # User-specific location preferences (factorized)
        self.user_embedding = nn.Embedding(num_users, 32, padding_idx=0)
        self.user_loc_proj = nn.Linear(32, embedding_dim // 4)
        
        # Temporal context (minimal)
        self.temporal_proj = nn.Linear(2, 32)  # weekday + hour
        
        # Main sequence encoder - GRU is parameter efficient
        self.encoder = nn.GRU(
            embedding_dim + embedding_dim // 4 + 32,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False
        )
        
        # Location transition modeling
        self.transition_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, num_locations)
        )
        
        # Location bias (frequency-aware)
        self.location_bias = nn.Parameter(torch.zeros(num_locations))
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'norm' not in name:
                if 'embedding' in name:
                    nn.init.normal_(param, std=0.02)
                elif 'gru' in name or 'rnn' in name:
                    nn.init.orthogonal_(param)
                elif param.dim() >= 2:
                    nn.init.xavier_uniform_(param, gain=0.8)
            elif 'bias' in name and 'location_bias' not in name:
                nn.init.zeros_(param)
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len=None):
        batch_size, seq_length = loc_seq.size()
        
        # Location embeddings
        loc_emb = self.loc_embedding(loc_seq)
        
        # User-location interaction
        user_emb = self.user_embedding(user_seq)
        user_loc_emb = self.user_loc_proj(user_emb)
        
        # Temporal features (simplified)
        hour = (start_min_seq.float() / 60).long().clamp(0, 23)
        temporal_features = torch.stack([
            weekday_seq.float() / 7.0,
            hour.float() / 24.0
        ], dim=-1)
        temporal_emb = F.gelu(self.temporal_proj(temporal_features))
        
        # Combine features
        x = torch.cat([loc_emb, user_loc_emb, temporal_emb], dim=-1)
        x = self.dropout(x)
        
        # Encode sequence
        encoded, hidden = self.encoder(x)
        
        # Self-attention to focus on relevant past locations
        mask = (loc_seq == 0)
        attn_out, _ = self.transition_attn(
            encoded, encoded, encoded,
            key_padding_mask=mask
        )
        
        # Combine encoder and attention
        combined = encoded + attn_out
        
        # Get final representation
        if seq_len is not None:
            indices = (seq_len - 1).clamp(0, seq_length - 1).to(combined.device)
            indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            final_repr = torch.gather(combined, 1, indices).squeeze(1)
        else:
            final_repr = combined[:, -1, :]
        
        # Predict with location bias
        logits = self.predictor(final_repr)
        logits = logits + self.location_bias
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
