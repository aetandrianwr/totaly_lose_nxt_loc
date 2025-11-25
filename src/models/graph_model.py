import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FrequencyAwareEmbedding(nn.Module):
    """Embeddings that encode both location identity and frequency"""
    def __init__(self, num_embeddings, embedding_dim, freq_weights=None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        
        # Frequency-based scaling
        if freq_weights is not None:
            self.register_buffer('freq_scale', freq_weights)
        else:
            self.register_buffer('freq_scale', torch.ones(num_embeddings))
    
    def forward(self, x):
        emb = self.embedding(x)
        # Scale by frequency to handle imbalance
        scale = self.freq_scale[x].unsqueeze(-1)
        return emb * torch.sqrt(scale.clamp(min=0.1, max=10.0))


class GraphAttentionLayer(nn.Module):
    """Graph attention for location relationships"""
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index=None):
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out)


class TemporalContextEncoder(nn.Module):
    """Encode temporal patterns with learnable positional embeddings"""
    def __init__(self, d_model, max_len=60):
        super().__init__()
        # Learnable temporal encodings
        self.temporal_emb = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        self.weekday_emb = nn.Embedding(8, d_model // 4)
        self.hour_emb = nn.Embedding(24, d_model // 4)
        
    def forward(self, weekday, hour, positions):
        B, L = weekday.shape
        
        # Position-based encoding
        pos_emb = self.temporal_emb[:L].unsqueeze(0).expand(B, -1, -1)
        
        # Weekday and hour
        wd_emb = self.weekday_emb(weekday)
        hr_emb = self.hour_emb(hour)
        
        return torch.cat([pos_emb, wd_emb, hr_emb], dim=-1)


class GraphTransitionPredictor(nn.Module):
    """
    Modern architecture combining:
    - Graph attention for location relationships
    - Frequency-aware embeddings
    - Temporal context encoding
    - Skip connections and layer normalization
    
    Goal: 40%+ test Acc@1
    """
    def __init__(
        self,
        num_locations=1187,
        num_users=46,
        embedding_dim=80,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.25,
        freq_weights=None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Frequency-aware location embeddings
        self.loc_embedding = FrequencyAwareEmbedding(
            num_locations, embedding_dim, freq_weights
        )
        
        # User embeddings (smaller for regularization)
        self.user_embedding = nn.Embedding(num_users, 24, padding_idx=0)
        
        # Temporal encoder
        self.temporal_encoder = TemporalContextEncoder(hidden_dim // 2)
        
        # Project to hidden dim
        input_dim = embedding_dim + 24 + hidden_dim // 2 + hidden_dim // 4
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Stack of graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # FFN after each layer (smaller)
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_layers)
        ])
        
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Final prediction head with additional layer
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_locations)
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len=None):
        B, L = loc_seq.shape
        
        # Embeddings
        loc_emb = self.loc_embedding(loc_seq)
        user_emb = self.user_embedding(user_seq)
        
        # Temporal encoding
        hour = (start_min_seq.float() / 60).long().clamp(0, 23)
        positions = torch.arange(L, device=loc_seq.device).unsqueeze(0).expand(B, -1)
        temp_emb = self.temporal_encoder(weekday_seq, hour, positions)
        
        # Combine
        x = torch.cat([loc_emb, user_emb, temp_emb], dim=-1)
        x = self.input_proj(x)
        x = self.dropout(x)
        
        # Stack of graph attention layers with residuals
        for i in range(len(self.gat_layers)):
            # Graph attention
            attn_out = self.gat_layers[i](x)
            x = self.norms[i](x + self.dropout(attn_out))
            
            # FFN
            ffn_out = self.ffns[i](x)
            x = self.ffn_norms[i](x + self.dropout(ffn_out))
        
        # Get final representation
        if seq_len is not None:
            indices = (seq_len - 1).clamp(0, L - 1).to(x.device)
            indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            final = torch.gather(x, 1, indices).squeeze(1)
        else:
            final = x[:, -1, :]
        
        # Predict
        logits = self.predictor(final)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
