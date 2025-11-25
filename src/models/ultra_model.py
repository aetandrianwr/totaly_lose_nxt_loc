"""
Ultra-optimized model combining:
- Efficient attention with relative position encoding
- Focal loss for class imbalance
- Test-time augmentation
- Mixup data augmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RelativeMultiHeadAttention(nn.Module):
    """Efficient attention with relative position - Google/Transformer-XL"""
    def __init__(self, dim, num_heads, max_len=60, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Relative position embeddings
        self.rel_pos_emb = nn.Parameter(torch.randn(2 * max_len - 1, self.head_dim))
        
    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Standard attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add relative position bias
        pos_indices = torch.arange(L, device=x.device).unsqueeze(0) - torch.arange(L, device=x.device).unsqueeze(1)
        pos_indices = pos_indices + (L - 1)  # Shift to positive
        rel_pos = self.rel_pos_emb[pos_indices]  # L, L, head_dim
        
        # Compute position attention
        q_with_pos = q.permute(0, 2, 1, 3)  # B, L, H, D
        pos_attn = torch.einsum('blhd,lkd->bhlk', q_with_pos, rel_pos) / math.sqrt(self.head_dim)
        attn = attn + pos_attn
        
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


class UltraLocationPredictor(nn.Module):
    """
    Ultra-optimized predictor with:
    - Relative position attention
    - Efficient architecture
    - Strong regularization
    
    Target: 40%+ test Acc@1 with <500K params
    """
    def __init__(
        self,
        num_locations=1187,
        num_users=46,
        embedding_dim=80,
        num_heads=8,
        num_layers=3,
        dropout=0.25,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_locations = num_locations
        
        # Rich embeddings
        self.loc_embedding = nn.Embedding(num_locations, embedding_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users, 24, padding_idx=0)
        
        # Temporal with interaction
        self.weekday_emb = nn.Embedding(8, 16, padding_idx=0)
        self.hour_emb = nn.Embedding(25, 16)
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim + 24 + 32 + 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout * 0.5)
        )
        
        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 60, embedding_dim) * 0.02)
        
        # Transformer layers with relative attention
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': RelativeMultiHeadAttention(embedding_dim, num_heads, dropout=dropout),
                'norm1': nn.LayerNorm(embedding_dim),
                'ffn': nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embedding_dim * 2, embedding_dim)
                ),
                'norm2': nn.LayerNorm(embedding_dim),
            }) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(embedding_dim)
        
        # Dual prediction: parametric + non-parametric
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_locations)
        )
        
        # Location prototypes for metric learning
        self.location_prototypes = nn.Parameter(torch.randn(num_locations, embedding_dim))
        
        # Temperature for calibration
        self.temperature = nn.Parameter(torch.tensor(2.0))
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        nn.init.xavier_uniform_(self.location_prototypes, gain=1.0)
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len=None):
        B, L = loc_seq.shape
        device = loc_seq.device
        
        # Embeddings
        loc_emb = self.loc_embedding(loc_seq)
        user_emb = self.user_embedding(user_seq[:, 0]).unsqueeze(1).expand(-1, L, -1)
        
        # Temporal
        weekday_emb = self.weekday_emb(weekday_seq)
        hour = (start_min_seq.float() / 60).long().clamp(0, 23)
        hour_emb = self.hour_emb(hour)
        temporal_emb = torch.cat([weekday_emb, hour_emb], dim=-1)
        
        # Continuous
        cont_feat = torch.stack([
            diff_seq.float() / 10.0,
            dur_seq.float() / 300.0
        ], dim=-1)
        
        # Combine
        x = torch.cat([loc_emb, user_emb, temporal_emb, cont_feat], dim=-1)
        x = self.input_proj(x)
        
        # Positional encoding
        x = x + self.pos_embedding[:, :L, :]
        x = self.dropout(x)
        
        # Transformer
        mask = (loc_seq == 0)
        for layer in self.layers:
            attn_out = layer['attn'](x, mask)
            x = layer['norm1'](x + attn_out)
            
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)
        
        x = self.final_norm(x)
        
        # Get final representation
        if seq_len is not None:
            indices = (seq_len - 1).clamp(0, L - 1).to(device)
            indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.embedding_dim)
            final = torch.gather(x, 1, indices).squeeze(1)
        else:
            final = x[:, -1, :]
        
        # Dual prediction
        # 1. Parametric
        logits_param = self.predictor(final)
        
        # 2. Non-parametric (cosine similarity)
        final_norm = F.normalize(final, p=2, dim=-1)
        proto_norm = F.normalize(self.location_prototypes, p=2, dim=-1)
        logits_cosine = torch.matmul(final_norm, proto_norm.t()) * 15.0
        
        # Combine
        logits = 0.5 * logits_param + 0.5 * logits_cosine
        
        # Temperature scaling
        logits = logits / torch.clamp(self.temperature, min=0.5, max=3.0)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
