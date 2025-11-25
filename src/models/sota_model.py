"""
State-of-the-art hybrid model combining proven techniques:
- Mixture of Experts (Google)
- Cosine similarity prediction (Meta)
- Multi-head prediction ensemble
- Adaptive temperature scaling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MixtureOfExpertsLayer(nn.Module):
    """Sparse Mixture of Experts - Google Brain technique"""
    def __init__(self, dim, num_experts=3, top_k=2):
        super().__init__()
        expert_dim = dim  # Same as input dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router network
        self.router = nn.Linear(dim, num_experts, bias=False)
        
        # Expert networks - lightweight
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        B, L, D = x.shape
        
        # Route to top-k experts
        router_logits = self.router(x)  # B, L, num_experts
        router_weights = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(router_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Apply experts
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, :, i]
            weights = top_k_weights[:, :, i].unsqueeze(-1)
            
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += weights[mask] * expert_output
        
        return output


class MultiHeadPrediction(nn.Module):
    """Multiple prediction heads with ensemble - DeepMind technique"""
    def __init__(self, dim, num_classes, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        
        self.heads = nn.ModuleList([
            nn.Linear(dim, num_classes) for _ in range(num_heads)
        ])
        
        # Learnable weights for combining heads
        self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
        
    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        
        # Weighted ensemble
        weights = F.softmax(self.head_weights, dim=0)
        combined = sum(w * out for w, out in zip(weights, outputs))
        
        return combined


class AdaptiveTemperatureScaling(nn.Module):
    """Learned temperature for calibration - Cornell/Google"""
    def __init__(self, initial_temp=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temp))
        
    def forward(self, logits):
        return logits / torch.clamp(self.temperature, min=0.1, max=5.0)


class SOTALocationPredictor(nn.Module):
    """
    State-of-the-art model combining proven techniques:
    - Transformer with MoE
    - Cosine similarity prediction
    - Multi-head ensemble
    - Adaptive temperature
    
    Target: 40%+ Acc@1 with <500K params
    """
    def __init__(
        self,
        num_locations=1187,
        num_users=46,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        num_experts=3,
        dropout=0.2,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_locations = num_locations
        
        # Embeddings with larger capacity
        self.loc_embedding = nn.Embedding(num_locations, embedding_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users, 24, padding_idx=0)
        
        # Temporal embeddings
        self.weekday_emb = nn.Embedding(8, 12, padding_idx=0)
        self.hour_emb = nn.Embedding(25, 12)  # 0-23 + 1 for padding
        
        # Project to common dimension
        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim + 24 + 24 + 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout * 0.5)
        )
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 60, embedding_dim) * 0.02)
        
        # Transformer layers with MoE
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True),
                'norm1': nn.LayerNorm(embedding_dim),
                'moe': MixtureOfExpertsLayer(embedding_dim, num_experts=num_experts, top_k=2),
                'norm2': nn.LayerNorm(embedding_dim),
            }))
        
        self.final_norm = nn.LayerNorm(embedding_dim)
        
        # Cosine similarity-based prediction (Meta technique)
        # Location prototypes for similarity matching
        self.location_prototypes = nn.Parameter(torch.randn(num_locations, embedding_dim))
        nn.init.xavier_uniform_(self.location_prototypes, gain=1.0)
        
        # Multi-head prediction ensemble
        self.multi_head_pred = MultiHeadPrediction(embedding_dim, num_locations, num_heads=2)
        
        # Temperature scaling for calibration
        self.temperature = AdaptiveTemperatureScaling(initial_temp=1.5)
        
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
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len=None):
        B, L = loc_seq.shape
        device = loc_seq.device
        
        # Embeddings
        loc_emb = self.loc_embedding(loc_seq)
        user_emb = self.user_embedding(user_seq[:, 0]).unsqueeze(1).expand(-1, L, -1)
        
        # Temporal features
        weekday_emb = self.weekday_emb(weekday_seq)
        hour = (start_min_seq.float() / 60).long().clamp(0, 23)
        hour_emb = self.hour_emb(hour)
        temporal_emb = torch.cat([weekday_emb, hour_emb], dim=-1)
        
        # Continuous features
        cont_feat = torch.stack([
            diff_seq.float() / 10.0,
            dur_seq.float() / 300.0
        ], dim=-1)
        
        # Combine
        x = torch.cat([loc_emb, user_emb, temporal_emb, cont_feat], dim=-1)
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pos_embedding[:, :L, :]
        x = self.dropout(x)
        
        # Transformer with MoE
        mask = (loc_seq == 0)
        for layer in self.layers:
            # Self-attention
            attn_out, _ = layer['attn'](x, x, x, key_padding_mask=mask)
            x = layer['norm1'](x + attn_out)
            
            # MoE feedforward
            moe_out = layer['moe'](x)
            x = layer['norm2'](x + moe_out)
        
        x = self.final_norm(x)
        
        # Get final representation
        if seq_len is not None:
            indices = (seq_len - 1).clamp(0, L - 1).to(device)
            indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.embedding_dim)
            final = torch.gather(x, 1, indices).squeeze(1)
        else:
            final = x[:, -1, :]
        
        # Hybrid prediction: Cosine similarity + Multi-head
        
        # 1. Cosine similarity with location prototypes (Meta approach)
        final_norm = F.normalize(final, p=2, dim=-1)
        proto_norm = F.normalize(self.location_prototypes, p=2, dim=-1)
        cosine_logits = torch.matmul(final_norm, proto_norm.t()) * 10.0  # Scale factor
        
        # 2. Multi-head prediction
        multi_head_logits = self.multi_head_pred(final)
        
        # 3. Combine both approaches
        logits = 0.6 * cosine_logits + 0.4 * multi_head_logits
        
        # 4. Temperature scaling for calibration
        logits = self.temperature(logits)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
