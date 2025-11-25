import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HierarchicalLocationEmbedding(nn.Module):
    """
    Two-level spatial representation with soft clustering.
    Captures both specific locations and spatial regions.
    """
    def __init__(self, num_locations, embedding_dim, num_clusters=50):
        super().__init__()
        self.num_locations = num_locations
        self.num_clusters = num_clusters
        
        # Fine-grained location embeddings
        self.location_emb = nn.Embedding(num_locations, embedding_dim, padding_idx=0)
        
        # Coarse-grained cluster embeddings
        self.cluster_emb = nn.Embedding(num_clusters, embedding_dim // 2)
        
        # Learnable soft assignment: location -> cluster distribution
        self.loc_to_cluster = nn.Linear(num_locations, num_clusters)
        
    def forward(self, loc_ids):
        # Fine-grained embeddings
        fine = self.location_emb(loc_ids)  # B, L, D
        
        # Soft cluster assignment
        B, L = loc_ids.shape
        loc_onehot = F.one_hot(loc_ids, self.num_locations).float()  # B, L, num_locs
        cluster_weights = F.softmax(self.loc_to_cluster(loc_onehot), dim=-1)  # B, L, num_clusters
        
        # Weighted cluster embeddings
        cluster_ids = torch.arange(self.num_clusters, device=loc_ids.device)
        coarse = torch.matmul(cluster_weights, self.cluster_emb.weight)  # B, L, D/2
        
        # Combine fine and coarse
        combined = torch.cat([fine, coarse], dim=-1)  # B, L, D + D/2
        
        return combined


class EnhancedTemporalEmbedding(nn.Module):
    """
    MLP-based interaction modelling between temporal features.
    Captures non-linear temporal dependencies.
    """
    def __init__(self, output_dim):
        super().__init__()
        self.weekday_emb = nn.Embedding(8, 16, padding_idx=0)
        self.hour_emb = nn.Embedding(24, 16)
        
        # MLP for temporal interactions
        self.temporal_mlp = nn.Sequential(
            nn.Linear(32 + 2, 64),  # weekday + hour + continuous (minute, duration)
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, weekday, start_min, dur):
        B, L = weekday.shape
        
        weekday_emb = self.weekday_emb(weekday)  # B, L, 16
        hour = (start_min.float() / 60).long().clamp(0, 23)
        hour_emb = self.hour_emb(hour)  # B, L, 16
        
        # Continuous temporal features
        minute_norm = (start_min % 60).float() / 60.0  # B, L
        dur_norm = dur.float() / 300.0  # B, L
        
        # Combine all temporal info
        temporal_cat = torch.cat([
            weekday_emb, 
            hour_emb, 
            minute_norm.unsqueeze(-1), 
            dur_norm.unsqueeze(-1)
        ], dim=-1)  # B, L, 34
        
        return self.temporal_mlp(temporal_cat)


class MultiScaleTemporalAttention(nn.Module):
    """
    Separates short-range and long-range attention.
    Short-range: exponential decay bias for recent history
    Long-range: periodic patterns without recency bias
    """
    def __init__(self, dim, num_heads=4, dropout=0.1, max_len=60):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Short-range attention (local)
        self.short_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # Long-range attention (global periodic)
        self.long_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # Exponential decay bias for short-range
        decay_bias = torch.zeros(max_len, max_len)
        for i in range(max_len):
            for j in range(max_len):
                if j <= i:
                    decay_bias[i, j] = math.exp(-(i - j) / 5.0)  # Recent positions more important
        self.register_buffer('decay_bias', decay_bias)
        
        # Gated fusion of short and long range
        self.gate = nn.Linear(dim * 2, dim)
        
    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        # Short-range attention with recency bias
        attn_mask_short = self.decay_bias[:L, :L].unsqueeze(0).expand(B * self.num_heads, -1, -1)
        attn_mask_short = attn_mask_short.masked_fill(attn_mask_short == 0, float('-inf'))
        attn_mask_short = attn_mask_short.masked_fill(attn_mask_short != float('-inf'), 0)
        
        short_out, _ = self.short_attn(x, x, x, key_padding_mask=mask, attn_mask=attn_mask_short)
        
        # Long-range attention without recency bias
        long_out, _ = self.long_attn(x, x, x, key_padding_mask=mask)
        
        # Gated fusion
        combined = torch.cat([short_out, long_out], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        output = gate * short_out + (1 - gate) * long_out
        
        return output


class CrossAttentionFusion(nn.Module):
    """
    Learned attention weights for cross-stream information flow.
    Enables selective focus on relevant historical information.
    """
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, query, key_value, mask=None):
        """
        query: reasoning state (B, L, D)
        key_value: historical context (B, L, D)
        """
        attn_out, attn_weights = self.cross_attn(query, key_value, key_value, key_padding_mask=mask)
        output = self.norm(query + attn_out)
        return output, attn_weights


class GatedFusionMechanism(nn.Module):
    """
    GRU-inspired gating for adaptive information integration.
    """
    def __init__(self, dim):
        super().__init__()
        self.reset_gate = nn.Linear(dim * 2, dim)
        self.update_gate = nn.Linear(dim * 2, dim)
        self.candidate = nn.Linear(dim * 2, dim)
        
    def forward(self, x1, x2):
        """Fuse two information sources with learned gates"""
        combined = torch.cat([x1, x2], dim=-1)
        
        r = torch.sigmoid(self.reset_gate(combined))
        z = torch.sigmoid(self.update_gate(combined))
        
        reset_combined = torch.cat([r * x1, x2], dim=-1)
        h_tilde = torch.tanh(self.candidate(reset_combined))
        
        output = (1 - z) * x1 + z * h_tilde
        return output


class AdvancedTransformerBlock(nn.Module):
    """Enhanced transformer block with all improvements"""
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.multi_scale_attn = MultiScaleTemporalAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        # FFN with gating
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        self.gated_fusion = GatedFusionMechanism(dim)
        
    def forward(self, x, mask=None):
        # Multi-scale attention
        attn_out = self.multi_scale_attn(self.norm1(x), mask)
        x = x + attn_out
        
        # FFN with gated fusion
        ffn_out = self.ffn(self.norm2(x))
        x = self.gated_fusion(x, ffn_out)
        
        return x


class AdvancedLocationPredictor(nn.Module):
    """
    Advanced architecture with:
    - Multi-scale temporal attention
    - Hierarchical location embeddings
    - Cross-attention mechanism
    - Gated fusion
    - Enhanced temporal modelling
    
    Target: >40% Acc@1 with <500K params
    """
    def __init__(
        self,
        num_locations=1187,
        num_users=46,
        embedding_dim=68,
        num_clusters=30,
        num_heads=4,
        num_layers=2,
        dropout=0.15,
        val_prior=None
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_locations = num_locations
        
        # Hierarchical location embeddings
        self.loc_embedding = HierarchicalLocationEmbedding(
            num_locations, embedding_dim, num_clusters
        )
        loc_output_dim = embedding_dim + embedding_dim // 2
        
        # User embedding
        self.user_embedding = nn.Embedding(num_users, 20, padding_idx=0)
        
        # Enhanced temporal embedding
        self.temporal_embedding = EnhancedTemporalEmbedding(embedding_dim)
        
        # Continuous features
        self.cont_proj = nn.Linear(2, 20)
        
        # Project all to common dimension
        total_dim = loc_output_dim + 20 + embedding_dim + 20
        self.input_proj = nn.Sequential(
            nn.Linear(total_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout * 0.5)
        )
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 60, embedding_dim) * 0.02)
        
        # Advanced transformer blocks
        self.blocks = nn.ModuleList([
            AdvancedTransformerBlock(embedding_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Cross-attention for reasoning
        self.cross_attn = CrossAttentionFusion(embedding_dim, num_heads, dropout)
        
        # Final gated fusion
        self.final_fusion = GatedFusionMechanism(embedding_dim)
        
        self.norm = nn.LayerNorm(embedding_dim)
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_locations)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Validation prior for test-time calibration
        if val_prior is not None:
            self.register_buffer('val_prior', val_prior)
        else:
            self.register_buffer('val_prior', torch.zeros(num_locations))
        
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
        
        # Hierarchical location embeddings
        loc_emb = self.loc_embedding(loc_seq)  # B, L, D + D/2
        
        # User embedding (broadcast)
        user_emb = self.user_embedding(user_seq[:, 0]).unsqueeze(1).expand(-1, L, -1)
        
        # Enhanced temporal embedding
        temporal_emb = self.temporal_embedding(weekday_seq, start_min_seq, dur_seq)
        
        # Continuous features
        cont_features = torch.stack([
            diff_seq.float() / 10.0,
            (dur_seq.float() / 300.0).clamp(0, 5)
        ], dim=-1)
        cont_emb = F.gelu(self.cont_proj(cont_features))
        
        # Combine all features
        combined = torch.cat([loc_emb, user_emb, temporal_emb, cont_emb], dim=-1)
        x = self.input_proj(combined)
        
        # Add positional encoding
        x = x + self.pos_embedding[:, :L, :]
        x = self.dropout(x)
        
        # Multi-scale transformer blocks
        mask = (loc_seq == 0)
        for block in self.blocks:
            x = block(x, mask)
        
        # Cross-attention for reasoning
        reasoning_state, _ = self.cross_attn(x, x, mask)
        
        # Gated fusion of reasoning and context
        x = self.final_fusion(x, reasoning_state)
        x = self.norm(x)
        
        # Get final representation
        if seq_len is not None:
            indices = (seq_len - 1).clamp(0, L - 1).to(device)
            indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.embedding_dim)
            final = torch.gather(x, 1, indices).squeeze(1)
        else:
            final = x[:, -1, :]
        
        # Predict
        logits = self.predictor(final)
        
        # Add validation prior if available for calibration
        if self.training == False and self.val_prior.sum() > 0:
            logits = logits + self.val_prior
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
