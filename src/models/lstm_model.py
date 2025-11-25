import torch
import torch.nn as nn
import math


class LSTMNextLocPredictor(nn.Module):
    """
    LSTM-based next location predictor with attention mechanism.
    """
    def __init__(
        self,
        num_locations=1187,
        num_users=46,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embeddings
        self.loc_embedding = nn.Embedding(num_locations, embedding_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users, 32, padding_idx=0)
        self.weekday_embedding = nn.Embedding(8, 16, padding_idx=0)
        
        # Temporal feature projections
        self.time_projection = nn.Linear(1, 16)
        self.duration_projection = nn.Linear(1, 16)
        self.diff_projection = nn.Linear(1, 16)
        
        # Input dimension
        input_dim = embedding_dim + 32 + 16 * 4
        
        # LSTM
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Output
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, num_locations)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len=None):
        batch_size = loc_seq.size(0)
        
        # Embeddings
        loc_emb = self.loc_embedding(loc_seq)
        user_emb = self.user_embedding(user_seq)
        weekday_emb = self.weekday_embedding(weekday_seq)
        
        # Temporal features
        time_emb = self.time_projection(start_min_seq.unsqueeze(-1).float() / 1440.0)
        dur_emb = self.duration_projection(dur_seq.unsqueeze(-1) / 500.0)
        diff_emb = self.diff_projection(diff_seq.unsqueeze(-1).float() / 7.0)
        
        # Combine
        x = torch.cat([loc_emb, user_emb, weekday_emb, time_emb, dur_emb, diff_emb], dim=-1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Output
        output = self.dropout(context)
        logits = self.output_layer(output)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
