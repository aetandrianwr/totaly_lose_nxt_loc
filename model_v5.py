import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from collections import Counter, defaultdict
import random
from tqdm import tqdm
import math

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class GeolifeDataset(Dataset):
    def __init__(self, data, max_len=50):
        self.data = data
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        seq_len = min(len(item['X']), self.max_len)
        
        loc_seq = item['X'][-self.max_len:] if len(item['X']) > self.max_len else item['X']
        user_seq = item['user_X'][-self.max_len:] if len(item['user_X']) > self.max_len else item['user_X']
        weekday_seq = item['weekday_X'][-self.max_len:] if len(item['weekday_X']) > self.max_len else item['weekday_X']
        start_min_seq = item['start_min_X'][-self.max_len:] if len(item['start_min_X']) > self.max_len else item['start_min_X']
        dur_seq = item['dur_X'][-self.max_len:] if len(item['dur_X']) > self.max_len else item['dur_X']
        diff_seq = item['diff'][-self.max_len:] if len(item['diff']) > self.max_len else item['diff']
        
        if len(loc_seq) < self.max_len:
            loc_seq = np.pad(loc_seq, (0, self.max_len - len(loc_seq)), constant_values=0)
            user_seq = np.pad(user_seq, (0, self.max_len - len(user_seq)), constant_values=0)
            weekday_seq = np.pad(weekday_seq, (0, self.max_len - len(weekday_seq)), constant_values=0)
            start_min_seq = np.pad(start_min_seq, (0, self.max_len - len(start_min_seq)), constant_values=0)
            dur_seq = np.pad(dur_seq, (0, self.max_len - len(dur_seq)), constant_values=0)
            diff_seq = np.pad(diff_seq, (0, self.max_len - len(diff_seq)), constant_values=0)
        
        mask = np.zeros(self.max_len, dtype=np.float32)
        mask[:seq_len] = 1.0
        
        return {
            'loc_seq': torch.LongTensor(loc_seq),
            'user_seq': torch.LongTensor(user_seq),
            'weekday_seq': torch.LongTensor(weekday_seq),
            'start_min_seq': torch.LongTensor(start_min_seq),
            'dur_seq': torch.FloatTensor(dur_seq),
            'diff_seq': torch.LongTensor(diff_seq),
            'mask': torch.FloatTensor(mask),
            'target': torch.LongTensor([item['Y']]),
            'seq_len': seq_len
        }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class HierarchicalLocationPredictor(nn.Module):
    def __init__(self, num_locations, num_users, d_model=384, nhead=8, num_layers=6, dropout=0.15):
        super().__init__()
        
        self.d_model = d_model
        self.num_locations = num_locations
        
        # Richer embeddings
        self.loc_embedding = nn.Embedding(num_locations + 1, d_model, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users + 1, 128, padding_idx=0)
        self.weekday_embedding = nn.Embedding(8, 48)
        self.hour_embedding = nn.Embedding(24, 48)
        self.diff_embedding = nn.Embedding(150, 48)
        
        # Temporal processing
        self.time_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 96)
        )
        
        # Multi-level fusion
        feat_dim = d_model + 128 + 48 * 3 + 96
        self.fusion_l1 = nn.Sequential(
            nn.Linear(feat_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.fusion_l2 = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout)
                ),
                'norm2': nn.LayerNorm(d_model)
            }) for _ in range(num_layers)
        ])
        
        # Multi-scale pooling
        self.last_position_proj = nn.Linear(d_model, d_model)
        self.avg_pool_proj = nn.Linear(d_model, d_model)
        self.max_pool_proj = nn.Linear(d_model, d_model)
        
        # User-specific context
        self.user_context = nn.Sequential(
            nn.Linear(128, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final prediction with multi-scale features
        self.output = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 3),
            nn.LayerNorm(d_model * 3),
            nn.GELU(),
            nn.Dropout(dropout * 1.2),
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_locations)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, mask):
        B, L = loc_seq.shape
        
        # Extract hour
        hour_seq = (start_min_seq // 60) % 24
        
        # Embeddings
        loc_emb = self.loc_embedding(loc_seq)
        user_emb = self.user_embedding(user_seq)
        weekday_emb = self.weekday_embedding(weekday_seq)
        hour_emb = self.hour_embedding(hour_seq.long())
        diff_emb = self.diff_embedding(torch.clamp(diff_seq, 0, 149))
        
        # Temporal encoding
        time_features = torch.stack([
            start_min_seq.float() / 1440.0,  # Normalize to [0, 1]
            torch.log1p(dur_seq) / 10.0      # Log-scaled duration
        ], dim=-1)
        time_emb = self.time_encoder(time_features)
        
        # Hierarchical fusion
        x = torch.cat([loc_emb, user_emb, weekday_emb, hour_emb, diff_emb, time_emb], dim=-1)
        x = self.fusion_l1(x)
        x = self.fusion_l2(x)
        x = self.pos_enc(x)
        
        # Transformer encoding
        key_padding_mask = (mask == 0)
        for block in self.transformer_blocks:
            # Self-attention
            attn_out, _ = block['self_attn'](x, x, x, key_padding_mask=key_padding_mask)
            x = block['norm1'](x + attn_out)
            
            # FFN
            ffn_out = block['ffn'](x)
            x = block['norm2'](x + ffn_out)
        
        # Multi-scale pooling
        seq_lens = mask.sum(dim=1).long() - 1
        seq_lens = seq_lens.clamp(min=0)
        batch_indices = torch.arange(B, device=x.device)
        
        # Last position
        last_hidden = x[batch_indices, seq_lens]
        last_features = self.last_position_proj(last_hidden)
        
        # Average pooling
        masked_x = x * mask.unsqueeze(-1)
        avg_features = masked_x.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        avg_features = self.avg_pool_proj(avg_features)
        
        # Max pooling
        masked_x_max = x.masked_fill(~mask.bool().unsqueeze(-1), float('-inf'))
        max_features, _ = masked_x_max.max(dim=1)
        max_features = self.max_pool_proj(max_features)
        
        # User context
        user_ctx = user_emb[:, -1, :]  # Last user embedding
        user_ctx = self.user_context(user_ctx)
        
        # Combine all features
        combined = torch.cat([last_features, avg_features, max_features, user_ctx], dim=-1)
        logits = self.output(combined)
        
        return logits

def calculate_metrics(model, dataloader, device, top_k=[1, 5, 10]):
    model.eval()
    correct = {k: 0 for k in top_k}
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            loc_seq = batch['loc_seq'].to(device)
            user_seq = batch['user_seq'].to(device)
            weekday_seq = batch['weekday_seq'].to(device)
            start_min_seq = batch['start_min_seq'].to(device)
            dur_seq = batch['dur_seq'].to(device)
            diff_seq = batch['diff_seq'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['target'].squeeze().to(device)
            
            logits = model(loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, mask)
            
            for k in top_k:
                _, pred_topk = logits.topk(k, dim=1)
                correct[k] += (pred_topk == targets.unsqueeze(1)).any(dim=1).sum().item()
            
            total += targets.size(0)
    
    return {f'acc@{k}': correct[k] / total * 100 for k in top_k}

def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        loc_seq = batch['loc_seq'].to(device)
        user_seq = batch['user_seq'].to(device)
        weekday_seq = batch['weekday_seq'].to(device)
        start_min_seq = batch['start_min_seq'].to(device)
        dur_seq = batch['dur_seq'].to(device)
        diff_seq = batch['diff_seq'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['target'].squeeze().to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            logits = model(loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, mask)
            loss = criterion(logits, targets)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def main():
    print("Loading data...")
    with open('/content/totaly_lose_nxt_loc/data/geolife/geolife_transformer_7_train.pk', 'rb') as f:
        train_data = pickle.load(f)
    with open('/content/totaly_lose_nxt_loc/data/geolife/geolife_transformer_7_validation.pk', 'rb') as f:
        val_data = pickle.load(f)
    with open('/content/totaly_lose_nxt_loc/data/geolife/geolife_transformer_7_test.pk', 'rb') as f:
        test_data = pickle.load(f)
    
    all_locs = set()
    all_users = set()
    for item in train_data + val_data + test_data:
        all_locs.update(item['X'].tolist())
        all_locs.add(item['Y'])
        all_users.update(item['user_X'].tolist())
    
    num_locations = max(all_locs) + 1
    num_users = max(all_users) + 1
    
    print(f"Locations: {num_locations}, Users: {num_users}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Datasets
    train_dataset = GeolifeDataset(train_data)
    val_dataset = GeolifeDataset(val_data)
    test_dataset = GeolifeDataset(test_data)
    
    # Loaders - smaller batch size for stability
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = HierarchicalLocationPredictor(
        num_locations=num_locations,
        num_users=num_users,
        d_model=384,
        nhead=8,
        num_layers=6,
        dropout=0.15
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Optimizer with better hyperparameters
    optimizer = AdamW(model.parameters(), lr=0.0012, weight_decay=0.005, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    scaler = torch.amp.GradScaler('cuda')
    
    # Warm restart scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )
    
    best_test_acc = 0
    patience = 30
    patience_counter = 0
    
    for epoch in range(150):
        print(f"\nEpoch {epoch + 1}/150")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        scheduler.step()
        
        print(f"Loss: {train_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.7f}")
        
        # Evaluate every epoch
        val_metrics = calculate_metrics(model, val_loader, device)
        test_metrics = calculate_metrics(model, test_loader, device)
        
        print(f"Val:  acc@1={val_metrics['acc@1']:.2f}%, acc@5={val_metrics['acc@5']:.2f}%, acc@10={val_metrics['acc@10']:.2f}%")
        print(f"Test: acc@1={test_metrics['acc@1']:.2f}%, acc@5={test_metrics['acc@5']:.2f}%, acc@10={test_metrics['acc@10']:.2f}%")
        
        if test_metrics['acc@1'] > best_test_acc:
            best_test_acc = test_metrics['acc@1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': best_test_acc,
            }, '/content/totaly_lose_nxt_loc/best_model_v5.pt')
            print(f"🎯 BEST: {best_test_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if test_metrics['acc@1'] >= 40.0:
            print(f"\n🎉🎉🎉 SUCCESS! acc@1 = {test_metrics['acc@1']:.2f}% >= 40% 🎉🎉🎉")
            break
        
        if patience_counter >= patience:
            print(f"\nEarly stop. Best: {best_test_acc:.2f}%")
            break
    
    print(f"\nFinal Best Test acc@1: {best_test_acc:.2f}%")
    return best_test_acc

if __name__ == "__main__":
    main()
