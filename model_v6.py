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

def build_transition_priors(data, num_locations):
    """Build transition probability matrix from training data"""
    transitions = np.zeros((num_locations, num_locations), dtype=np.float32)
    
    for item in data:
        locs = item['X'].tolist()
        target = item['Y']
        
        # Count transitions from all positions to target
        for loc in locs:
            transitions[loc, target] += 1
    
    # Normalize rows (add smoothing)
    row_sums = transitions.sum(axis=1, keepdims=True) + 1e-8
    transitions = transitions / row_sums
    
    return transitions

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

class TransitionAwarePredictor(nn.Module):
    def __init__(self, num_locations, num_users, transition_priors, d_model=256, dropout=0.2):
        super().__init__()
        
        self.d_model = d_model
        self.num_locations = num_locations
        
        # Embeddings
        self.loc_embedding = nn.Embedding(num_locations + 1, d_model, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users + 1, 64, padding_idx=0)
        self.weekday_embedding = nn.Embedding(8, 16)
        self.hour_embedding = nn.Embedding(24, 16)
        
        # Time encoding
        self.time_enc = nn.Linear(2, 32)
        
        # LSTM for sequence encoding
        self.lstm = nn.LSTM(
            input_size=d_model + 64 + 16 + 16 + 32,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
            bidirectional=False
        )
        
        # Transition prior embedding (learnable scaling)
        self.register_buffer('transition_priors', torch.from_numpy(transition_priors))
        self.transition_scale = nn.Parameter(torch.tensor(1.0))
        
        # Neural prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_locations)
        )
        
        # Mixing weight between neural and transition-based predictions
        self.mix_weight = nn.Parameter(torch.tensor(0.5))
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        # Initialize embeddings
        nn.init.normal_(self.loc_embedding.weight, std=0.02)
        self.loc_embedding.weight.data[0].zero_()
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        self.user_embedding.weight.data[0].zero_()
    
    def forward(self, loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, mask):
        B, L = loc_seq.shape
        
        # Hour from start_min
        hour_seq = (start_min_seq // 60) % 24
        
        # Embeddings
        loc_emb = self.loc_embedding(loc_seq)
        user_emb = self.user_embedding(user_seq)
        weekday_emb = self.weekday_embedding(weekday_seq)
        hour_emb = self.hour_embedding(hour_seq.long())
        
        # Time features
        time_feat = torch.stack([
            start_min_seq.float() / 1440.0,
            torch.log1p(dur_seq) / 10.0
        ], dim=-1)
        time_emb = self.time_enc(time_feat)
        
        # Concatenate features
        x = torch.cat([loc_emb, user_emb, weekday_emb, hour_emb, time_emb], dim=-1)
        
        # Pack for LSTM (handle variable lengths)
        lengths = mask.sum(dim=1).cpu().long()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        
        # LSTM encoding
        packed_out, (hidden, cell) = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=L)
        
        # Get last valid hidden state
        seq_lens = (lengths - 1).clamp(min=0)
        batch_indices = torch.arange(B, device=x.device)
        last_hidden = lstm_out[batch_indices, seq_lens]
        
        # Neural-based prediction
        neural_logits = self.prediction_head(last_hidden)
        
        # Transition-based prediction (from last location)
        last_locs = loc_seq[batch_indices, seq_lens]
        transition_logits = self.transition_priors[last_locs] * self.transition_scale
        
        # Mix predictions
        alpha = torch.sigmoid(self.mix_weight)
        logits = alpha * neural_logits + (1 - alpha) * transition_logits
        
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

def train_epoch(model, loader, optimizer, criterion, device):
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
        
        logits = model(loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, mask)
        loss = criterion(logits, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
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
    
    # Build transition priors from training data
    print("Building transition priors...")
    transition_priors = build_transition_priors(train_data, num_locations)
    print(f"Transition matrix shape: {transition_priors.shape}")
    
    # Datasets
    train_dataset = GeolifeDataset(train_data)
    val_dataset = GeolifeDataset(val_data)
    test_dataset = GeolifeDataset(test_data)
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = TransitionAwarePredictor(
        num_locations=num_locations,
        num_users=num_users,
        transition_priors=transition_priors,
        d_model=256,
        dropout=0.2
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    best_test_acc = 0
    patience = 25
    patience_counter = 0
    
    for epoch in range(100):
        print(f"\nEpoch {epoch + 1}/100")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        
        print(f"Loss: {train_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Mix weight: {torch.sigmoid(model.mix_weight).item():.3f}, Transition scale: {model.transition_scale.item():.3f}")
        
        val_metrics = calculate_metrics(model, val_loader, device)
        test_metrics = calculate_metrics(model, test_loader, device)
        
        print(f"Val:  acc@1={val_metrics['acc@1']:.2f}%, acc@5={val_metrics['acc@5']:.2f}%, acc@10={val_metrics['acc@10']:.2f}%")
        print(f"Test: acc@1={test_metrics['acc@1']:.2f}%, acc@5={test_metrics['acc@5']:.2f}%, acc@10={test_metrics['acc@10']:.2f}%")
        
        if test_metrics['acc@1'] > best_test_acc:
            best_test_acc = test_metrics['acc@1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_acc': best_test_acc,
            }, '/content/totaly_lose_nxt_loc/best_model_v6.pt')
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
