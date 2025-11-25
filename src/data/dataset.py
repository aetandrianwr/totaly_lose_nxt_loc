import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class GeolifeDataset(Dataset):
    def __init__(self, data_path, max_seq_len=60):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Extract features
        loc_seq = sample['X']  # Location sequence
        user_seq = sample['user_X']  # User IDs
        weekday_seq = sample['weekday_X']  # Weekday
        start_min_seq = sample['start_min_X']  # Start time in minutes
        dur_seq = sample['dur_X']  # Duration
        diff_seq = sample['diff']  # Time difference
        target = sample['Y']  # Target location
        
        seq_len = len(loc_seq)
        
        # Pad sequences
        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            loc_seq = np.pad(loc_seq, (0, pad_len), mode='constant', constant_values=0)
            user_seq = np.pad(user_seq, (0, pad_len), mode='constant', constant_values=0)
            weekday_seq = np.pad(weekday_seq, (0, pad_len), mode='constant', constant_values=0)
            start_min_seq = np.pad(start_min_seq, (0, pad_len), mode='constant', constant_values=0)
            dur_seq = np.pad(dur_seq, (0, pad_len), mode='constant', constant_values=0.0)
            diff_seq = np.pad(diff_seq, (0, pad_len), mode='constant', constant_values=0)
        else:
            # Truncate if necessary
            loc_seq = loc_seq[-self.max_seq_len:]
            user_seq = user_seq[-self.max_seq_len:]
            weekday_seq = weekday_seq[-self.max_seq_len:]
            start_min_seq = start_min_seq[-self.max_seq_len:]
            dur_seq = dur_seq[-self.max_seq_len:]
            diff_seq = diff_seq[-self.max_seq_len:]
            seq_len = self.max_seq_len
        
        return {
            'loc_seq': torch.LongTensor(loc_seq),
            'user_seq': torch.LongTensor(user_seq),
            'weekday_seq': torch.LongTensor(weekday_seq),
            'start_min_seq': torch.LongTensor(start_min_seq),
            'dur_seq': torch.FloatTensor(dur_seq),
            'diff_seq': torch.LongTensor(diff_seq),
            'seq_len': seq_len,
            'target': torch.LongTensor([target]).squeeze()
        }


def get_dataloaders(train_path, val_path, test_path, batch_size=128, max_seq_len=60, num_workers=4):
    train_dataset = GeolifeDataset(train_path, max_seq_len)
    val_dataset = GeolifeDataset(val_path, max_seq_len)
    test_dataset = GeolifeDataset(test_path, max_seq_len)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
