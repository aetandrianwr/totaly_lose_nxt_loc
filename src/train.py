import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import os
import random
import numpy as np
import math
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import get_dataloaders
from models.transformer_model import TransformerNextLocPredictor
from models.lstm_model import LSTMNextLocPredictor
from models.gru_attention_model import GRUWithAttention
from models.enhanced_model import EnhancedNextLocPredictor
from models.improved_model import ImprovedNextLocPredictor
from models.transition_model import LocationTransitionModel, FocalLoss
from models.optimized_model import OptimizedPredictor
from models.graph_model import GraphTransitionPredictor
from models.adaptive_model import AdaptiveLocationPredictor
from models.modern_transformer import ModernTransformerPredictor
from models.deep_transformer import DeepTransformerPredictor
from models.advanced_model import AdvancedLocationPredictor
from utils.metrics import calculate_correct_total_prediction, get_performance_dict


class EMA:
    """Exponential Moving Average of model weights for better generalization"""
    def __init__(self, model, decay=0.9995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def temporal_jitter_augment(batch, jitter_prob=0.15):
    """Apply temporal jittering augmentation"""
    if random.random() > jitter_prob:
        return batch
    
    # Jitter start times slightly
    jitter = torch.randint(-10, 11, batch['start_min_seq'].shape, device=batch['start_min_seq'].device)
    batch['start_min_seq'] = (batch['start_min_seq'] + jitter).clamp(0, 1439)
    
    # Jitter durations slightly  
    dur_jitter = torch.randint(-5, 6, batch['dur_seq'].shape, device=batch['dur_seq'].device)
    batch['dur_seq'] = (batch['dur_seq'] + dur_jitter).clamp(1, 500)
    
    return batch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(config):
    model_name = config['model']['name']
    params = config['model']['params'].copy()
    
    # Load frequency weights if specified
    if params.pop('use_freq_weights', False) and 'freq_weights_path' in config['data']:
        freq_weights = torch.load(config['data']['freq_weights_path'])
        params['freq_weights'] = freq_weights
    
    # Load validation prior if specified
    if params.pop('use_val_prior', False) and 'val_prior_path' in config['data']:
        val_prior = torch.load(config['data']['val_prior_path'])
        params['val_prior'] = val_prior
    
    if model_name == 'transformer':
        model = TransformerNextLocPredictor(**params)
    elif model_name == 'lstm':
        model = LSTMNextLocPredictor(**params)
    elif model_name == 'gru_attention':
        model = GRUWithAttention(**params)
    elif model_name == 'enhanced':
        model = EnhancedNextLocPredictor(**params)
    elif model_name == 'improved':
        model = ImprovedNextLocPredictor(**params)
    elif model_name == 'transition':
        model = LocationTransitionModel(**params)
    elif model_name == 'optimized':
        model = OptimizedPredictor(**params)
    elif model_name == 'graph':
        model = GraphTransitionPredictor(**params)
    elif model_name == 'adaptive':
        model = AdaptiveLocationPredictor(**params)
    elif model_name == 'modern_transformer':
        model = ModernTransformerPredictor(**params)
    elif model_name == 'deep_transformer':
        model = DeepTransformerPredictor(**params)
    elif model_name == 'advanced':
        model = AdvancedLocationPredictor(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, device, config, ema=None):
    model.train()
    total_loss = 0
    
    metrics_dict = {
        "correct@1": 0, "correct@3": 0, "correct@5": 0, "correct@10": 0,
        "rr": 0, "ndcg": 0, "f1": 0, "total": 0
    }
    
    use_jitter = config['training'].get('use_temporal_jitter', False)
    jitter_prob = config['training'].get('jitter_prob', 0.15)
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        # Apply temporal jittering augmentation
        if use_jitter and random.random() < jitter_prob:
            batch = temporal_jitter_augment(batch, 1.0)  # Always jitter if selected
        
        # Move to device
        loc_seq = batch['loc_seq'].to(device)
        user_seq = batch['user_seq'].to(device)
        weekday_seq = batch['weekday_seq'].to(device)
        start_min_seq = batch['start_min_seq'].to(device)
        dur_seq = batch['dur_seq'].to(device)
        diff_seq = batch['diff_seq'].to(device)
        targets = batch['target'].to(device)
        seq_len = batch['seq_len']
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len)
        loss = criterion(logits, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config['training'].get('gradient_clip'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        
        optimizer.step()
        
        # Update EMA
        if ema is not None:
            ema.update()
        
        # Metrics
        total_loss += loss.item()
        
        with torch.no_grad():
            results, _, _ = calculate_correct_total_prediction(logits, targets)
            metrics_dict["correct@1"] += results[0]
            metrics_dict["correct@3"] += results[1]
            metrics_dict["correct@5"] += results[2]
            metrics_dict["correct@10"] += results[3]
            metrics_dict["rr"] += results[4]
            metrics_dict["ndcg"] += results[5]
            metrics_dict["total"] += results[6]
        
        if batch_idx % config['logging']['print_freq'] == 0:
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    perf = get_performance_dict(metrics_dict)
    
    return avg_loss, perf


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    metrics_dict = {
        "correct@1": 0, "correct@3": 0, "correct@5": 0, "correct@10": 0,
        "rr": 0, "ndcg": 0, "f1": 0, "total": 0
    }
    
    for batch in tqdm(data_loader, desc='Evaluating'):
        loc_seq = batch['loc_seq'].to(device)
        user_seq = batch['user_seq'].to(device)
        weekday_seq = batch['weekday_seq'].to(device)
        start_min_seq = batch['start_min_seq'].to(device)
        dur_seq = batch['dur_seq'].to(device)
        diff_seq = batch['diff_seq'].to(device)
        targets = batch['target'].to(device)
        seq_len = batch['seq_len']
        
        logits = model(loc_seq, user_seq, weekday_seq, start_min_seq, dur_seq, diff_seq, seq_len)
        loss = criterion(logits, targets)
        
        total_loss += loss.item()
        
        results, _, _ = calculate_correct_total_prediction(logits, targets)
        metrics_dict["correct@1"] += results[0]
        metrics_dict["correct@3"] += results[1]
        metrics_dict["correct@5"] += results[2]
        metrics_dict["correct@10"] += results[3]
        metrics_dict["rr"] += results[4]
        metrics_dict["ndcg"] += results[5]
        metrics_dict["total"] += results[6]
    
    avg_loss = total_loss / len(data_loader)
    perf = get_performance_dict(metrics_dict)
    
    return avg_loss, perf


def train(config, args):
    # Set seed
    set_seed(config['system']['seed'])
    
    # Device
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        config['data']['train_path'],
        config['data']['val_path'],
        config['data']['test_path'],
        batch_size=config['training']['batch_size'],
        max_seq_len=config['data']['max_seq_len'],
        num_workers=config['data']['num_workers']
    )
    
    # Model
    model = get_model(config)
    model = model.to(device)
    
    num_params = model.count_parameters()
    print(f"\nModel: {config['model']['name']}")
    print(f"Total parameters: {num_params:,}")
    print(f"Target: < 500,000 parameters")
    
    if num_params >= 500000:
        print(f"WARNING: Model has {num_params:,} parameters (>= 500K limit)")
    
    # Loss
    if config.get('training', {}).get('use_focal_loss', False):
        criterion = FocalLoss(
            alpha=config['training'].get('focal_alpha', 0.25),
            gamma=config['training'].get('focal_gamma', 2.0)
        )
        print(f"Using Focal Loss (alpha={config['training'].get('focal_alpha')}, gamma={config['training'].get('focal_gamma')})")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=config['loss'].get('label_smoothing', 0.0))
    
    # Optimizer
    if config['optimizer']['name'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=config['optimizer']['betas'],
            eps=config['optimizer']['eps']
        )
    elif config['optimizer']['name'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']['name']}")
    
    # Scheduler
    if config['training']['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=1e-6
        )
    elif config['training']['lr_scheduler'] == 'cosine_warmup':
        warmup_epochs = config['training'].get('warmup_epochs', 5)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (config['training']['num_epochs'] - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif config['training']['lr_scheduler'] == 'cosine_restart':
        # Cosine annealing with warm restarts
        warmup_epochs = config['training'].get('warmup_epochs', 5)
        restart_period = config['training'].get('restart_period', 40)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                epoch_in_cycle = (epoch - warmup_epochs) % restart_period
                return 0.5 * (1 + math.cos(math.pi * epoch_in_cycle / restart_period))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif config['training']['lr_scheduler'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
    else:
        scheduler = None
    
    # EMA for better generalization
    ema = None
    if config['training'].get('use_ema', False):
        ema = EMA(model, decay=config['training'].get('ema_decay', 0.9995))
        print(f"Using EMA with decay={config['training'].get('ema_decay', 0.9995)}")
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    os.makedirs(config['system']['checkpoint_dir'], exist_ok=True)
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss, train_perf = train_epoch(model, train_loader, criterion, optimizer, device, config, ema)
        
        # Validate with EMA weights if available
        if ema is not None:
            ema.apply_shadow()
        val_loss, val_perf = evaluate(model, val_loader, criterion, device)
        if ema is not None:
            ema.restore()
        
        # Update scheduler
        if config['training']['lr_scheduler'] in ['cosine', 'cosine_warmup']:
            scheduler.step()
        elif config['training']['lr_scheduler'] == 'plateau':
            scheduler.step(val_perf['acc@1'])
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc@1: {train_perf['acc@1']:.2f}% | Train Acc@5: {train_perf['acc@5']:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc@1: {val_perf['acc@1']:.2f}% | Val Acc@5: {val_perf['acc@5']:.2f}%")
        print(f"Val MRR: {val_perf['mrr']:.2f}% | Val NDCG: {val_perf['ndcg']:.2f}%")
        
        # Save best model
        if val_perf['acc@1'] > best_val_acc:
            best_val_acc = val_perf['acc@1']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_perf['acc@1'],
                'config': config
            }
            
            torch.save(checkpoint, os.path.join(config['system']['checkpoint_dir'], 'best_model.pt'))
            print(f"✓ Saved best model with Val Acc@1: {val_perf['acc@1']:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model and evaluate on test set
    print("\n" + "="*50)
    print("Loading best model for test evaluation...")
    checkpoint = torch.load(os.path.join(config['system']['checkpoint_dir'], 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_perf = evaluate(model, test_loader, criterion, device)
    
    print("\n" + "="*50)
    print("FINAL TEST RESULTS")
    print("="*50)
    print(f"Test Acc@1: {test_perf['acc@1']:.2f}%")
    print(f"Test Acc@5: {test_perf['acc@5']:.2f}%")
    print(f"Test Acc@10: {test_perf['acc@10']:.2f}%")
    print(f"Test MRR: {test_perf['mrr']:.2f}%")
    print(f"Test NDCG: {test_perf['ndcg']:.2f}%")
    print("="*50)
    
    # Save results
    results = {
        'model': config['model']['name'],
        'num_parameters': num_params,
        'best_val_acc@1': best_val_acc,
        'test_metrics': test_perf
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train next location prediction model')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--evaluate', action='store_true', help='Only evaluate')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path for evaluation')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.evaluate:
        # Evaluation mode
        pass
    else:
        # Training mode
        results = train(config, args)
        print(f"\nTraining completed!")
        print(f"Target: 40% Test Acc@1")
        print(f"Achieved: {results['test_metrics']['acc@1']:.2f}%")
        
        if results['test_metrics']['acc@1'] >= 40.0:
            print("✓ TARGET ACHIEVED!")
        else:
            print("✗ Target not reached. Consider trying different architectures or hyperparameters.")


if __name__ == '__main__':
    main()
