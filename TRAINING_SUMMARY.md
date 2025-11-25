# Next Location Prediction - Training Summary

## Objective
Achieve stable 40% test Acc@1 on Geolife dataset with <500K parameters using GPU.

## Dataset Statistics
- Train: 7,424 sequences
- Validation: 3,334 sequences  
- Test: 3,502 sequences
- Vocabulary: 1,167 unique locations
- Avg sequence length: ~18
- High class imbalance: Top location appears 18K times
- Test target coverage: 97.8% seen in training

## Models Explored

### 1. GRU + Attention (baseline)
- Parameters: ~473K
- Test Acc@1: **31.44%**
- Val Acc@1: 35.00%

### 2. Enhanced Model (TCN + Attention)
- Parameters: 490K
- Test Acc@1: **33.04%**
- Val Acc@1: 42.86%
- Observation: High validation accuracy, but overfitting

### 3. Improved Model (BiGRU + regularization)
- Parameters: 473K
- Test Acc@1: **35.89%**
- Val Acc@1: 40.13%
- Best so far!

### 4. Transition Model (Focal Loss)
- Parameters: 495K
- Test Acc@1: **34.09%**
- Val Acc@1: 38.72%

### 5. Optimized Model (BiGRU + residual)
- Parameters: 493K
- Test Acc@1: **32.95%**
- Val Acc@1: 39.83%

## Key Findings

1. **Consistent Val-Test Gap**: All models show 4-7% drop from validation to test accuracy
2. **Best Test Result**: 35.89% (Improved Model)
3. **Overfitting**: Models quickly reach high train accuracy (65-70%+) but plateau on test
4. **Class Imbalance**: Heavy imbalance affects learning
5. **Architecture Insights**:
   - BiGRU performs better than single-direction
   - Attention mechanisms help
   - Too much capacity leads to overfitting
   - Stronger regularization (dropout 0.25-0.3) is necessary

## Challenges

- Test set appears to have different distribution than validation
- High class imbalance makes learning difficult
- Parameter budget (<500K) limits model capacity
- Need for better generalization strategies

## Next Steps for 40% Acc@1

Would need to explore:
1. **Data augmentation**: Sequence perturbation, masking
2. **Ensemble methods**: Combine multiple models
3. **Better preprocessing**: Location clustering, hierarchy
4. **Advanced techniques**: Mixup, CutMix for sequences
5. **Loss functions**: Focal loss with better tuning
6. **Transfer learning**: Pre-train on location co-occurrence
7. **Feature engineering**: Better temporal/spatial encodings

## Best Checkpoint

Model: `improved_model.py`
Config: `configs/improved.yaml`
Test Acc@1: 35.89%
Parameters: 473,139
