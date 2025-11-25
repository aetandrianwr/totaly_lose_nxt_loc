# Next Location Prediction on Geolife Dataset

A PyTorch-based deep learning system for predicting next locations in human mobility trajectories using the Geolife dataset.

## Project Structure

```
.
├── src/
│   ├── models/          # Model architectures
│   ├── data/            # Data loading and preprocessing
│   ├── utils/           # Utility functions and metrics
│   └── train.py         # Training script
├── configs/             # Configuration files
├── checkpoints/         # Saved model checkpoints
├── logs/                # Training logs
├── results/             # Experimental results
├── data/                # Dataset
├── environment.yml      # Conda environment
└── requirements.txt     # Python dependencies
```

## Setup

### Create Conda Environment

```bash
conda env create -f environment.yml
conda activate geolife_prediction
```

### Or use pip

```bash
pip install -r requirements.txt
```

## Dataset

The Geolife dataset contains:
- **Train**: 7,424 sequences
- **Validation**: 3,334 sequences
- **Test**: 3,502 sequences
- **Vocabulary**: 1,167 unique locations
- **Users**: 45
- **Avg sequence length**: ~18

Each sample contains:
- `X`: Location sequence
- `user_X`: User IDs
- `weekday_X`: Day of week
- `start_min_X`: Start time in minutes
- `dur_X`: Duration
- `diff`: Time difference
- `Y`: Target next location

## Usage

```bash
# Train model
python src/train.py --config configs/default.yaml

# Evaluate
python src/train.py --config configs/default.yaml --evaluate --checkpoint checkpoints/best_model.pt
```

## Objective

Achieve **40% test Acc@1** with < 500K parameters using GPU acceleration.

## Metrics

- Acc@1, Acc@5, Acc@10
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- F1 Score
