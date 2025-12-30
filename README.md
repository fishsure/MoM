# RankRouter Reproduction

This directory contains the reproduction code for the RankRouter paper experiments.

## Overview

RankRouter is a capability-aware routing framework for matching queries with the most suitable models in a mixture of large language models (LLMs). The framework consists of:

1. **Query Representation Module**: Encodes input queries using Qwen2-0.5B and extracts auxiliary clustering features
2. **Model Capability Profiling Module**: Profiles models based on their capabilities across multiple tasks
3. **Pairwise Ranking Loss**: Optimizes query-model matching using BPR (Bayesian Personalized Ranking) loss

## Directory Structure

```
MoM/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data_loader.py            # Data loading and preprocessing
├── query_representation.py   # Query representation module
├── model_profiling.py        # Model capability profiling module
├── rankrouter.py            # Main RankRouter model
├── loss.py                  # Loss functions (BPR, BCE)
├── metrics.py               # Evaluation metrics
└── train.py                 # Training script
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure data and model paths are correct:
   - Data: `../data/chatbot_arena/data/train-00000-of-00001-cced8514c7ed782a.parquet`
   - Model: `../model/Qwen2-0.5B`

## Usage

### Training

Train RankRouter with default settings:
```bash
python train.py
```

Train with custom parameters:
```bash
python train.py \
    --data_path ../data/chatbot_arena/data/train-00000-of-00001-cced8514c7ed782a.parquet \
    --model_path ../model/Qwen2-0.5B \
    --output_dir ./outputs \
    --batch_size 32 \
    --epochs 10 \
    --lr 1e-4 \
    --embedding_dim 512 \
    --margin 1.0 \
    --similarity_metric dot \
    --num_clusters 50 \
    --loss_type bpr
```

### Arguments

- `--data_path`: Path to training data parquet file
- `--model_path`: Path to Qwen2-0.5B model directory
- `--output_dir`: Output directory for checkpoints and logs (default: `./outputs`)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 1e-4)
- `--embedding_dim`: Embedding dimension (default: 512)
- `--margin`: Margin for pairwise ranking loss (default: 1.0)
- `--similarity_metric`: Similarity metric: 'dot' or 'cosine' (default: 'dot')
- `--num_clusters`: Number of clusters for query clustering (default: 50)
- `--loss_type`: Loss type: 'bpr' (pairwise ranking) or 'bce' (binary cross-entropy) (default: 'bpr')
- `--seed`: Random seed (default: 42)

## Model Architecture

### Query Representation Module

- Uses Qwen2-0.5B as the backbone encoder
- Extracts semantic embeddings from query text
- Applies K-means clustering on training queries to extract auxiliary features
- Combines semantic and clustering features into final query embedding

### Model Capability Profiling Module

- Creates learnable capability vectors for each model
- Embeds model IDs
- Combines capability vectors and ID embeddings into model embeddings

### RankRouter

- Dual-tower architecture: separate towers for queries and models
- Computes similarity scores using dot product or cosine similarity
- Optimized with pairwise ranking loss (BPR)

## Evaluation Metrics

The framework evaluates routing performance using:
- **Accuracy**: Proportion of queries where the router identifies the ground-truth optimal model
- **Precision**: Precision of model selection
- **Recall**: Recall of model selection
- **F1 Score**: F1 score of model selection
- **AUC**: Area Under the ROC Curve for ranking quality

## Output

Training produces:
- `best_model.pt`: Best model checkpoint (based on validation AUC)
- `training_history.json`: Training history with losses and metrics

## Notes

- The code assumes data and models are stored outside the MoM directory
- No data or model files should be stored in the MoM directory
- The framework automatically splits data into train/val/test sets (80/10/10 by default)
- Clustering is fitted on training queries before training starts

## Citation

If you use this code, please cite the original RankRouter paper.
