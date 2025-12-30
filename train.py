"""
Training script for RankRouter
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_chatbot_arena_data, get_all_models, create_model_to_id_mapping
from rankrouter import RankRouter
from loss import PairwiseRankingLoss, BinaryCrossEntropyLoss
from metrics import evaluate_router


class RankingDataset(Dataset):
    """Dataset for pairwise ranking"""
    
    def __init__(self, data: list):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Custom collate function for batching"""
    return batch


def train_epoch(
    model: RankRouter,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        queries = [sample['query'] for sample in batch]
        
        # Get positive and negative models
        pos_models = []
        neg_models = []
        
        for sample in batch:
            if sample['winner'] == 'model_a':
                pos_models.append(sample['model_a'])
                neg_models.append(sample['model_b'])
            else:  # model_b wins
                pos_models.append(sample['model_b'])
                neg_models.append(sample['model_a'])
        
        # Forward pass
        pos_scores = model(queries, pos_models)
        neg_scores = model(queries, neg_models)
        
        # Compute loss
        loss = criterion(pos_scores, neg_scores)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description='Train RankRouter')
    parser.add_argument('--data_path', type=str, 
                       default='../data/chatbot_arena/data/train-00000-of-00001-cced8514c7ed782a.parquet',
                       help='Path to training data parquet file')
    parser.add_argument('--model_path', type=str,
                       default='../model/Qwen2-0.5B',
                       help='Path to Qwen2-0.5B model')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=512,
                       help='Embedding dimension')
    parser.add_argument('--margin', type=float, default=1.0,
                       help='Margin for pairwise ranking loss')
    parser.add_argument('--similarity_metric', type=str, default='dot',
                       choices=['dot', 'cosine'],
                       help='Similarity metric')
    parser.add_argument('--num_clusters', type=int, default=50,
                       help='Number of clusters for query clustering')
    parser.add_argument('--loss_type', type=str, default='bpr',
                       choices=['bpr', 'bce'],
                       help='Loss type: bpr (pairwise ranking) or bce (binary cross-entropy)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_data, val_data, test_data = load_chatbot_arena_data(
        args.data_path,
        seed=args.seed
    )
    
    # Get all models
    all_models = get_all_models(train_data + val_data + test_data)
    model_to_idx = create_model_to_id_mapping(all_models)
    print(f"Found {len(all_models)} unique models: {all_models}")
    
    # Initialize model
    print("Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = RankRouter(
        model_path=args.model_path,
        model_names=all_models,
        embedding_dim=args.embedding_dim,
        similarity_metric=args.similarity_metric,
        num_clusters=args.num_clusters
    ).to(device)
    
    # Fit clustering on training queries
    print("Fitting query clustering...")
    train_queries = [sample['query'] for sample in train_data]
    model.fit_clustering(train_queries, batch_size=args.batch_size)
    
    # Initialize loss
    if args.loss_type == 'bpr':
        criterion = PairwiseRankingLoss(margin=args.margin)
    else:
        criterion = BinaryCrossEntropyLoss()
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create dataloaders
    train_dataset = RankingDataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Training loop
    print("Starting training...")
    best_val_auc = 0.0
    train_losses = []
    val_metrics_history = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        print("Evaluating on validation set...")
        val_metrics = evaluate_router(model, val_data, model_to_idx, device, args.batch_size)
        val_metrics_history.append(val_metrics)
        print(f"Val Metrics: {val_metrics}")
        
        # Save best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            checkpoint_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'model_names': all_models,
                'args': vars(args)
            }, checkpoint_path)
            print(f"Saved best model with AUC: {best_val_auc:.4f}")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_router(model, test_data, model_to_idx, device, args.batch_size)
    print(f"Test Metrics: {test_metrics}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_metrics': val_metrics_history,
        'test_metrics': test_metrics,
        'args': vars(args)
    }
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed! Results saved to {output_dir}")


if __name__ == '__main__':
    main()

