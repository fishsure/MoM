"""
Evaluation metrics for RankRouter
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from typing import Dict, List, Tuple


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_scores: Prediction scores (probabilities)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # AUC (requires scores)
    try:
        metrics['auc'] = roc_auc_score(y_true, y_scores)
    except ValueError:
        # If only one class present, set AUC to 0.5
        metrics['auc'] = 0.5
    
    return metrics


def evaluate_router(
    model,
    data: List[Dict],
    model_to_idx: Dict[str, int],
    device: torch.device,
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Evaluate router on dataset.
    
    Args:
        model: RankRouter model
        data: List of data samples
        model_to_idx: Mapping from model name to index
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_pos_scores = []
    all_neg_scores = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            
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
            
            # Compute scores
            pos_scores = model(queries, pos_models)
            neg_scores = model(queries, neg_models)
            
            # Move to CPU and convert to numpy
            if isinstance(pos_scores, torch.Tensor):
                pos_scores = pos_scores.cpu().numpy()
            if isinstance(neg_scores, torch.Tensor):
                neg_scores = neg_scores.cpu().numpy()
            
            all_pos_scores.extend(pos_scores)
            all_neg_scores.extend(neg_scores)
            
            # Predictions: positive if pos_score > neg_score
            preds = (pos_scores > neg_scores).astype(int)
            all_preds.extend(preds)
            
            # Labels: all positive (1) since we're comparing positive vs negative
            all_labels.extend([1] * len(batch))
    
    # Convert to numpy arrays
    all_pos_scores = np.array(all_pos_scores)
    all_neg_scores = np.array(all_neg_scores)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Use positive scores for AUC calculation
    all_scores = all_pos_scores
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, all_scores)
    
    return metrics

