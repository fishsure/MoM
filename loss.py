"""
Loss functions for RankRouter training
"""
import torch
import torch.nn as nn
from typing import Tuple


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise Ranking Loss (BPR - Bayesian Personalized Ranking)
    Ensures positive query-model pair has higher score than negative pair
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: Margin parameter for ranking loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise ranking loss.
        
        Args:
            pos_scores: Scores for positive query-model pairs (batch_size,)
            neg_scores: Scores for negative query-model pairs (batch_size,)
        
        Returns:
            Loss value (scalar)
        """
        # BPR loss: max(0, margin - pos_score + neg_score)
        loss = torch.clamp(self.margin - pos_scores + neg_scores, min=0.0)
        return loss.mean()


class BinaryCrossEntropyLoss(nn.Module):
    """
    Binary Cross-Entropy Loss for pointwise optimization
    """
    
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute binary cross-entropy loss.
        
        Args:
            pos_scores: Scores for positive query-model pairs (batch_size,)
            neg_scores: Scores for negative query-model pairs (batch_size,)
        
        Returns:
            Loss value (scalar)
        """
        # Combine scores and labels
        all_scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([
            torch.ones_like(pos_scores),
            torch.zeros_like(neg_scores)
        ], dim=0)
        
        return self.bce(all_scores, labels)

