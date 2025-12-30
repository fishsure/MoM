"""
Model Capability Profiling Module for RankRouter
Profiles models based on their capabilities across different tasks
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional


class ModelCapabilityProfiling(nn.Module):
    """
    Model Capability Profiling Module that:
    1. Creates capability vectors for each model
    2. Embeds model IDs
    3. Combines them into model embeddings
    """
    
    def __init__(
        self,
        model_names: List[str],
        capability_dim: int = 64,
        embedding_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            model_names: List of all model names
            capability_dim: Dimension of capability vector
            embedding_dim: Final embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.model_names = model_names
        self.num_models = len(model_names)
        self.capability_dim = capability_dim
        self.embedding_dim = embedding_dim
        
        # Model ID embedding
        self.model_id_embedding = nn.Embedding(self.num_models, embedding_dim // 2)
        
        # Capability vector (learnable per model)
        # Initialize with random values (can be replaced with actual capability scores)
        self.capability_vectors = nn.Parameter(
            torch.randn(self.num_models, capability_dim)
        )
        
        # Project capability vector to embedding space
        self.capability_proj = nn.Sequential(
            nn.Linear(capability_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, embedding_dim // 2)
        )
        
        # Final combination
        self.final_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Create model name to index mapping
        self.model_to_idx = {name: idx for idx, name in enumerate(model_names)}
    
    def get_model_idx(self, model_name: str) -> int:
        """Get model index from model name."""
        return self.model_to_idx.get(model_name, 0)
    
    def forward(self, model_names: List[str]) -> torch.Tensor:
        """
        Forward pass to get model embeddings.
        
        Args:
            model_names: List of model names
        
        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        # Get model indices
        model_indices = torch.tensor(
            [self.get_model_idx(name) for name in model_names],
            dtype=torch.long,
            device=next(self.parameters()).device
        )
        
        # Get model ID embeddings
        id_emb = self.model_id_embedding(model_indices)
        
        # Get capability vectors
        capability_vecs = self.capability_vectors[model_indices]
        capability_emb = self.capability_proj(capability_vecs)
        
        # Concatenate
        model_emb = torch.cat([id_emb, capability_emb], dim=1)
        
        # Final projection
        model_emb = self.final_proj(model_emb)
        
        return model_emb
    
    def initialize_capability_vectors(self, capability_scores: Optional[Dict[str, np.ndarray]] = None):
        """
        Initialize capability vectors with actual scores if available.
        
        Args:
            capability_scores: Dictionary mapping model name to capability vector
        """
        if capability_scores is None:
            # Use random initialization (already done)
            return
        
        # Initialize with provided scores
        for model_name, scores in capability_scores.items():
            if model_name in self.model_to_idx:
                idx = self.model_to_idx[model_name]
                if len(scores) == self.capability_dim:
                    self.capability_vectors.data[idx] = torch.from_numpy(scores).float()

