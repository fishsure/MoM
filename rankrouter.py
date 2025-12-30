"""
RankRouter Model: Main architecture combining Query and Model representations
"""
import torch
import torch.nn as nn
from typing import List, Literal

try:
    from .query_representation import QueryRepresentationModule
    from .model_profiling import ModelCapabilityProfiling
except ImportError:
    from query_representation import QueryRepresentationModule
    from model_profiling import ModelCapabilityProfiling


class RankRouter(nn.Module):
    """
    RankRouter: Dual-tower architecture for query-model matching
    """
    
    def __init__(
        self,
        model_path: str,
        model_names: List[str],
        embedding_dim: int = 512,
        similarity_metric: Literal['dot', 'cosine'] = 'dot',
        **kwargs
    ):
        """
        Args:
            model_path: Path to Qwen2-0.5B model
            model_names: List of all model names
            embedding_dim: Dimension of embeddings
            similarity_metric: 'dot' or 'cosine' for computing similarity
            **kwargs: Additional arguments for sub-modules
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.similarity_metric = similarity_metric
        
        # Query representation module
        self.query_module = QueryRepresentationModule(
            model_path=model_path,
            embedding_dim=embedding_dim,
            **kwargs
        )
        
        # Model capability profiling module
        self.model_module = ModelCapabilityProfiling(
            model_names=model_names,
            embedding_dim=embedding_dim,
            **kwargs
        )
    
    def compute_similarity(
        self,
        query_emb: torch.Tensor,
        model_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity score between query and model embeddings.
        
        Args:
            query_emb: Query embeddings (batch_size, embedding_dim)
            model_emb: Model embeddings (batch_size, embedding_dim)
        
        Returns:
            Similarity scores (batch_size,)
        """
        if self.similarity_metric == 'dot':
            # Dot product
            scores = torch.sum(query_emb * model_emb, dim=1)
        elif self.similarity_metric == 'cosine':
            # Cosine similarity
            query_norm = torch.norm(query_emb, dim=1, keepdim=True)
            model_norm = torch.norm(model_emb, dim=1, keepdim=True)
            scores = torch.sum(query_emb * model_emb, dim=1) / (query_norm.squeeze() * model_norm.squeeze() + 1e-8)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        return scores
    
    def forward(
        self,
        queries: List[str],
        model_names: List[str]
    ) -> torch.Tensor:
        """
        Forward pass to compute query-model matching scores.
        
        Args:
            queries: List of query strings
            model_names: List of model names
        
        Returns:
            Similarity scores (batch_size,)
        """
        # Get query embeddings
        query_emb = self.query_module(queries)
        
        # Get model embeddings
        model_emb = self.model_module(model_names)
        
        # Compute similarity
        scores = self.compute_similarity(query_emb, model_emb)
        
        return scores
    
    def fit_clustering(self, queries: List[str], batch_size: int = 32):
        """
        Fit clustering on training queries.
        
        Args:
            queries: List of training query strings
            batch_size: Batch size for embedding extraction
        """
        self.query_module.fit_clustering(queries, batch_size)

