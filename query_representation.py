"""
Query Representation Module for RankRouter
Extracts semantic embeddings and auxiliary clustering features
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import numpy as np
from typing import List, Dict, Optional
import os


class QueryRepresentationModule(nn.Module):
    """
    Query Representation Module that combines:
    1. Semantic embeddings from Qwen2-0.5B
    2. Auxiliary clustering features
    """
    
    def __init__(
        self,
        model_path: str,
        embedding_dim: int = 512,
        cluster_dim: int = 32,
        num_clusters: int = 50,
        dropout: float = 0.1
    ):
        """
        Args:
            model_path: Path to Qwen2-0.5B model
            embedding_dim: Dimension of query embedding
            cluster_dim: Dimension of clustering features
            num_clusters: Number of clusters for K-means
            dropout: Dropout rate
        """
        super().__init__()
        
        # Load Qwen2-0.5B model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.backbone = AutoModel.from_pretrained(model_path)
        
        # Get hidden size from model config
        hidden_size = self.backbone.config.hidden_size
        
        # Projection layer to reduce dimension
        self.semantic_proj = nn.Linear(hidden_size, embedding_dim - cluster_dim)
        
        # Clustering features (will be initialized after clustering)
        self.cluster_dim = cluster_dim
        self.num_clusters = num_clusters
        self.cluster_centers = None
        self.cluster_embedding = nn.Embedding(num_clusters, cluster_dim)
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def extract_semantic_embedding(self, queries: List[str]) -> torch.Tensor:
        """
        Extract semantic embeddings from queries using Qwen2-0.5B.
        
        Args:
            queries: List of query strings
        
        Returns:
            Tensor of shape (batch_size, hidden_size)
        """
        # Tokenize
        inputs = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings (use mean pooling)
        outputs = self.backbone(**inputs)
        # Use mean pooling
        embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        # Masked mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        return embeddings
    
    def compute_clustering_features(
        self,
        semantic_embeddings: torch.Tensor,
        cluster_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute clustering features from semantic embeddings.
        
        Args:
            semantic_embeddings: Tensor of shape (batch_size, hidden_size)
            cluster_ids: Optional cluster IDs, if None will use nearest cluster
        
        Returns:
            Tensor of shape (batch_size, cluster_dim)
        """
        if self.cluster_centers is None:
            # If no cluster centers, return zero features
            batch_size = semantic_embeddings.size(0)
            return torch.zeros(batch_size, self.cluster_dim, device=semantic_embeddings.device)
        
        # Find nearest cluster for each embedding
        if cluster_ids is None:
            # Compute distances to all cluster centers
            semantic_np = semantic_embeddings.detach().cpu().numpy()
            centers_np = self.cluster_centers
            
            # Compute distances
            distances = np.sqrt(((semantic_np[:, None, :] - centers_np[None, :, :]) ** 2).sum(axis=2))
            cluster_ids = torch.from_numpy(np.argmin(distances, axis=1)).to(semantic_embeddings.device)
        
        # Get cluster embeddings
        cluster_features = self.cluster_embedding(cluster_ids)
        
        return cluster_features
    
    def fit_clustering(self, queries: List[str], batch_size: int = 32):
        """
        Fit K-means clustering on training queries.
        
        Args:
            queries: List of training query strings
            batch_size: Batch size for embedding extraction
        """
        self.eval()
        all_embeddings = []
        
        # Extract embeddings in batches
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i+batch_size]
                embeddings = self.extract_semantic_embedding(batch_queries)
                all_embeddings.append(embeddings.cpu())
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        
        # Fit K-means
        print(f"Fitting K-means with {self.num_clusters} clusters on {len(all_embeddings)} queries...")
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        kmeans.fit(all_embeddings)
        
        # Store cluster centers
        self.cluster_centers = kmeans.cluster_centers_
        print(f"K-means clustering completed. Cluster centers shape: {self.cluster_centers.shape}")
    
    def forward(self, queries: List[str]) -> torch.Tensor:
        """
        Forward pass to get query embeddings.
        
        Args:
            queries: List of query strings
        
        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        # Extract semantic embeddings
        semantic_emb = self.extract_semantic_embedding(queries)
        semantic_emb = self.semantic_proj(semantic_emb)
        
        # Compute clustering features
        cluster_features = self.compute_clustering_features(semantic_emb)
        
        # Concatenate
        query_emb = torch.cat([semantic_emb, cluster_features], dim=1)
        
        # Final projection
        query_emb = self.final_proj(query_emb)
        
        return query_emb

