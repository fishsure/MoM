"""
Data loading and preprocessing module for RankRouter
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json


def extract_query_from_conversation(conversation: List[Dict]) -> str:
    """
    Extract the user query from conversation.
    The query is the first user message in the conversation.
    
    Args:
        conversation: List of conversation turns with 'role' and 'content'
    
    Returns:
        Query string
    """
    for turn in conversation:
        if turn.get('role') == 'user':
            return turn.get('content', '')
    return ''


def load_chatbot_arena_data(
    parquet_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load and preprocess Chatbot Arena dataset.
    
    Args:
        parquet_path: Path to the parquet file
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        seed: Random seed for splitting
    
    Returns:
        Tuple of (train_data, val_data, test_data)
        Each data is a list of dicts with keys:
        - query: str, the user query
        - model_a: str, name of model A
        - model_b: str, name of model B
        - winner: str, 'model_a', 'model_b', or 'tie'
        - question_id: str, unique question ID
    """
    # Load parquet file
    df = pd.read_parquet(parquet_path)
    
    print(f"Loaded {len(df)} samples from {parquet_path}")
    
    # Extract queries and create samples
    samples = []
    for idx, row in df.iterrows():
        # Extract query from conversation_a (both conversations have the same query)
        query = extract_query_from_conversation(row['conversation_a'])
        
        if not query:  # Skip if no query found
            continue
        
        # Get model names and winner
        model_a = str(row['model_a'])
        model_b = str(row['model_b'])
        winner = str(row['winner'])
        
        # Skip 'tie' cases for now (can be handled separately if needed)
        if winner not in ['model_a', 'model_b']:
            continue
        
        samples.append({
            'query': query,
            'model_a': model_a,
            'model_b': model_b,
            'winner': winner,
            'question_id': str(row['question_id'])
        })
    
    print(f"Processed {len(samples)} valid samples")
    
    # Split data
    np.random.seed(seed)
    indices = np.random.permutation(len(samples))
    
    train_end = int(len(samples) * train_ratio)
    val_end = train_end + int(len(samples) * val_ratio)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_data = [samples[i] for i in train_indices]
    val_data = [samples[i] for i in val_indices]
    test_data = [samples[i] for i in test_indices]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data


def get_all_models(data: List[Dict]) -> List[str]:
    """
    Get all unique model names from the dataset.
    
    Args:
        data: List of data samples
    
    Returns:
        List of unique model names
    """
    models = set()
    for sample in data:
        models.add(sample['model_a'])
        models.add(sample['model_b'])
    return sorted(list(models))


def create_model_to_id_mapping(models: List[str]) -> Dict[str, int]:
    """
    Create mapping from model name to ID.
    
    Args:
        models: List of model names
    
    Returns:
        Dictionary mapping model name to ID
    """
    return {model: idx for idx, model in enumerate(models)}

