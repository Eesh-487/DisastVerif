"""
Module for fusing the outputs of different verification models.
This module provides logic for combining confidence scores from multiple modalities.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

def ensemble_scoring(module_scores, quality_factors=None):
    """
    Combine confidence scores from different modules into a single ensemble score.
    
    Args:
        module_scores (dict): Dictionary mapping module names to their confidence scores
        quality_factors (dict): Optional quality factors for each module (if None, equal weights are used)
        
    Returns:
        float: Ensemble score between 0 and 1
    """
    # Default weights if quality factors not provided
    default_weights = {'geospatial': 0.4, 'text': 0.3, 'news': 0.3}
    
    # Use quality factors as weights if provided, otherwise use default weights
    if quality_factors is None:
        weights = default_weights
    else:
        # Normalize quality factors to sum to 1
        total_quality = sum(quality_factors.values())
        weights = {k: v / total_quality for k, v in quality_factors.items()} if total_quality > 0 else default_weights
    
    # Calculate weighted score
    weighted_score = 0.0
    weight_sum = 0.0
    
    for module, score in module_scores.items():
        if module in weights and score is not None:
            module_weight = weights.get(module, 0.0)
            weighted_score += score * module_weight
            weight_sum += module_weight
    
    # Normalize by total weight
    if weight_sum > 0:
        ensemble_score = weighted_score / weight_sum
    else:
        ensemble_score = 0.0
    
    logger.info(f"Ensemble scoring with weights {weights}, produced score: {ensemble_score:.4f}")
    return ensemble_score

def weighted_confidence_score(scores, weights=None):
    """
    Calculate a weighted confidence score from a list of individual scores.
    
    Args:
        scores (list): List of confidence scores
        weights (list): Optional list of weights for each score
        
    Returns:
        float: Weighted confidence score
    """
    if not scores:
        return 0.0
    
    if weights is None:
        weights = [1.0] * len(scores)
    
    # Ensure weights and scores have the same length
    if len(weights) != len(scores):
        weights = weights[:len(scores)] + [1.0] * (len(scores) - len(weights))
    
    # Calculate weighted sum
    weighted_sum = sum(s * w for s, w in zip(scores, weights) if s is not None)
    total_weight = sum(w for s, w in zip(scores, weights) if s is not None)
    
    if total_weight > 0:
        return weighted_sum / total_weight
    else:
        return 0.0

def adaptive_fusion(geospatial_score, text_score, news_score):
    """
    Adaptively fuse confidence scores based on their relative strengths.
    
    Args:
        geospatial_score (float): Score from geospatial analysis
        text_score (float): Score from text similarity analysis
        news_score (float): Score from news verification
        
    Returns:
        float: Fused confidence score
    """
    scores = [geospatial_score, text_score, news_score]
    valid_scores = [s for s in scores if s is not None]
    
    if not valid_scores:
        return 0.0
    
    # If any score is very high (>0.9), give it more weight
    max_score = max(valid_scores)
    if max_score > 0.9:
        max_index = scores.index(max_score)
        weights = [1.0, 1.0, 1.0]
        weights[max_index] = 2.0
        return weighted_confidence_score(scores, weights)
    
    # If scores are consistent (low std dev), use simple average
    if np.std(valid_scores) < 0.2:
        return np.mean(valid_scores)
    
    # Otherwise, weight by inverse of distance from median
    median_score = np.median(valid_scores)
    distances = [abs(s - median_score) if s is not None else float('inf') for s in scores]
    weights = [1.0 / (d + 0.1) for d in distances]  # Add 0.1 to avoid division by zero
    
    return weighted_confidence_score(scores, weights)

def combine_evidence(geospatial_evidence, text_evidence, news_evidence):
    """
    Combine evidence from different modules.
    
    Args:
        geospatial_evidence (list): Evidence items from geospatial analysis
        text_evidence (list): Evidence items from text similarity analysis
        news_evidence (list): Evidence items from news verification
        
    Returns:
        list: Combined evidence items
    """
    combined_evidence = []
    
    # Add prefix to evidence items to indicate their source
    for item in geospatial_evidence:
        item['source'] = 'geospatial'
        combined_evidence.append(item)
    
    for item in text_evidence:
        item['source'] = 'text'
        combined_evidence.append(item)
    
    for item in news_evidence:
        item['source'] = 'news'
        combined_evidence.append(item)
    
    # Sort evidence by score (descending)
    combined_evidence.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    return combined_evidence