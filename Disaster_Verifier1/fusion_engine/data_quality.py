"""
Module for assessing the quality and reliability of the data and models.
This module provides functions for evaluating consistency and reliability.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

def cross_validation_consistency(module_score_samples):
    """
    Evaluate the consistency of each module based on cross-validation samples.
    
    Args:
        module_score_samples (list): List of dictionaries with module scores
        
    Returns:
        dict: Quality factors for each module
    """
    # Initialize quality factors dictionary
    quality_factors = {}
    
    # Get list of available modules
    all_modules = set()
    for sample in module_score_samples:
        all_modules.update(sample.keys())
    
    # Calculate consistency metrics for each module
    for module in all_modules:
        # Get scores for this module across all samples
        scores = [sample.get(module) for sample in module_score_samples if module in sample]
        scores = [s for s in scores if s is not None]
        
        if not scores:
            quality_factors[module] = 0.0
            continue
        
        # Consistency is inversely related to the standard deviation
        std_dev = np.std(scores)
        consistency = 1.0 / (1.0 + 5.0 * std_dev)  # Scale factor of 5 to emphasize differences
        
        # Scale the consistency to [0, 1]
        quality_factors[module] = min(1.0, max(0.0, consistency))
    
    logger.info(f"Quality factors calculated: {quality_factors}")
    return quality_factors

def assess_data_quality(df):
    """
    Assess the quality of the input data.
    
    Args:
        df (DataFrame): Input disaster data
        
    Returns:
        dict: Data quality metrics
    """
    quality_metrics = {}
    
    # Check for missing values
    missing_pct = df.isnull().mean().to_dict()
    quality_metrics['missing_values'] = missing_pct
    
    # Check for data completeness
    required_columns = ['latitude', 'longitude', 'description']
    completeness = sum(col in df.columns for col in required_columns) / len(required_columns)
    quality_metrics['completeness'] = completeness
    
    # Check for consistency in coordinates
    if 'latitude' in df.columns and 'longitude' in df.columns:
        valid_coords = (df['latitude'].between(-90, 90) & df['longitude'].between(-180, 180)).mean()
        quality_metrics['coordinate_validity'] = valid_coords
    else:
        quality_metrics['coordinate_validity'] = 0.0
    
    # Check for text quality in descriptions
    if 'description' in df.columns:
        # Simple check: Average length of descriptions
        avg_desc_length = df['description'].astype(str).apply(len).mean()
        text_quality = min(1.0, avg_desc_length / 200.0)  # Scale factor: 200 chars is "good"
        quality_metrics['description_quality'] = text_quality
    else:
        quality_metrics['description_quality'] = 0.0
    
    # Calculate overall quality score
    overall_quality = (
        quality_metrics.get('completeness', 0.0) * 0.4 +
        quality_metrics.get('coordinate_validity', 0.0) * 0.3 +
        quality_metrics.get('description_quality', 0.0) * 0.3
    )
    quality_metrics['overall_quality'] = overall_quality
    
    return quality_metrics

def reliability_weighting(confidence_scores, reliability_factors):
    """
    Apply reliability weighting to confidence scores.
    
    Args:
        confidence_scores (dict): Dictionary of confidence scores by module
        reliability_factors (dict): Dictionary of reliability factors by module
        
    Returns:
        dict: Reliability-weighted confidence scores
    """
    weighted_scores = {}
    
    for module, score in confidence_scores.items():
        reliability = reliability_factors.get(module, 0.5)  # Default to 0.5 if not provided
        weighted_scores[module] = score * reliability
    
    return weighted_scores