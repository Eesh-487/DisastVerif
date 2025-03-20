"""
Fusion Logic Module

This module implements the weighted aggregation and ensemble scoring.
It also supports a dynamic fusion technique that adjusts weights based on module quality factors.
"""

def weighted_aggregation(geospatial_score, text_score, news_score, weights):
    """
    Aggregate individual module scores using specified weights.
    
    Parameters:
        geospatial_score (float): Score from the geospatial module.
        text_score (float): Score from the text similarity module.
        news_score (float): Score from the news verification module.
        weights (dict): A dictionary with keys 'geospatial', 'text', and 'news' representing their weights.
    
    Returns:
        float: The aggregated score.
    """
    return (weights['geospatial'] * geospatial_score +
            weights['text'] * text_score +
            weights['news'] * news_score)

def dynamic_fusion(module_scores, quality_factors):
    """
    Dynamically adjust weights based on provided quality factors.
    
    We compute normalized weights from quality factors and aggregate the scores.
    
    Parameters:
        module_scores (dict): Dictionary of raw scores from each module.
                              Example: {'geospatial': 0.8, 'text': 0.6, 'news': 0.9}
        quality_factors (dict): Dictionary of quality factors for each module.
                                Example: {'geospatial': 0.7, 'text': 0.9, 'news': 0.8}
    
    Returns:
        float: The ensemble score after dynamic fusion.
    """
    total_quality = sum(quality_factors.values())
    dynamic_weights = {module: quality_factors[module] / total_quality for module in quality_factors}
    return weighted_aggregation(module_scores['geospatial'],
                                module_scores['text'],
                                module_scores['news'],
                                dynamic_weights)

def ensemble_scoring(module_scores, quality_factors=None):
    """
    Compute the final ensemble score.

    If quality_factors are provided, use dynamic fusion; otherwise, use fixed weights.

    Parameters:
        module_scores (dict): Raw scores from modules.
        quality_factors (dict, optional): Quality measures for each module.

    Returns:
        float: The ensemble score.
    """
    if quality_factors:
        return dynamic_fusion(module_scores, quality_factors)
    # Default fixed weights if dynamic factors not provided.
    default_weights = {'geospatial': 0.4, 'text': 0.3, 'news': 0.3}
    return weighted_aggregation(module_scores['geospatial'],
                                module_scores['text'],
                                module_scores['news'],
                                default_weights)

if __name__ == '__main__':
    # Sample usage for ensemble scoring:
    sample_module_scores = {
        'geospatial': 0.8,
        'text': 0.6,
        'news': 0.9
    }
    # Fixed weight ensemble score.
    fixed_score = ensemble_scoring(sample_module_scores)
    print("Ensemble score with fixed weights:", fixed_score)

    # Dynamic fusion example.
    sample_quality_factors = {
        'geospatial': 0.7,
        'text': 0.9,
        'news': 0.8
    }
    dynamic_score = ensemble_scoring(sample_module_scores, sample_quality_factors)
    print("Ensemble score with dynamic fusion:", dynamic_score)