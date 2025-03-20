"""
Data Quality Module

This module implements robust cross-validation techniques to assess the consistency
of the individual module scores across multiple samples. Lower variability indicates
more reliable module performance.
"""

import numpy as np

def cross_validation_consistency(module_score_samples):
    """
    Assess the consistency of module scores across multiple samples by computing
    a simple consistency metric based on the standard deviation.
    
    Parameters:
        module_score_samples (list): A list of dictionaries where each dictionary contains
                                     module scores with keys 'geospatial', 'text', and 'news'.
                                     Example: [{'geospatial': 0.8, 'text': 0.6, 'news': 0.9}, ...]
    
    Returns:
        dict: A consistency metric for each module. Higher value indicates greater consistency.
              Computed as 1 / (1 + std), where std is the standard deviation of the scores.
    """
    modules = ['geospatial', 'text', 'news']
    consistency = {}
    for mod in modules:
        scores = [sample[mod] for sample in module_score_samples]
        std_score = np.std(scores)
        consistency[mod] = 1 / (1 + std_score)
    return consistency

if __name__ == '__main__':
    # Example usage: list of module scores from several samples.
    samples = [
        {'geospatial': 0.8, 'text': 0.6, 'news': 0.9},
        {'geospatial': 0.75, 'text': 0.65, 'news': 0.88},
        {'geospatial': 0.82, 'text': 0.63, 'news': 0.91},
        {'geospatial': 0.78, 'text': 0.66, 'news': 0.87},
    ]
    consistency_metrics = cross_validation_consistency(samples)
    print("Consistency Metrics:", consistency_metrics)