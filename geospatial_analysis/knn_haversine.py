# This file implements the traditional Haversine + KNN approach.
import math
from collections import Counter

def haversine_distance(coord1, coord2):
    """
    Compute the Haversine distance (in kilometers) between two (lat, lon) pairs.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    # Haversine formula
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2 
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371  # Radius of earth in kilometers
    return c * r

def knn_haversine(train_coords, train_labels, test_coord, k=5):
    """
    Given training coordinates and labels along with a single test coordinate,
    compute Haversine distance to each training coordinate. Return the label that has
    the majority vote from the k-nearest neighbors.
    
    train_coords: list of (lat, lon) tuples
    train_labels: list of labels (same order as train_coords)
    test_coord: a (lat, lon) tuple of the test point
    k: number of neighbors to consider
    """
    distances = []
    for coord in train_coords:
        d = haversine_distance(coord, test_coord)
        distances.append(d)
    
    # Combine distances with labels and sort by distance
    neighbor_info = sorted(zip(distances, train_labels), key=lambda x: x[0])
    k_neighbors = neighbor_info[:k]
    
    # Get the majority label
    labels = [label for _, label in k_neighbors]
    majority_label = Counter(labels).most_common(1)[0][0]
    return majority_label

# Example usage:
if __name__ == '__main__':
    # Dummy training data: coordinates and labels
    train_coords = [(40.7128, -74.0060), (34.0522, -118.2437), (41.8781, -87.6298)]
    train_labels = ['disaster_A', 'disaster_B', 'disaster_A']
    
    # Test coordinate
    test_coord = (40.730610, -73.935242)  # somewhere in New York
    predicted_label = knn_haversine(train_coords, train_labels, test_coord, k=3)
    print("Predicted label using Haversine + KNN:", predicted_label)