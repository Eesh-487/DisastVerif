"""
Utility functions for geospatial analysis in the Disaster Verifier system.
"""

import math
import numpy as np

def haversine_distance(point1, point2):
    """
    Calculate the great-circle distance between two points on the Earth's surface
    given their latitude and longitude in decimal degrees.
    
    Args:
        point1: Tuple of (latitude, longitude) for the first point
        point2: Tuple of (latitude, longitude) for the second point
        
    Returns:
        Distance between the points in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1 = point1
    lat2, lon2 = point2
    
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Difference in coordinates
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    return distance

def calculate_proximity_threshold(coordinates, percentile=50):
    """
    Calculate a suitable proximity threshold based on the distribution of distances
    between points.
    
    Args:
        coordinates: List of (latitude, longitude) tuples
        percentile: Percentile of the distance distribution to use (0-100)
        
    Returns:
        Proximity threshold in kilometers
    """
    if len(coordinates) < 2:
        return 100.0  # Default threshold if not enough points
        
    # Calculate distances between all pairs of points
    distances = []
    for i in range(len(coordinates)):
        for j in range(i+1, len(coordinates)):
            dist = haversine_distance(coordinates[i], coordinates[j])
            distances.append(dist)
            
    # Calculate the percentile value
    threshold = np.percentile(distances, percentile)
    
    # Apply constraints: minimum 10km, maximum 500km
    threshold = max(10.0, min(500.0, threshold))
    
    return threshold

def find_nearest_events(input_coords, events, max_distance=None, max_results=5):
    """
    Find the nearest events to the input coordinates.
    
    Args:
        input_coords: Tuple of (latitude, longitude) for the input location
        events: List of event dictionaries with 'coords' key
        max_distance: Maximum distance in kilometers (optional)
        max_results: Maximum number of results to return
        
    Returns:
        List of (event, distance) tuples sorted by distance
    """
    if not events:
        return []
        
    # Calculate distance to each event
    event_distances = []
    for event in events:
        if 'coords' in event:
            distance = haversine_distance(input_coords, event['coords'])
            event_distances.append((event, distance))
    
    # Sort by distance
    event_distances.sort(key=lambda x: x[1])
    
    # Filter by maximum distance if specified
    if max_distance is not None:
        event_distances = [(e, d) for e, d in event_distances if d <= max_distance]
    
    # Return the top results
    return event_distances[:max_results]

def cluster_events(events, distance_threshold):
    """
    Cluster events based on proximity.
    
    Args:
        events: List of event dictionaries with 'coords' key
        distance_threshold: Maximum distance for events to be considered in the same cluster
        
    Returns:
        List of lists, where each inner list contains events in a cluster
    """
    if not events:
        return []
        
    # Initialize clusters
    clusters = []
    unclustered = events.copy()
    
    while unclustered:
        # Start a new cluster with the first unclustered event
        current_cluster = [unclustered.pop(0)]
        
        # Find all events within the threshold distance of any event in the current cluster
        i = 0
        while i < len(unclustered):
            event = unclustered[i]
            # Check if event is close to any event in the current cluster
            for cluster_event in current_cluster:
                distance = haversine_distance(event['coords'], cluster_event['coords'])
                if distance <= distance_threshold:
                    # Add to current cluster and remove from unclustered
                    current_cluster.append(event)
                    unclustered.pop(i)
                    # Reset counter because we've modified the list
                    i = 0
                    break
            else:
                # If we didn't find a match, move to the next event
                i += 1
        
        # Add the completed cluster to our list of clusters
        clusters.append(current_cluster)
    
    return clusters

def is_likely_disaster_area(point, known_disaster_points, threshold=100.0):
    """
    Determine if a point is in an area likely to be affected by a disaster
    based on proximity to known disaster points.
    
    Args:
        point: Tuple of (latitude, longitude) to check
        known_disaster_points: List of (latitude, longitude) tuples for known disasters
        threshold: Distance threshold in kilometers
        
    Returns:
        Boolean indicating if the point is likely in a disaster area
    """
    if not known_disaster_points:
        return False
        
    # Find the distance to the closest known disaster point
    min_distance = min(haversine_distance(point, kdp) for kdp in known_disaster_points)
    
    # Return True if the point is within the threshold distance
    return min_distance <= threshold