# This file contains helper functions for geospatial calculations.

import math

def haversine_distance(coord1, coord2):
    """
    Compute the Haversine distance (in kilometers) between two (lat, lon) pairs.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2 
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371  # Earth radius in kilometers
    return c * r

# You can add additional utility functions here if needed.