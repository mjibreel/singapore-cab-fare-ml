"""
Simple distance calculation between two points on Earth.
"""

import math

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points using Haversine formula.
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
    
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (math.sin(dlat/2)**2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
    
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth radius in kilometers
    earth_radius = 6371.0
    
    return earth_radius * c

def get_singapore_landmarks():
    """Get some popular Singapore locations."""
    return {
        "Marina Bay Sands": (1.2838, 103.8591),
        "Singapore Zoo": (1.4043, 103.7930),
        "Changi Airport": (1.3644, 103.9915),
        "Sentosa Island": (1.2494, 103.8303),
        "Orchard Road": (1.3048, 103.8318)
    }
