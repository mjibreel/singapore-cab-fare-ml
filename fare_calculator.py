"""
Simple taxi fare calculation for Singapore.
"""

def calculate_fare(distance_km, duration_minutes, is_peak_hour=False, is_weekend=False):
    """
    Calculate taxi fare using simple formula.
    
    Args:
        distance_km: Trip distance in kilometers
        duration_minutes: Trip duration in minutes
        is_peak_hour: Whether it's peak hours (7-9 AM or 6-8 PM)
        is_weekend: Whether it's weekend
    
    Returns:
        Fare amount in SGD
    """
    # Base fare
    base_fare = 3.20
    
    # Distance cost: $1.50 per km
    distance_cost = 1.50 * distance_km
    
    # Time cost: $0.40 per minute
    time_cost = 0.40 * duration_minutes
    
    # Calculate total
    total_fare = base_fare + distance_cost + time_cost
    
    # Add peak hour surcharge (20% more during peak hours)
    if is_peak_hour:
        total_fare *= 1.20
    
    # Add weekend surcharge (10% more on weekends)
    if is_weekend:
        total_fare *= 1.10
    
    # Round to 2 decimal places
    return round(total_fare, 2)

def estimate_duration(distance_km):
    """
    Estimate trip duration based on distance.
    Assumes average speed of 30 km/h in Singapore.
    
    Args:
        distance_km: Trip distance in kilometers
    
    Returns:
        Estimated duration in minutes
    """
    # Average speed: 30 km/h
    speed_kmh = 30
    
    # Calculate time in hours, then convert to minutes
    time_hours = distance_km / speed_kmh
    time_minutes = time_hours * 60
    
    return round(time_minutes, 1)

def is_peak_hour(hour):
    """Check if given hour is peak hour."""
    return (7 <= hour <= 9) or (18 <= hour <= 20)

def is_weekend(day_of_week):
    """Check if given day is weekend (0=Monday, 6=Sunday)."""
    return day_of_week in [5, 6]  # Saturday, Sunday
