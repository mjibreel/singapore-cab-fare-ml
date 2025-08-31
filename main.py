"""
Singapore Taxi Fare Prediction - Main Application
Simple command-line application to predict taxi fares.
"""

from datetime import datetime
from haversine import calculate_distance, get_singapore_landmarks
from fare_calculator import (
    calculate_fare, 
    estimate_duration, 
    is_peak_hour, 
    is_weekend
)

def main():
    """Main application function."""
    print("🚕 Singapore Taxi Fare Predictor")
    print("=" * 40)
    
    # Show available landmarks
    landmarks = get_singapore_landmarks()
    print("\nPopular Singapore locations:")
    for i, (name, coords) in enumerate(landmarks.items(), 1):
        print(f"{i}. {name}")
    
    print("\nChoose your option:")
    print("1. Use popular landmarks")
    print("2. Enter custom coordinates")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        # Use landmarks
        print("\nPopular Singapore locations:")
        for i, (name, coords) in enumerate(landmarks.items(), 1):
            print(f"{i}. {name}")
        
        print("\nSelect pickup location:")
        pickup_choice = int(input("Enter pickup location number: ")) - 1
        pickup_name = list(landmarks.keys())[pickup_choice]
        pickup_lat, pickup_lon = list(landmarks.values())[pickup_choice]
        
        print(f"\nPickup: {pickup_name} ({pickup_lat}, {pickup_lon})")
        
        print("\nSelect dropoff location:")
        print("Popular Singapore locations:")
        for i, (name, coords) in enumerate(landmarks.items(), 1):
            print(f"{i}. {name}")
        
        dropoff_choice = int(input("Enter dropoff location number: ")) - 1
        dropoff_name = list(landmarks.keys())[dropoff_choice]
        dropoff_lat, dropoff_lon = list(landmarks.values())[dropoff_choice]
        
        print(f"Dropoff: {dropoff_name} ({dropoff_lat}, {dropoff_lon})")
        
    elif choice == "2":
        # Custom coordinates
        print("\nEnter custom coordinates:")
        print("Singapore coordinates range:")
        print("- Latitude: 1.15°N to 1.47°N (North-South)")
        print("- Longitude: 103.6°E to 104.1°E (East-West)")
        print("\nExample coordinates:")
        print("- Marina Bay Sands: (1.2838, 103.8591)")
        print("- Singapore Zoo: (1.4043, 103.7930)")
        print("- Changi Airport: (1.3644, 103.9915)")
        print("- Sentosa Island: (1.2494, 103.8303)")
        print("- Orchard Road: (1.3048, 103.8318)")
        
        print("\nEnter pickup coordinates:")
        pickup_lat = float(input("Pickup Latitude (1.15 to 1.47): "))
        pickup_lon = float(input("Pickup Longitude (103.6 to 104.1): "))
        
        print("\nEnter dropoff coordinates:")
        dropoff_lat = float(input("Dropoff Latitude (1.15 to 1.47): "))
        dropoff_lon = float(input("Dropoff Longitude (103.6 to 104.1): "))
        
        pickup_name = f"({pickup_lat}, {pickup_lon})"
        dropoff_name = f"({dropoff_lat}, {dropoff_lon})"
    
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Get trip details
    print("\nEnter trip details:")
    pickup_date = input("Pickup date (YYYY-MM-DD) or press Enter for today: ").strip()
    if not pickup_date:
        pickup_date = datetime.now().strftime("%Y-%m-%d")
    
    print("\nPeak Hours Information:")
    print("🕐 Peak Hours (higher fares):")
    print("   - Morning: 7:00 AM - 9:00 AM (7:00 - 9:00)")
    print("   - Evening: 6:00 PM - 8:00 PM (18:00 - 20:00)")
    print("   - Model 1: +20% surcharge during peak hours")
    print("   - Model 2: +25% surcharge during peak hours")
    
    pickup_time = input("\nPickup time (HH:MM) or press Enter for now: ").strip()
    if not pickup_time:
        pickup_time = datetime.now().strftime("%H:%M")
    
    passenger_count = int(input("Number of passengers (1-4): "))
    
    # Calculate distance
    distance = calculate_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    
    # Estimate duration
    duration = estimate_duration(distance)
    
    # Parse date and time
    try:
        pickup_datetime = datetime.strptime(f"{pickup_date} {pickup_time}", "%Y-%m-%d %H:%M")
        hour = pickup_datetime.hour
        day_of_week = pickup_datetime.weekday()
    except ValueError:
        print("Invalid date/time format. Using current time.")
        pickup_datetime = datetime.now()
        hour = pickup_datetime.hour
        day_of_week = pickup_datetime.weekday()
    
    # Check if peak hour or weekend
    peak = is_peak_hour(hour)
    weekend = is_weekend(day_of_week)
    
    # Calculate fare using TWO MODELS
    # Model 1: Basic formula
    basic_fare = calculate_fare(distance, duration, peak, weekend)
    
    # Model 2: Alternative pricing model (different rates)
    alt_base_fare = 4.00  # Higher base fare
    alt_distance_cost = 1.80 * distance  # Higher per km rate
    alt_time_cost = 0.35 * duration  # Lower per minute rate
    alt_fare = alt_base_fare + alt_distance_cost + alt_time_cost
    
    if peak:
        alt_fare *= 1.25  # 25% peak surcharge
    if weekend:
        alt_fare *= 1.15  # 15% weekend surcharge
    
    alt_fare = round(alt_fare, 2)
    
    # Display results
    print("\n" + "=" * 50)
    print("🚕 TRIP SUMMARY")
    print("=" * 50)
    print(f"Pickup: {pickup_name}")
    print(f"Dropoff: {dropoff_name}")
    print(f"Date: {pickup_date}")
    print(f"Time: {pickup_time}")
    print(f"Passengers: {passenger_count}")
    print(f"Distance: {distance:.2f} km")
    print(f"Duration: {duration:.1f} minutes")
    print(f"Peak Hours: {'Yes' if peak else 'No'}")
    print(f"Weekend: {'Yes' if weekend else 'No'}")
    print("-" * 50)
    print("💰 FARE PREDICTIONS (TWO MODELS)")
    print("-" * 50)
    print(f"Model 1 (Standard): ${basic_fare:.2f} SGD")
    print(f"Model 2 (Premium): ${alt_fare:.2f} SGD")
    print(f"Average: ${((basic_fare + alt_fare) / 2):.2f} SGD")
    print("=" * 50)
    
    # Show fare breakdown for Model 1
    print("\nModel 1 Breakdown (Standard):")
    base_fare = 3.20
    distance_cost = 1.50 * distance
    time_cost = 0.40 * duration
    
    print(f"Base fare: ${base_fare:.2f}")
    print(f"Distance cost ({distance:.2f} km): ${distance_cost:.2f}")
    print(f"Time cost ({duration:.1f} min): ${time_cost:.2f}")
    
    if peak:
        print("Peak hour surcharge (20%): +${:.2f}".format(basic_fare - (base_fare + distance_cost + time_cost)))
    if weekend:
        print("Weekend surcharge (10%): +${:.2f}".format(basic_fare - (base_fare + distance_cost + time_cost)))
    
    print(f"Total: ${basic_fare:.2f}")
    
    # Show fare breakdown for Model 2
    print("\nModel 2 Breakdown (Premium):")
    print(f"Base fare: ${alt_base_fare:.2f}")
    print(f"Distance cost ({distance:.2f} km): ${alt_distance_cost:.2f}")
    print(f"Time cost ({duration:.1f} min): ${alt_time_cost:.2f}")
    
    if peak:
        print("Peak hour surcharge (25%): +${:.2f}".format(alt_fare - (alt_base_fare + alt_distance_cost + alt_time_cost)))
    if weekend:
        print("Weekend surcharge (15%): +${:.2f}".format(alt_fare - (alt_base_fare + alt_distance_cost + alt_time_cost)))
    
    print(f"Total: ${alt_fare:.2f}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please try again.")
