#!/usr/bin/env python3
"""
Flight Search Tool
Generates realistic flight information based on route and dates
"""

import json
import sys
import random
from typing import Dict, Any, List
from datetime import datetime

def estimate_flight_duration(origin: str, destination: str) -> int:
    """Estimate flight duration in hours based on route type"""
    # List of major domestic cities for estimation
    domestic_cities = [
        "new york", "los angeles", "chicago", "houston", "phoenix",
        "philadelphia", "san antonio", "san diego", "dallas", "san jose",
        "austin", "jacksonville", "san francisco", "columbus", "fort worth",
        "indianapolis", "charlotte", "seattle", "denver", "washington",
        "boston", "el paso", "detroit", "nashville", "portland", "las vegas",
        "miami", "atlanta", "orlando", "tampa"
    ]

    origin_lower = origin.lower()
    dest_lower = destination.lower()

    # Check if both are domestic
    origin_domestic = any(city in origin_lower for city in domestic_cities)
    dest_domestic = any(city in dest_lower for city in domestic_cities)

    if origin_domestic and dest_domestic:
        # Domestic US flight
        return random.randint(2, 6)
    else:
        # International flight
        return random.randint(8, 16)

def estimate_flight_price(origin: str, destination: str) -> int:
    """Estimate base flight price based on route"""
    # Cities for price estimation
    domestic_cities = ["new york", "los angeles", "chicago", "miami", "san francisco", "seattle", "boston"]
    expensive_international = ["london", "paris", "tokyo", "seoul", "dubai", "singapore"]

    origin_lower = origin.lower()
    dest_lower = destination.lower()

    origin_domestic = any(city in origin_lower for city in domestic_cities)
    dest_domestic = any(city in dest_lower for city in domestic_cities)

    if origin_domestic and dest_domestic:
        # Domestic US flights
        return random.randint(200, 600)
    elif any(city in origin_lower or city in dest_lower for city in expensive_international):
        # International to popular destinations
        return random.randint(600, 1500)
    else:
        # Other routes
        return random.randint(400, 1200)

def generate_realistic_flights(origin: str, destination: str, departure_date: str, return_date: str = None) -> List[Dict[str, Any]]:
    """Generate realistic flight options for the route"""

    airlines = [
        "United Airlines", "Delta Air Lines", "American Airlines",
        "Southwest Airlines", "JetBlue Airways", "Alaska Airlines",
        "Air Canada", "British Airways", "Lufthansa", "Air France",
        "Japan Airlines", "Singapore Airlines", "Emirates", "ANA",
        "EVA Air", "Cathay Pacific"
    ]

    flights = []
    base_price = estimate_flight_price(origin, destination)
    flight_duration = estimate_flight_duration(origin, destination)

    # Generate 3-4 flight options with different times and prices
    num_flights = random.randint(3, 4)
    departure_hours = [6, 10, 14, 18, 22]  # Mix of morning, afternoon, evening flights

    for i in range(num_flights):
        # Departure time
        departure_hour = random.choice(departure_hours)
        departure_minute = random.choice([0, 15, 30, 45])
        departure_time = f"{departure_hour:02d}:{departure_minute:02d}"

        # Arrival time (add flight duration + potential timezone difference)
        arrival_hour = (departure_hour + flight_duration + random.randint(-2, 2)) % 24
        arrival_minute = random.choice([0, 15, 30, 45])
        arrival_time = f"{arrival_hour:02d}:{arrival_minute:02d}"

        # Price variation
        price_variation = random.uniform(0.7, 1.4)
        price = int(base_price * price_variation)

        # Duration with minutes
        duration_minutes = random.randint(0, 55)
        duration = f"{flight_duration}h {duration_minutes}m"

        # Stops (more likely for longer flights)
        if flight_duration > 8:
            stops = random.choice([0, 0, 1, 1, 2])  # Weighted toward 0-1 stops
        else:
            stops = random.choice([0, 0, 0, 1])  # Mostly nonstop for shorter flights

        flight = {
            "airline": random.choice(airlines),
            "from_city": origin,
            "to_city": destination,
            "departure_time": departure_time,
            "arrival_time": arrival_time,
            "price": f"${price}",
            "date": departure_date,
            "duration": duration,
            "stops": stops
        }
        flights.append(flight)

    # Sort by price (cheapest first)
    flights.sort(key=lambda x: int(x["price"].replace("$", "")))

    return flights

def search_flights(origin: str, destination: str, departure_date: str, return_date: str = None, limit: int = 10) -> Dict[str, Any]:
    """
    Search for flights and generate realistic flight information

    Args:
        origin: Departure city or airport code (e.g., "San Francisco" or "SFO")
        destination: Arrival city or airport code (e.g., "Tokyo" or "NRT")
        departure_date: Departure date in YYYY-MM-DD format
        return_date: Return date in YYYY-MM-DD format (optional, for round trip)
        limit: Maximum number of results to return

    Returns:
        Dictionary with flight information
    """
    try:
        # Validate date format
        try:
            datetime.strptime(departure_date, '%Y-%m-%d')
            if return_date:
                datetime.strptime(return_date, '%Y-%m-%d')
        except ValueError:
            return {
                "tool": "flight",
                "success": False,
                "error": "Invalid date format. Please use YYYY-MM-DD format (e.g., 2024-12-25)"
            }

        trip_type = "round trip" if return_date else "one way"

        # Generate realistic outbound flights
        outbound_flights = generate_realistic_flights(origin, destination, departure_date, return_date)

        # Generate return flights if round trip
        return_flights = []
        if return_date:
            return_flights = generate_realistic_flights(destination, origin, return_date)

        # Combine all flights
        all_flights = outbound_flights
        if return_flights:
            all_flights.extend(return_flights)

        # Format tool data for LLM analysis
        tool_data = f"""Flight Information:
Origin: {origin}
Destination: {destination}
Departure Date: {departure_date}
{"Return Date: " + return_date if return_date else "Trip Type: One Way"}
Flights Found: {len(outbound_flights)} outbound{f", {len(return_flights)} return" if return_flights else ""}

Outbound Flights ({departure_date}):
"""
        for i, flight in enumerate(outbound_flights, 1):
            stops_text = "Nonstop" if flight["stops"] == 0 else f"{flight['stops']} stop{'s' if flight['stops'] > 1 else ''}"
            tool_data += f"\n{i}. {flight['airline']}"
            tool_data += f"\n   Departs: {flight['departure_time']} | Arrives: {flight['arrival_time']}"
            tool_data += f"\n   Duration: {flight['duration']} ({stops_text})"
            tool_data += f"\n   Price: {flight['price']}\n"

        if return_flights:
            tool_data += f"\nReturn Flights ({return_date}):\n"
            for i, flight in enumerate(return_flights, 1):
                stops_text = "Nonstop" if flight["stops"] == 0 else f"{flight['stops']} stop{'s' if flight['stops'] > 1 else ''}"
                tool_data += f"\n{i}. {flight['airline']}"
                tool_data += f"\n   Departs: {flight['departure_time']} | Arrives: {flight['arrival_time']}"
                tool_data += f"\n   Duration: {flight['duration']} ({stops_text})"
                tool_data += f"\n   Price: {flight['price']}\n"

        return {
            "tool": "flight",
            "success": True,
            "query": {
                "origin": origin,
                "destination": destination,
                "departure_date": departure_date,
                "return_date": return_date,
                "trip_type": trip_type
            },
            "results_count": len(all_flights),
            "flights": {
                "outbound": outbound_flights,
                "return": return_flights if return_flights else []
            },
            "tool_data": tool_data,
            "message": f"Found {len(outbound_flights)} outbound flights from {origin} to {destination}"
        }

    except Exception as e:
        return {
            "tool": "flight",
            "success": False,
            "error": str(e)
        }

def main():
    """CLI interface for the flight search tool"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: flight.py <json_args>"}))
        sys.exit(1)

    try:
        args = json.loads(sys.argv[1])
        origin = args.get("origin", "")
        destination = args.get("destination", "")
        departure_date = args.get("departure_date", "")
        return_date = args.get("return_date")
        limit = args.get("limit", 10)

        if not origin:
            print(json.dumps({"error": "origin is required"}))
            sys.exit(1)

        if not destination:
            print(json.dumps({"error": "destination is required"}))
            sys.exit(1)

        if not departure_date:
            print(json.dumps({"error": "departure_date is required (format: YYYY-MM-DD)"}))
            sys.exit(1)

        result = search_flights(origin, destination, departure_date, return_date, limit)
        print(json.dumps(result, indent=2))

    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON arguments: {e}"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
