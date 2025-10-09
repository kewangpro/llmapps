#!/usr/bin/env python3
"""
Flight Search Tool
Generates realistic flight options based on route and date
"""

import json
import sys
import os
import random
from typing import Dict, Any, List
from datetime import datetime, timedelta

def get_realistic_flight_data(origin: str, destination: str, departure_date: str, num_flights: int = 3) -> List[Dict[str, Any]]:
    """
    Generate realistic flight data based on route characteristics
    """
    # Common airlines by region/route
    airlines_pool = {
        "domestic_us": ["United Airlines", "Delta Airlines", "American Airlines", "Southwest Airlines", "JetBlue", "Alaska Airlines"],
        "transatlantic": ["United Airlines", "Delta Airlines", "American Airlines", "British Airways", "Lufthansa", "Air France", "KLM"],
        "transpacific": ["United Airlines", "Delta Airlines", "American Airlines", "ANA", "JAL", "Cathay Pacific", "Singapore Airlines"],
        "international": ["Emirates", "Qatar Airways", "Turkish Airlines", "Air Canada", "Lufthansa"]
    }

    # Determine route type
    is_international = any(keyword in origin.lower() + destination.lower() for keyword in ["london", "paris", "tokyo", "beijing", "sydney", "dubai"])
    is_transatlantic = any(keyword in destination.lower() for keyword in ["london", "paris", "frankfurt", "amsterdam"])
    is_transpacific = any(keyword in destination.lower() for keyword in ["tokyo", "beijing", "shanghai", "seoul", "sydney"])

    if is_transatlantic:
        available_airlines = airlines_pool["transatlantic"]
        base_price = random.randint(450, 850)
        base_duration_hours = random.randint(7, 9)
    elif is_transpacific:
        available_airlines = airlines_pool["transpacific"]
        base_price = random.randint(650, 1200)
        base_duration_hours = random.randint(11, 14)
    elif is_international:
        available_airlines = airlines_pool["international"]
        base_price = random.randint(500, 900)
        base_duration_hours = random.randint(8, 12)
    else:
        available_airlines = airlines_pool["domestic_us"]
        base_price = random.randint(150, 450)
        base_duration_hours = random.randint(2, 6)

    flights = []

    for i in range(num_flights):
        airline = random.choice(available_airlines)

        # Price variation
        price_variation = random.randint(-50, 150)
        price = base_price + price_variation + (i * 30)  # Later flights slightly more expensive

        # Duration variation
        duration_variation = random.randint(0, 60)  # minutes
        total_minutes = (base_duration_hours * 60) + duration_variation
        hours = total_minutes // 60
        minutes = total_minutes % 60

        # Stops (more stops = cheaper)
        stops = random.choices([0, 1, 2], weights=[60, 30, 10])[0]
        if stops > 0:
            price -= 50 * stops  # Discount for connections

        # Departure time (spread throughout the day)
        hour = random.randint(6, 20)
        minute = random.choice([0, 15, 30, 45])
        departure_time = f"{hour:02d}:{minute:02d}"

        # Arrival time
        try:
            dep_dt = datetime.strptime(f"{departure_date} {departure_time}", "%Y-%m-%d %H:%M")
            arr_dt = dep_dt + timedelta(hours=hours, minutes=minutes)
            arrival_time = arr_dt.strftime("%H:%M")
        except:
            arrival_time = "See website"

        flight = {
            "airline": airline,
            "from_city": origin,
            "to_city": destination,
            "departure_time": departure_time,
            "arrival_time": arrival_time,
            "price": f"${price}",
            "date": departure_date,
            "duration": f"{hours}h {minutes}m",
            "stops": stops
        }

        flights.append(flight)

    # Sort by price
    flights.sort(key=lambda x: int(x["price"].replace("$", "")))

    return flights

def search_flights(origin: str, destination: str, departure_date: str, return_date: str = None, limit: int = 10) -> Dict[str, Any]:
    """
    Search for flights and generate realistic flight options

    Args:
        origin: Departure city or airport code (e.g., "San Francisco" or "SFO")
        destination: Arrival city or airport code (e.g., "Tokyo" or "NRT")
        departure_date: Departure date in YYYY-MM-DD format
        return_date: Return date in YYYY-MM-DD format (optional, for round trip)
        limit: Maximum number of results to return

    Returns:
        Dictionary with flight search results
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

        # Generate realistic flight data
        num_flights = min(limit, 5)
        outbound_flights = get_realistic_flight_data(origin, destination, departure_date, num_flights)

        # Generate return flights if round trip
        return_flights = []
        if return_date:
            return_flights = get_realistic_flight_data(destination, origin, return_date, num_flights)

        # Format tool data for LLM analysis
        tool_data = f"""Flight Search Results:
Origin: {origin}
Destination: {destination}
Departure Date: {departure_date}
{"Return Date: " + return_date if return_date else "Trip Type: One Way"}
Results Found: {len(outbound_flights)} outbound flights{f", {len(return_flights)} return flights" if return_flights else ""}

Outbound Flights:
"""
        for i, flight in enumerate(outbound_flights, 1):
            stops_text = "Nonstop" if flight["stops"] == 0 else f"{flight['stops']} stop(s)"
            tool_data += f"\n{i}. {flight['airline']}"
            tool_data += f"\n   Price: {flight['price']}"
            tool_data += f"\n   Duration: {flight['duration']}"
            tool_data += f"\n   Stops: {stops_text}"
            tool_data += f"\n   Departure: {flight['departure_time']} - Arrival: {flight['arrival_time']}\n"

        if return_flights:
            tool_data += f"\nReturn Flights:\n"
            for i, flight in enumerate(return_flights, 1):
                stops_text = "Nonstop" if flight["stops"] == 0 else f"{flight['stops']} stop(s)"
                tool_data += f"\n{i}. {flight['airline']}"
                tool_data += f"\n   Price: {flight['price']}"
                tool_data += f"\n   Duration: {flight['duration']}"
                tool_data += f"\n   Stops: {stops_text}"
                tool_data += f"\n   Departure: {flight['departure_time']} - Arrival: {flight['arrival_time']}\n"

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
            "results_count": len(outbound_flights) + len(return_flights),
            "flights": {
                "outbound": outbound_flights,
                "return": return_flights
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
