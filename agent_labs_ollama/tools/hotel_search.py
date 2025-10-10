#!/usr/bin/env python3
"""
Hotel Search Tool
Generates realistic hotel options based on location and dates
"""

import json
import sys
import os
import random
from typing import Dict, Any, List
from datetime import datetime, timedelta

def get_realistic_hotel_data(location: str, check_in: str, check_out: str, num_hotels: int = 5) -> List[Dict[str, Any]]:
    """
    Generate realistic hotel data based on location and dates
    """
    # Hotel chains and boutique options
    hotel_types = {
        "luxury": ["Four Seasons", "Ritz-Carlton", "St. Regis", "Waldorf Astoria", "Park Hyatt", "Mandarin Oriental"],
        "upscale": ["Marriott", "Hilton", "Hyatt Regency", "InterContinental", "Westin", "Sheraton"],
        "mid_range": ["Courtyard by Marriott", "Holiday Inn", "Best Western Plus", "Radisson", "Doubletree"],
        "budget": ["Hampton Inn", "Holiday Inn Express", "Comfort Inn", "La Quinta", "Days Inn"],
        "boutique": ["The Hoxton", "Ace Hotel", "Kimpton", "Aloft", "Moxy"]
    }

    # Amenities
    standard_amenities = ["Free WiFi", "Air Conditioning", "TV"]
    common_amenities = ["Swimming Pool", "Fitness Center", "Restaurant", "Room Service", "Parking"]
    premium_amenities = ["Spa", "Concierge", "Business Center", "Airport Shuttle", "Rooftop Bar"]

    # Determine location premium
    is_major_city = any(keyword in location.lower() for keyword in ["new york", "london", "paris", "tokyo", "san francisco", "los angeles", "miami", "chicago"])
    is_tourist = any(keyword in location.lower() for keyword in ["beach", "resort", "disney", "vegas", "hawaii", "bali"])

    hotels = []

    # Calculate number of nights
    try:
        check_in_dt = datetime.strptime(check_in, "%Y-%m-%d")
        check_out_dt = datetime.strptime(check_out, "%Y-%m-%d")
        nights = (check_out_dt - check_in_dt).days
    except:
        nights = 1

    for i in range(num_hotels):
        # Distribute hotels across categories
        if i == 0:
            category = "luxury"
            base_price = random.randint(400, 800)
            stars = random.choice([4.5, 5.0])
        elif i == 1:
            category = "upscale"
            base_price = random.randint(200, 400)
            stars = random.choice([4.0, 4.5])
        elif i == 2:
            category = "mid_range"
            base_price = random.randint(120, 200)
            stars = random.choice([3.5, 4.0])
        elif i == 3:
            category = "budget"
            base_price = random.randint(60, 120)
            stars = random.choice([3.0, 3.5])
        else:
            category = "boutique"
            base_price = random.randint(150, 350)
            stars = random.choice([4.0, 4.5])

        # Location premium
        if is_major_city:
            base_price = int(base_price * 1.4)
        elif is_tourist:
            base_price = int(base_price * 1.2)

        # Price variation
        price_variation = random.randint(-20, 30)
        price_per_night = base_price + price_variation
        total_price = price_per_night * nights

        # Hotel name
        chain = random.choice(hotel_types[category])
        hotel_name = f"{chain} {location.split(',')[0]}"

        # Rating (out of 5)
        rating = round(stars + random.uniform(-0.3, 0.2), 1)
        rating = min(5.0, max(3.0, rating))

        # Number of reviews
        reviews = random.randint(200, 2000) if category in ["luxury", "upscale"] else random.randint(50, 500)

        # Amenities based on category
        amenities = standard_amenities.copy()
        if category in ["luxury", "upscale"]:
            amenities.extend(random.sample(common_amenities, k=min(4, len(common_amenities))))
            amenities.extend(random.sample(premium_amenities, k=min(3, len(premium_amenities))))
        elif category == "mid_range":
            amenities.extend(random.sample(common_amenities, k=min(3, len(common_amenities))))
        elif category == "boutique":
            amenities.extend(random.sample(common_amenities, k=min(2, len(common_amenities))))
            amenities.extend(random.sample(premium_amenities, k=min(2, len(premium_amenities))))
        else:
            amenities.extend(random.sample(common_amenities, k=min(2, len(common_amenities))))

        # Distance from center
        distance = round(random.uniform(0.2, 5.0), 1) if is_major_city else round(random.uniform(0.1, 2.0), 1)

        # Cancellation policy
        cancellation = "Free cancellation" if random.random() > 0.3 else "Non-refundable"

        hotel = {
            "name": hotel_name,
            "category": category.replace("_", " ").title(),
            "location": location,
            "rating": rating,
            "reviews": reviews,
            "price_per_night": f"${price_per_night}",
            "total_price": f"${total_price}",
            "nights": nights,
            "check_in": check_in,
            "check_out": check_out,
            "amenities": amenities,
            "distance_from_center": f"{distance} miles",
            "cancellation_policy": cancellation
        }

        hotels.append(hotel)

    # Sort by rating (highest first), then by price
    hotels.sort(key=lambda x: (-x["rating"], int(x["price_per_night"].replace("$", ""))))

    return hotels

def search_hotels(location: str, check_in: str, check_out: str, guests: int = 2, limit: int = 10) -> Dict[str, Any]:
    """
    Search for hotels and generate realistic hotel options

    Args:
        location: City or area (e.g., "San Francisco, CA" or "Paris, France")
        check_in: Check-in date in YYYY-MM-DD format
        check_out: Check-out date in YYYY-MM-DD format
        guests: Number of guests (default: 2)
        limit: Maximum number of results to return

    Returns:
        Dictionary with hotel search results
    """
    try:
        # Validate date format
        try:
            check_in_dt = datetime.strptime(check_in, '%Y-%m-%d')
            check_out_dt = datetime.strptime(check_out, '%Y-%m-%d')

            if check_out_dt <= check_in_dt:
                return {
                    "tool": "hotel_search",
                    "success": False,
                    "error": "Check-out date must be after check-in date"
                }
        except ValueError:
            return {
                "tool": "hotel_search",
                "success": False,
                "error": "Invalid date format. Please use YYYY-MM-DD format (e.g., 2024-12-25)"
            }

        # Calculate nights
        nights = (check_out_dt - check_in_dt).days

        # Generate realistic hotel data
        num_hotels = min(limit, 8)
        hotels = get_realistic_hotel_data(location, check_in, check_out, num_hotels)

        # Format tool data for LLM analysis
        tool_data = f"""Hotel Search Results:
Location: {location}
Check-in: {check_in}
Check-out: {check_out}
Nights: {nights}
Guests: {guests}
Results Found: {len(hotels)} hotels

Hotels:
"""
        for i, hotel in enumerate(hotels, 1):
            tool_data += f"\n{i}. {hotel['name']}"
            tool_data += f"\n   Category: {hotel['category']}"
            tool_data += f"\n   Rating: {hotel['rating']}/5.0 ({hotel['reviews']} reviews)"
            tool_data += f"\n   Price: {hotel['price_per_night']}/night (Total: {hotel['total_price']} for {nights} nights)"
            tool_data += f"\n   Distance: {hotel['distance_from_center']} from center"
            tool_data += f"\n   Amenities: {', '.join(hotel['amenities'][:5])}"
            tool_data += f"\n   {hotel['cancellation_policy']}\n"

        return {
            "tool": "hotel_search",
            "success": True,
            "query": {
                "location": location,
                "check_in": check_in,
                "check_out": check_out,
                "nights": nights,
                "guests": guests
            },
            "results_count": len(hotels),
            "hotels": hotels,
            "tool_data": tool_data,
            "message": f"Found {len(hotels)} hotels in {location}"
        }

    except Exception as e:
        return {
            "tool": "hotel_search",
            "success": False,
            "error": str(e)
        }

def main():
    """CLI interface for the hotel search tool"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: hotel_search.py <json_args>"}))
        sys.exit(1)

    try:
        args = json.loads(sys.argv[1])
        location = args.get("location", "")
        check_in = args.get("check_in", "")
        check_out = args.get("check_out", "")
        guests = args.get("guests", 2)
        limit = args.get("limit", 10)

        if not location:
            print(json.dumps({"error": "location is required"}))
            sys.exit(1)

        if not check_in:
            print(json.dumps({"error": "check_in date is required (format: YYYY-MM-DD)"}))
            sys.exit(1)

        if not check_out:
            print(json.dumps({"error": "check_out date is required (format: YYYY-MM-DD)"}))
            sys.exit(1)

        result = search_hotels(location, check_in, check_out, guests, limit)
        print(json.dumps(result, indent=2))

    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON arguments: {e}"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
