#!/usr/bin/env python3
"""
Trip Planner CLI Tool

A command-line interface for planning trips using the Trip Planner API.
Takes trip parameters and displays a complete itinerary.
"""

import argparse
import asyncio
import aiohttp
import json
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plan your trip with AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --origin Seattle --destinations Tokyo --start-date 2025-09-02 --duration 7
  python run.py --origin "San Francisco" --destinations Tokyo Seoul --start-date 2025-10-15 --duration 12 --budget high --mode comprehensive
  python run.py --origin NYC --destinations Paris London --start-date 2025-06-01 --duration 10 --preferences "food, museums" --mode simple
        """
    )
    
    parser.add_argument(
        '--origin', '-o',
        required=True,
        help='Origin city (e.g., "Seattle", "San Francisco")'
    )
    
    parser.add_argument(
        '--destinations', '-d',
        nargs='+',
        required=True,
        help='Destination cities (e.g., Tokyo Seoul, "New York" Paris)'
    )
    
    parser.add_argument(
        '--start-date', '-s',
        required=True,
        help='Trip start date in YYYY-MM-DD format (e.g., 2025-09-02)'
    )
    
    parser.add_argument(
        '--duration', '-t',
        type=int,
        required=True,
        help='Trip duration in days (e.g., 7, 14)'
    )
    
    parser.add_argument(
        '--budget', '-b',
        choices=['low', 'medium', 'high'],
        default='medium',
        help='Budget level (default: medium)'
    )
    
    parser.add_argument(
        '--preferences', '-p',
        default='culture, food',
        help='Trip preferences (default: "culture, food")'
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['simple', 'comprehensive'],
        default='simple',
        help='Collaboration mode: simple (pure LLM reasoning, 30-60s) or comprehensive (5-agent collaboration, 3-5min) (default: simple)'
    )
    
    parser.add_argument(
        '--api-url',
        default='http://localhost:8000',
        help='API base URL (default: http://localhost:8000)'
    )
    
    return parser.parse_args()


def validate_date(date_string: str) -> bool:
    """Validate date format and ensure it's not in the past."""
    try:
        date = datetime.strptime(date_string, '%Y-%m-%d')
        if date < datetime.now():
            print(f"❌ Error: Start date {date_string} is in the past")
            return False
        return True
    except ValueError:
        print(f"❌ Error: Invalid date format '{date_string}'. Use YYYY-MM-DD format")
        return False


def print_header():
    """Print the CLI header."""
    print("🌍 AI Trip Planner")
    print("=" * 50)


def print_trip_request(args):
    """Print the trip request details."""
    print(f"📍 Origin: {args.origin}")
    print(f"🎯 Destinations: {' → '.join(args.destinations)}")
    print(f"📅 Start Date: {args.start_date}")
    print(f"⏱️  Duration: {args.duration} days")
    print(f"💰 Budget: {args.budget}")
    print(f"🎨 Preferences: {args.preferences}")
    print(f"🤖 Mode: {args.mode} ({'pure LLM reasoning, 30-60s' if args.mode == 'simple' else '5-agent collaboration, 3-5min'})")
    print()


async def call_trip_api(args) -> Dict[str, Any]:
    """Call the trip planning API."""
    request_data = {
        "origin": args.origin,
        "destinations": args.destinations,
        "start_date": args.start_date,
        "duration_days": args.duration,
        "budget": args.budget,
        "preferences": args.preferences,
        "collaboration_mode": args.mode
    }
    
    print("🤖 Planning your trip with AI agents...")
    if args.mode == 'simple':
        print("⏳ Simple mode (Pure LLM reasoning): This may take 30-60 seconds for any trip...")
    else:
        print("⏳ Comprehensive mode (5-agent collaboration): This may take 3-5 minutes for detailed multi-agent analysis...")
    print()
    
    # Set timeout based on mode - Simple uses pure LLM (faster), Comprehensive uses multi-agent (longer)
    timeout_seconds = 180 if args.mode == 'simple' else 600  # 3 min simple, 10 min comprehensive
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(
                f"{args.api_url}/plan-trip",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
                    
        except asyncio.TimeoutError:
            raise Exception("Request timed out. The trip planning is taking longer than expected.")
        except aiohttp.ClientError as e:
            raise Exception(f"Connection error: {e}")


def print_flight_info(flight: Dict[str, Any]):
    """Print flight information."""
    from_city = flight.get('from_city', flight.get('from', 'Unknown'))
    to_city = flight.get('to_city', flight.get('to', 'Unknown'))
    airline = flight.get('airline', 'Unknown Airline')
    departure = flight.get('departure_time', 'TBD')
    arrival = flight.get('arrival_time', 'TBD')
    price = flight.get('estimated_price', flight.get('price', 'Price TBD'))
    if isinstance(price, (int, float)):
        price = f"${price:.2f}"
    date = flight.get('date', 'Date TBD')
    
    print(f"  ✈️  {from_city} → {to_city}")
    print(f"      {airline}")
    print(f"      Depart: {departure}, Arrive: {arrival}")
    print(f"      Date: {date}, Price: {price}")
    print()

def print_curated_flight_info(flight: Dict[str, Any], is_primary: bool = False):
    """Print curated flight information with recommendation reason."""
    from_city = flight.get('from_city', flight.get('from', 'Unknown'))
    to_city = flight.get('to_city', flight.get('to', 'Unknown'))
    airline = flight.get('airline', 'Unknown Airline')
    departure = flight.get('departure_time', 'TBD')
    arrival = flight.get('arrival_time', 'TBD')
    price = flight.get('estimated_price', flight.get('price', 'Price TBD'))
    if isinstance(price, (int, float)):
        price = f"${price:.2f}"
    date = flight.get('date', 'Date TBD')
    reason = flight.get('recommendation_reason', '')
    
    prefix = "🌟 " if is_primary else "  ✈️  "
    print(f"{prefix}{from_city} → {to_city}")
    print(f"      {airline}")
    print(f"      Depart: {departure}, Arrive: {arrival}")
    print(f"      Date: {date}, Price: {price}")
    if is_primary and reason:
        print(f"      💡 {reason}")
    print()


def print_hotel_info(hotel: Dict[str, Any]):
    """Print hotel information."""
    name = hotel.get('name', 'Unknown Hotel')
    city = hotel.get('city', 'Unknown City')
    rating = hotel.get('rating', 0)
    price = hotel.get('price_per_night', hotel.get('pricePerNight', 'Price TBD'))
    if isinstance(price, (int, float)):
        price = f"${price:.2f}"
    address = hotel.get('address', '')
    
    print(f"  🏨 {name}")
    print(f"      📍 {city}")
    if rating > 0:
        stars = '⭐' * min(int(rating), 5)
        print(f"      {stars} {rating}/5.0")
    print(f"      💰 {price}/night")
    if address:
        print(f"      📮 {address}")
    print()

def print_curated_hotel_info(hotel: Dict[str, Any], is_primary: bool = False):
    """Print curated hotel information with recommendation reason."""
    name = hotel.get('name', 'Unknown Hotel')
    city = hotel.get('city', 'Unknown City')
    rating = hotel.get('rating', 0)
    price = hotel.get('price_per_night', hotel.get('pricePerNight', 'Price TBD'))
    if isinstance(price, (int, float)):
        price = f"${price:.2f}"
    address = hotel.get('address', '')
    reason = hotel.get('recommendation_reason', '')
    
    prefix = "🌟 " if is_primary else "  🏨 "
    print(f"{prefix}{name}")
    print(f"      📍 {city}")
    if rating > 0:
        stars = '⭐' * min(int(rating), 5)
        print(f"      {stars} {rating}/5.0")
    print(f"      💰 {price}/night")
    if address:
        print(f"      📮 {address}")
    if is_primary and reason:
        print(f"      💡 {reason}")
    print()


def print_daily_plan(day_plan: Dict[str, Any]):
    """Print daily plan information."""
    day = day_plan.get('day', 0)
    date = day_plan.get('date', 'Unknown Date')
    city = day_plan.get('city', 'Unknown City')
    activities = day_plan.get('activities', [])
    transportation = day_plan.get('transportation', '')
    city_tips = day_plan.get('city_tips', [])
    
    print(f"  📅 Day {day} - {city}")
    print(f"      Date: {date}")
    
    if activities:
        print(f"      🎯 Activities:")
        for activity in activities[:3]:  # Show top 3 activities
            print(f"         • {activity}")
    
    if transportation:
        print(f"      🚌 Transportation: {transportation}")
    
    if city_tips:
        print(f"      💡 Tips: {city_tips[0]}")  # Show first tip
    
    print()


def print_itinerary(trip_data: Dict[str, Any]):
    """Print the complete itinerary."""
    print("🎯 Your Complete Itinerary")
    print("=" * 50)
    
    # Trip Overview
    route = trip_data.get('route_order', [])
    total_days = trip_data.get('total_days', 0)
    estimated_budget = trip_data.get('estimated_budget', 'Not specified')
    
    print(f"📍 Route: {' → '.join(route)}")
    print(f"⏱️  Duration: {total_days} days")
    print(f"💰 Estimated Budget: {estimated_budget}")
    print()
    
    # Check if we have curated structures
    curated_flights = trip_data.get('curated_flights', {})
    curated_hotels = trip_data.get('curated_hotels', {})
    
    # Flights - show curated if available, otherwise show all flights
    if curated_flights.get('primary', {}).get('outbound') or curated_flights.get('primary', {}).get('return'):
        print("✈️  RECOMMENDED FLIGHTS")
        print("-" * 20)
        
        # Show primary outbound flight
        outbound = curated_flights.get('primary', {}).get('outbound')
        if outbound:
            print("🛫 Outbound Flight:")
            print_curated_flight_info(outbound, is_primary=True)
        
        # Show primary return flight
        return_flight = curated_flights.get('primary', {}).get('return')
        if return_flight:
            print("🛬 Return Flight:")
            print_curated_flight_info(return_flight, is_primary=True)
        
        # Show alternatives if available
        alternatives = curated_flights.get('alternatives', [])
        if alternatives:
            print("✈️  OTHER FLIGHT OPTIONS")
            print("-" * 20)
            for alt_flight in alternatives:
                flight_type = alt_flight.get('flight_type', 'flight')
                print(f"Alternative {flight_type.title()} Flight:")
                print_curated_flight_info(alt_flight, is_primary=False)
    else:
        # Fallback to legacy display
        flights = trip_data.get('flights', [])
        if flights:
            print("✈️  FLIGHTS")
            print("-" * 20)
            for flight in flights:
                print_flight_info(flight)
    
    # Hotels - show curated if available, otherwise show all hotels
    if curated_hotels.get('primary'):
        print("🏨 RECOMMENDED ACCOMMODATION")
        print("-" * 20)
        print_curated_hotel_info(curated_hotels['primary'], is_primary=True)
        
        # Show alternatives if available
        alternatives = curated_hotels.get('alternatives', [])
        if alternatives:
            print("🏨 OTHER ACCOMMODATION OPTIONS")
            print("-" * 20)
            for alt_hotel in alternatives:
                print_curated_hotel_info(alt_hotel, is_primary=False)
    else:
        # Fallback to legacy display
        hotels = trip_data.get('hotels', [])
        if hotels:
            print("🏨 ACCOMMODATIONS")
            print("-" * 20)
            for hotel in hotels:
                print_hotel_info(hotel)
    
    # Daily Plans
    daily_plans = trip_data.get('daily_plans', [])
    if daily_plans:
        print("📅 DAILY ITINERARY")
        print("-" * 20)
        for day_plan in daily_plans:
            print_daily_plan(day_plan)
    
    # Travel Tips
    travel_tips = trip_data.get('travel_tips', [])
    if travel_tips:
        print("💡 TRAVEL TIPS")
        print("-" * 20)
        for tip in travel_tips[:3]:  # Show top 3 tips
            print(f"  • {tip}")
        print()
    
    print("🎉 Happy travels!")


async def main():
    """Main CLI function."""
    args = parse_arguments()
    
    # Validate inputs
    if not validate_date(args.start_date):
        sys.exit(1)
    
    if args.duration < 1 or args.duration > 365:
        print("❌ Error: Duration must be between 1 and 365 days")
        sys.exit(1)
    
    # Print header and request details
    print_header()
    print_trip_request(args)
    
    try:
        # Call the API
        trip_data = await call_trip_api(args)
        
        if not trip_data or not isinstance(trip_data, dict):
            print("❌ Error: No valid response from API. Make sure the API server is running and returns valid JSON.")
            sys.exit(1)
        
        print("✅ Trip planning completed!")
        print()
        
        # Print the itinerary
        print_itinerary(trip_data)
        
    except KeyboardInterrupt:
        print("\n⚠️  Trip planning cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure the API server is running:")
        print("   python main.py")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())