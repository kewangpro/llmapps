"""
Curation System for Trip Planner
Selects the best flight and hotel options from raw search results
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from models import Flight, Hotel
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TripCurator:
    """
    Curator that selects the best flight and hotel options from search results.
    """
    
    def __init__(self):
        self.logger = logger
    
    def curate_flights(self, flights_data: Dict[str, Any], request) -> Tuple[List[Flight], Dict[str, Any]]:
        """
        Curate flights into primary recommendations and alternatives.
        
        Returns:
            Tuple[List[Flight], Dict[str, Any]]: (legacy_flight_list, curated_structure)
        """
        try:
            # Handle new curated structure
            if isinstance(flights_data, dict) and "primary" in flights_data:
                return self._process_curated_flights(flights_data, request)
            
            # Handle legacy structure (list of flights)
            elif isinstance(flights_data, list):
                return self._curate_legacy_flights(flights_data, request)
            
            else:
                logger.warning(f"Unknown flights data structure: {type(flights_data)}")
                return [], {"primary": {"outbound": None, "return": None}, "alternatives": []}
                
        except Exception as e:
            logger.error(f"Error curating flights: {e}")
            return [], {"primary": {"outbound": None, "return": None}, "alternatives": []}
    
    def _process_curated_flights(self, flights_data: Dict[str, Any], request) -> Tuple[List[Flight], Dict[str, Any]]:
        """Process already curated flight structure from agent."""
        flights = []
        curated_structure = {"primary": {"outbound": None, "return": None}, "alternatives": []}
        
        try:
            # Process primary flights
            if "primary" in flights_data:
                primary = flights_data["primary"]
                
                # Process outbound primary flight
                if "outbound" in primary and primary["outbound"]:
                    outbound_flight = self._create_flight_from_data(primary["outbound"], request)
                    if outbound_flight:
                        flights.append(outbound_flight)
                        curated_structure["primary"]["outbound"] = primary["outbound"]
                
                # Process return primary flight
                if "return" in primary and primary["return"]:
                    return_flight = self._create_flight_from_data(primary["return"], request)
                    if return_flight:
                        flights.append(return_flight)
                        curated_structure["primary"]["return"] = primary["return"]
            
            # Process alternative flights
            if "alternatives" in flights_data:
                for alt_flight_data in flights_data["alternatives"]:
                    alt_flight = self._create_flight_from_data(alt_flight_data, request)
                    if alt_flight:
                        flights.append(alt_flight)
                        curated_structure["alternatives"].append(alt_flight_data)
            
            logger.info(f"✅ Processed curated flights: {len(flights)} total flights")
            return flights, curated_structure
            
        except Exception as e:
            logger.error(f"Error processing curated flights: {e}")
            return flights, curated_structure
    
    def _curate_legacy_flights(self, flights_data: List[Dict], request) -> Tuple[List[Flight], Dict[str, Any]]:
        """Curate legacy flight list into primary and alternatives."""
        flights = []
        curated_structure = {"primary": {"outbound": None, "return": None}, "alternatives": []}
        
        # Separate outbound and return flights
        outbound_flights = []
        return_flights = []
        
        # Calculate return date
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        return_date = (start_date + timedelta(days=request.duration_days)).strftime('%Y-%m-%d')
        
        for flight_data in flights_data:
            flight = self._create_flight_from_data(flight_data, request)
            if flight:
                flights.append(flight)
                
                # Determine if outbound or return based on route
                from_city = flight_data.get('from_city', '').lower()
                to_city = flight_data.get('to_city', '').lower()
                origin = request.origin.lower()
                
                # Check if this is a return flight (from destination back to origin)
                is_return = False
                for dest in request.destinations:
                    if dest.lower() in from_city and origin in to_city:
                        is_return = True
                        break
                
                if is_return:
                    return_flights.append(flight_data)
                else:
                    outbound_flights.append(flight_data)
        
        # Select best outbound flight
        if outbound_flights:
            best_outbound = self._select_best_flight(outbound_flights)
            if best_outbound:
                curated_structure["primary"]["outbound"] = best_outbound
                curated_structure["primary"]["outbound"]["recommendation_reason"] = "Best value for price and timing"
        
        # Select best return flight
        if return_flights:
            best_return = self._select_best_flight(return_flights)
            if best_return:
                curated_structure["primary"]["return"] = best_return  
                curated_structure["primary"]["return"]["recommendation_reason"] = "Best value for price and timing"
        
        # Add remaining flights as alternatives
        all_flight_data = outbound_flights + return_flights
        primary_flights = [curated_structure["primary"]["outbound"], curated_structure["primary"]["return"]]
        
        for flight_data in all_flight_data:
            if flight_data not in primary_flights:
                flight_data["flight_type"] = "outbound" if flight_data in outbound_flights else "return"
                curated_structure["alternatives"].append(flight_data)
        
        logger.info(f"✅ Curated legacy flights: {len(outbound_flights)} outbound, {len(return_flights)} return")
        return flights, curated_structure
    
    def _select_best_flight(self, flight_options: List[Dict]) -> Optional[Dict]:
        """Select the best flight from options based on price and convenience."""
        if not flight_options:
            return None
        
        # Simple scoring: prioritize lower price and shorter duration
        def score_flight(flight_data):
            try:
                price_str = str(flight_data.get('price', 1000))
                # Remove $ and other currency symbols, handle various formats
                price_clean = price_str.replace('$', '').replace(',', '').strip()
                price = float(price_clean)
                duration_str = flight_data.get('duration', '5h 0m')
                
                # Parse duration (assume format like "5h 30m")
                duration_minutes = 300  # Default 5 hours
                if 'h' in duration_str:
                    parts = duration_str.split('h')
                    hours = int(parts[0].strip())
                    minutes = 0
                    if len(parts) > 1 and 'm' in parts[1]:
                        minutes = int(parts[1].replace('m', '').strip())
                    duration_minutes = hours * 60 + minutes
                
                # Lower is better for both price and duration
                # Normalize: price weight = 0.7, duration weight = 0.3
                price_score = max(0, 1000 - price) / 1000  # Higher score for lower price
                duration_score = max(0, 600 - duration_minutes) / 600  # Higher score for shorter duration
                
                return 0.7 * price_score + 0.3 * duration_score
                
            except Exception as e:
                logger.warning(f"Error scoring flight: {e}")
                return 0
        
        # Return flight with highest score
        return max(flight_options, key=score_flight)
    
    def curate_hotels(self, hotels_data: Union[Dict[str, Any], List[Dict]], request) -> Tuple[List[Hotel], Dict[str, Any]]:
        """
        Curate hotels into primary recommendation and alternatives.
        
        Returns:
            Tuple[List[Hotel], Dict[str, Any]]: (legacy_hotel_list, curated_structure)
        """
        try:
            # Handle new curated structure
            if isinstance(hotels_data, dict) and "primary" in hotels_data:
                return self._process_curated_hotels(hotels_data, request)
            
            # Handle legacy structure (list of hotels)
            elif isinstance(hotels_data, list):
                return self._curate_legacy_hotels(hotels_data, request)
            
            else:
                logger.warning(f"Unknown hotels data structure: {type(hotels_data)}")
                return [], {"primary": None, "alternatives": []}
                
        except Exception as e:
            logger.error(f"Error curating hotels: {e}")
            return [], {"primary": None, "alternatives": []}
    
    def _process_curated_hotels(self, hotels_data: Dict[str, Any], request) -> Tuple[List[Hotel], Dict[str, Any]]:
        """Process already curated hotel structure from agent."""
        hotels = []
        curated_structure = {"primary": None, "alternatives": []}
        
        try:
            # Process primary hotel
            if "primary" in hotels_data and hotels_data["primary"]:
                primary_hotel = self._create_hotel_from_data(hotels_data["primary"])
                if primary_hotel:
                    hotels.append(primary_hotel)
                    curated_structure["primary"] = hotels_data["primary"]
            
            # Process alternative hotels
            if "alternatives" in hotels_data:
                for alt_hotel_data in hotels_data["alternatives"]:
                    alt_hotel = self._create_hotel_from_data(alt_hotel_data)
                    if alt_hotel:
                        hotels.append(alt_hotel)
                        curated_structure["alternatives"].append(alt_hotel_data)
            
            logger.info(f"✅ Processed curated hotels: {len(hotels)} total hotels")
            return hotels, curated_structure
            
        except Exception as e:
            logger.error(f"Error processing curated hotels: {e}")
            return hotels, curated_structure
    
    def _curate_legacy_hotels(self, hotels_data: List[Dict], request) -> Tuple[List[Hotel], Dict[str, Any]]:
        """Curate legacy hotel list into primary and alternatives."""
        hotels = []
        curated_structure = {"primary": None, "alternatives": []}
        
        # Convert to Hotel objects
        hotel_objects = []
        for hotel_data in hotels_data:
            hotel = self._create_hotel_from_data(hotel_data)
            if hotel:
                hotels.append(hotel)
                hotel_objects.append((hotel, hotel_data))
        
        if hotel_objects:
            # Select best hotel
            best_hotel_data = self._select_best_hotel([data for _, data in hotel_objects])
            if best_hotel_data:
                curated_structure["primary"] = best_hotel_data
                curated_structure["primary"]["recommendation_reason"] = "Best combination of location, rating and value"
            
            # Add remaining hotels as alternatives
            for _, hotel_data in hotel_objects:
                if hotel_data != best_hotel_data:
                    curated_structure["alternatives"].append(hotel_data)
        
        logger.info(f"✅ Curated legacy hotels: {len(hotels)} total hotels")
        return hotels, curated_structure
    
    def _select_best_hotel(self, hotel_options: List[Dict]) -> Optional[Dict]:
        """Select the best hotel from options based on rating and value."""
        if not hotel_options:
            return None
        
        def score_hotel(hotel_data):
            try:
                rating = float(hotel_data.get('rating', 3.0))
                price_str = str(hotel_data.get('price_per_night', 200))
                # Remove $ and other currency symbols, handle various formats
                price_clean = price_str.replace('$', '').replace(',', '').strip()
                price = float(price_clean)
                
                # Higher rating is better, lower price is better
                # Normalize: rating weight = 0.6, price weight = 0.4
                rating_score = rating / 5.0  # Normalize to 0-1
                price_score = max(0, 500 - price) / 500  # Higher score for lower price
                
                return 0.6 * rating_score + 0.4 * price_score
                
            except Exception as e:
                logger.warning(f"Error scoring hotel: {e}")
                return 0
        
        # Return hotel with highest score
        return max(hotel_options, key=score_hotel)
    
    def _create_flight_from_data(self, flight_data: Dict, request) -> Optional[Flight]:
        """Create Flight object from flight data."""
        try:
            # Supply default date if missing
            flight_date = flight_data.get('date', request.start_date)
            if not flight_date or not str(flight_date).strip():
                flight_date = request.start_date
            
            # Handle price field which may already be formatted as string like "$1265"
            price_raw = flight_data.get('price', 0)
            if isinstance(price_raw, str):
                # Already formatted (e.g., "$1265"), use as-is
                estimated_price = price_raw
            else:
                # Numeric value, format with dollar sign
                estimated_price = f"${price_raw:.2f}"
            
            return Flight(
                from_city=flight_data.get('from_city', 'Unknown'),
                to_city=flight_data.get('to_city', 'Unknown'),
                date=flight_date,
                departure_time=flight_data.get('departure_time', 'TBD'),
                arrival_time=flight_data.get('arrival_time', 'TBD'),
                airline=flight_data.get('airline', 'Unknown'),
                estimated_price=estimated_price,
                data_source=flight_data.get('source', 'search')
            )
        except Exception as e:
            logger.error(f"Error creating flight from data: {e}")
            return None
    
    def _create_hotel_from_data(self, hotel_data: Dict) -> Optional[Hotel]:
        """Create Hotel object from hotel data."""
        try:
            # Handle price which might already be formatted as string
            price_raw = hotel_data.get('price_per_night', 0)
            if isinstance(price_raw, str):
                # Already formatted (e.g., "$200"), use as-is
                price_per_night = price_raw
            else:
                # Numeric value, format with dollar sign
                price_per_night = f"${price_raw:.2f}"
            
            return Hotel(
                city=hotel_data.get('city', 'Unknown'),
                name=hotel_data.get('name', 'Unknown Hotel'),
                price_per_night=price_per_night,
                rating=float(hotel_data.get('rating', 3.0)),
                amenities=hotel_data.get('amenities', []),
                data_source=hotel_data.get('source', 'search')
            )
        except Exception as e:
            logger.error(f"Error creating hotel from data: {e}")
            return None