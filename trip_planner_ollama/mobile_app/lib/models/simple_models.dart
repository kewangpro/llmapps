// Simplified models without JSON serialization for quick testing

class Flight {
  final String from;
  final String to;
  final String date;
  final String departureTime;
  final String arrivalTime;
  final String? airline;
  final String? estimatedPrice;
  final String? dataSource; // "langchain_agents", "api", "fallback", etc.
  final double? confidence; // 0.0-1.0 confidence score

  Flight({
    required this.from,
    required this.to,
    required this.date,
    required this.departureTime,
    required this.arrivalTime,
    this.airline,
    this.estimatedPrice,
    this.dataSource,
    this.confidence,
  });

  factory Flight.fromJson(Map<String, dynamic> json) {
    return Flight(
      from: json['from_city'] ?? json['from'] ?? '',
      to: json['to_city'] ?? json['to'] ?? '',
      date: json['date'] ?? '',
      departureTime: json['departure_time'] ?? '',
      arrivalTime: json['arrival_time'] ?? '',
      airline: json['airline'],
      estimatedPrice: json['estimated_price'],
      dataSource: json['data_source'],
      confidence: json['confidence']?.toDouble(),
    );
  }
}

class Hotel {
  final String name;
  final String city;
  final double rating;
  final String pricePerNight;
  final List<String> amenities;
  final String? address;
  final String? dataSource; // "langchain_agents", "api", "fallback", etc.
  final double? confidence; // 0.0-1.0 confidence score

  Hotel({
    required this.name,
    required this.city,
    required this.rating,
    required this.pricePerNight,
    required this.amenities,
    this.address,
    this.dataSource,
    this.confidence,
  });

  factory Hotel.fromJson(Map<String, dynamic> json) {
    return Hotel(
      name: json['name'] ?? '',
      city: json['city'] ?? '',
      rating: (json['rating'] ?? 0.0).toDouble(),
      pricePerNight: json['price_per_night'] ?? '',
      amenities: List<String>.from(json['amenities'] ?? []),
      address: json['address'],
      dataSource: json['data_source'],
      confidence: json['confidence']?.toDouble(),
    );
  }
}

class TripRequest {
  final String origin;
  final List<String> destinations;
  final String startDate;
  final int durationDays;
  final String budget;
  final String preferences;

  TripRequest({
    required this.origin,
    required this.destinations,
    required this.startDate,
    required this.durationDays,
    this.budget = 'medium',
    this.preferences = '',
  });

  Map<String, dynamic> toJson() {
    return {
      'origin': origin,
      'destinations': destinations,
      'start_date': startDate,
      'duration_days': durationDays,
      'budget': budget,
      'preferences': preferences,
    };
  }
}

class DayPlan {
  final int day;
  final String date;
  final String city;
  final List<String> activities;
  final String? accommodation;
  final String? transportation;
  final List<String> cityTips;

  DayPlan({
    required this.day,
    required this.date,
    required this.city,
    required this.activities,
    this.accommodation,
    this.transportation,
    this.cityTips = const [],
  });

  factory DayPlan.fromJson(Map<String, dynamic> json) {
    return DayPlan(
      day: json['day'] ?? 0,
      date: json['date'] ?? '',
      city: json['city'] ?? '',
      activities: List<String>.from(json['activities'] ?? []),
      accommodation: json['accommodation'],
      transportation: json['transportation'],
      cityTips: List<String>.from(json['city_tips'] ?? []),
    );
  }
}

class ItineraryItem {
  final String type; // 'flight', 'day', or 'hotel'
  final String date;
  final String? title;
  final Flight? flight;
  final DayPlan? dayPlan;
  final Hotel? hotel;

  ItineraryItem({
    required this.type,
    required this.date,
    this.title,
    this.flight,
    this.dayPlan,
    this.hotel,
  });
}

class TripPlan {
  final int totalDays;
  final List<String> routeOrder;
  final List<DayPlan> dailyPlans;
  final String? estimatedBudget;
  final List<String> travelTips;
  final List<Flight> flights;
  final List<Hotel> hotels;

  TripPlan({
    required this.totalDays,
    required this.routeOrder,
    required this.dailyPlans,
    this.estimatedBudget,
    this.travelTips = const [],
    this.flights = const [],
    this.hotels = const [],
  });

  // Generate merged itinerary combining flights, hotels, and daily plans
  List<ItineraryItem> get itinerary {
    List<ItineraryItem> items = [];
    
    // Sort flights by date to maintain chronological order
    List<Flight> sortedFlights = List.from(flights);
    sortedFlights.sort((a, b) => a.date.compareTo(b.date));
    
    // Sort daily plans by day number to maintain logical order
    List<DayPlan> sortedDays = List.from(dailyPlans);
    sortedDays.sort((a, b) => a.day.compareTo(b.day));
    
    Set<String> processedDates = {};
    Set<String> processedHotels = {};
    
    // Identify return flight (last flight that goes back to origin)
    Flight? returnFlight;
    if (sortedFlights.length > 1) {
      Flight lastFlight = sortedFlights.last;
      Flight firstFlight = sortedFlights.first;
      if (lastFlight.to == firstFlight.from) {
        returnFlight = lastFlight;
      }
    }
    
    // Process all flights except the return flight
    List<Flight> arrivalFlights = returnFlight != null 
        ? sortedFlights.sublist(0, sortedFlights.length - 1)
        : sortedFlights;
    
    for (int i = 0; i < arrivalFlights.length; i++) {
      var flight = arrivalFlights[i];
      
      // Add flight card
      items.add(ItineraryItem(
        type: 'flight',
        date: flight.date,
        title: '${flight.from} → ${flight.to}',
        flight: flight,
      ));
      
      // Add the best hotel for the destination city
      var cityHotels = hotels.where((hotel) => 
        hotel.city == flight.to && !processedHotels.contains(hotel.name)
      ).toList();
      
      if (cityHotels.isNotEmpty) {
        // Sort by rating (descending), then by price (ascending for same rating)
        cityHotels.sort((a, b) {
          if (a.rating != b.rating) {
            return b.rating.compareTo(a.rating); // Higher rating first
          }
          // For same rating, prefer lower price
          String priceA = a.pricePerNight.replaceAll(RegExp(r'[^\d]'), '');
          String priceB = b.pricePerNight.replaceAll(RegExp(r'[^\d]'), '');
          int numA = int.tryParse(priceA) ?? 0;
          int numB = int.tryParse(priceB) ?? 0;
          return numA.compareTo(numB); // Lower price first
        });
        
        var cityHotel = cityHotels.first;
        items.add(ItineraryItem(
          type: 'hotel',
          date: flight.date,
          title: 'Stay in ${cityHotel.city}',
          hotel: cityHotel,
        ));
        
        // Mark ALL hotels in this city as processed to avoid duplicates
        for (var hotel in cityHotels) {
          processedHotels.add(hotel.name);
        }
      }
      
      // Find activities for this destination city
      // For the last arrival flight, include all remaining activities in that city
      // For other flights, only include activities until the next departure
      String? nextDepartureDate;
      if (i + 1 < arrivalFlights.length) {
        nextDepartureDate = arrivalFlights[i + 1].date;
      } else if (returnFlight != null) {
        // For the last city, include activities until the return flight
        nextDepartureDate = returnFlight.date;
      }
      
      List<DayPlan> cityDays = sortedDays
          .where((dayPlan) => 
            dayPlan.city == flight.to && 
            !processedDates.contains(dayPlan.date) &&
            (nextDepartureDate == null || dayPlan.date.compareTo(nextDepartureDate) < 0)
          )
          .toList();
      
      // Sort city days by date to maintain chronological order
      cityDays.sort((a, b) => a.date.compareTo(b.date));
      
      // Add activities for this city
      for (var dayPlan in cityDays) {
        items.add(ItineraryItem(
          type: 'day',
          date: dayPlan.date,
          dayPlan: dayPlan,
        ));
        processedDates.add(dayPlan.date);
      }
    }
    
    // Add any remaining daily plans that weren't processed
    for (var day in sortedDays) {
      if (!processedDates.contains(day.date)) {
        items.add(ItineraryItem(
          type: 'day',
          date: day.date,
          dayPlan: day,
        ));
      }
    }
    
    // Add the return flight at the very end (after ALL activities)
    if (returnFlight != null) {
      items.add(ItineraryItem(
        type: 'flight',
        date: returnFlight.date,
        title: '${returnFlight.from} → ${returnFlight.to}',
        flight: returnFlight,
      ));
    }
    
    return items;
  }

  factory TripPlan.fromJson(Map<String, dynamic> json) {
    var dailyPlansJson = json['daily_plans'] as List? ?? [];
    List<DayPlan> dailyPlans = dailyPlansJson
        .map((dayJson) => DayPlan.fromJson(dayJson))
        .toList();

    var flightsJson = json['flights'] as List? ?? [];
    List<Flight> flights = flightsJson
        .map((flightJson) => Flight.fromJson(flightJson))
        .toList();

    var hotelsJson = json['hotels'] as List? ?? [];
    List<Hotel> hotels = hotelsJson
        .map((hotelJson) => Hotel.fromJson(hotelJson))
        .toList();

    return TripPlan(
      totalDays: json['total_days'] ?? 0,
      routeOrder: List<String>.from(json['route_order'] ?? []),
      dailyPlans: dailyPlans,
      estimatedBudget: json['estimated_budget'],
      travelTips: List<String>.from(json['travel_tips'] ?? []),
      flights: flights,
      hotels: hotels,
    );
  }
}