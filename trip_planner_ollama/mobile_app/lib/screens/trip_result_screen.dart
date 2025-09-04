import 'package:flutter/material.dart';
import '../models/simple_models.dart';

class TripResultScreen extends StatelessWidget {
  final TripPlan tripPlan;

  const TripResultScreen({Key? key, required this.tripPlan}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Your Trip Plan'),
        backgroundColor: Colors.green[600],
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Trip Overview Card
            Card(
              elevation: 4,
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const Icon(Icons.route, color: Colors.blue, size: 24),
                        const SizedBox(width: 8),
                        const Text(
                          'Trip Overview',
                          style: TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    _buildInfoRow('Duration', '${tripPlan.totalDays} days'),
                    _buildInfoRow('Route', tripPlan.routeOrder.join(' → ')),
                    if (tripPlan.estimatedBudget != null)
                      _buildInfoRow('Budget', tripPlan.estimatedBudget!),
                  ],
                ),
              ),
            ),
            
            const SizedBox(height: 20),
            
            // Unified Itinerary
            const Text(
              'Complete Itinerary',
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            
            ...tripPlan.itinerary.map((item) => _buildItineraryItem(item)),
            
            
            const SizedBox(height: 20),
            
            // Plan Another Trip Button
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () => Navigator.pop(context),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue[600],
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
                child: const Text(
                  'Plan Another Trip',
                  style: TextStyle(fontSize: 16),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildItineraryItem(ItineraryItem item) {
    if (item.type == 'flight' && item.flight != null) {
      return _buildFlightCard(item.flight!);
    } else if (item.type == 'hotel' && item.hotel != null) {
      return _buildHotelCard(item.hotel!);
    } else if (item.type == 'day' && item.dayPlan != null) {
      return _buildDayPlanCard(item.dayPlan!);
    }
    return const SizedBox.shrink();
  }

  Widget _buildFlightCard(Flight flight) {
    return Card(
      elevation: 3,
      margin: const EdgeInsets.only(bottom: 12.0),
      color: Colors.blue[50], // Light blue background for flights
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Flight header
            Row(
              children: [
                const Icon(Icons.flight_takeoff, color: Colors.blue, size: 24),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    '${flight.from} → ${flight.to}',
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                // Price badge first
                if (flight.estimatedPrice != null)
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.green[100],
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      flight.estimatedPrice!,
                      style: TextStyle(
                        color: Colors.green[800],
                        fontWeight: FontWeight.bold,
                        fontSize: 12,
                      ),
                    ),
                  ),
                // Data source indicator at rightmost position
                if (flight.estimatedPrice != null && flight.dataSource != null)
                  const SizedBox(width: 8),
                if (flight.dataSource != null)
                  _buildDataSourceBadge(flight.dataSource!),
              ],
            ),
            
            const SizedBox(height: 12),
            
            // Flight details
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          const Icon(Icons.schedule, size: 16, color: Colors.grey),
                          const SizedBox(width: 6),
                          Text(
                            'Departure: ${flight.departureTime}',
                            style: const TextStyle(fontSize: 14),
                          ),
                        ],
                      ),
                      const SizedBox(height: 4),
                      Row(
                        children: [
                          const Icon(Icons.flight_land, size: 16, color: Colors.grey),
                          const SizedBox(width: 6),
                          Text(
                            'Arrival: ${flight.arrivalTime}',
                            style: const TextStyle(fontSize: 14),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
                if (flight.airline != null)
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      Row(
                        children: [
                          const Icon(Icons.airplanemode_active, size: 16, color: Colors.grey),
                          const SizedBox(width: 6),
                          Text(
                            flight.airline!,
                            style: const TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 4),
                      Text(
                        flight.date,
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.grey[600],
                        ),
                      ),
                    ],
                  ),
              ],
            ),
            
            // Confidence indicator only (data source shown in header)
            if (flight.confidence != null) ...[
              const SizedBox(height: 8),
              Row(
                children: [
                  const Icon(Icons.verified, size: 16, color: Colors.grey),
                  const SizedBox(width: 6),
                  Text(
                    'Confidence: ${(flight.confidence! * 100).toStringAsFixed(0)}%',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.grey[600],
                    ),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8.0),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 80,
            child: Text(
              '$label:',
              style: const TextStyle(
                fontWeight: FontWeight.w600,
                color: Colors.grey,
              ),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(fontSize: 14),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDayPlanCard(DayPlan dayPlan) {
    return Card(
      elevation: 2,
      margin: const EdgeInsets.only(bottom: 12.0),
      color: Colors.green[50], // Light green background for daily plans
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Day header
            Row(
              children: [
                CircleAvatar(
                  backgroundColor: Colors.blue[600],
                  radius: 16,
                  child: Text(
                    '${dayPlan.day}',
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        dayPlan.city,
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Text(
                        dayPlan.date,
                        style: TextStyle(
                          fontSize: 14,
                          color: Colors.grey[600],
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            
            const SizedBox(height: 12),
            
            // Activities
            if (dayPlan.activities.isNotEmpty) ...[
              const Row(
                children: [
                  Icon(Icons.local_activity, size: 16, color: Colors.green),
                  SizedBox(width: 6),
                  Text(
                    'Activities:',
                    style: TextStyle(fontWeight: FontWeight.w600),
                  ),
                ],
              ),
              const SizedBox(height: 6),
              ...dayPlan.activities.map((activity) => Padding(
                padding: const EdgeInsets.only(left: 22, bottom: 4),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      child: Text('• ${activity.description.isNotEmpty ? activity.description : activity.name}'),
                    ),
                    if (activity.source != null) ...[
                      const SizedBox(width: 8),
                      _buildDataSourceBadge(activity.source!),
                    ],
                  ],
                ),
              )),
            ],
            
            // Transportation
            if (dayPlan.transportation != null) ...[
              const SizedBox(height: 8),
              Row(
                children: [
                  const Icon(Icons.directions, size: 16, color: Colors.orange),
                  const SizedBox(width: 6),
                  const Text(
                    'Transportation: ',
                    style: TextStyle(fontWeight: FontWeight.w600),
                  ),
                  Expanded(child: Text(dayPlan.transportation!)),
                ],
              ),
            ],
            
            // Accommodation
            if (dayPlan.accommodation != null) ...[
              const SizedBox(height: 8),
              Row(
                children: [
                  const Icon(Icons.hotel, size: 16, color: Colors.purple),
                  const SizedBox(width: 6),
                  const Text(
                    'Stay: ',
                    style: TextStyle(fontWeight: FontWeight.w600),
                  ),
                  Expanded(child: Text(dayPlan.accommodation!)),
                ],
              ),
            ],
            
            // City Tips
            if (dayPlan.cityTips.isNotEmpty) ...[
              const SizedBox(height: 8),
              const Row(
                children: [
                  Icon(Icons.lightbulb_outline, size: 16, color: Colors.amber),
                  SizedBox(width: 6),
                  Text(
                    'Local Tips:',
                    style: TextStyle(fontWeight: FontWeight.w600),
                  ),
                ],
              ),
              const SizedBox(height: 6),
              ...dayPlan.cityTips.map((tip) => Padding(
                padding: const EdgeInsets.only(left: 22, bottom: 4),
                child: Text(
                  '💡 $tip',
                  style: const TextStyle(fontSize: 14, fontStyle: FontStyle.italic),
                ),
              )),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildDataSourceBadge(String dataSource) {
    Color badgeColor;
    Color textColor;
    IconData icon;
    String displayText;

    switch (dataSource.toLowerCase()) {
      case 'llm':
        // Simple Mode - LLM reasoning
        badgeColor = Colors.purple[100]!;
        textColor = Colors.purple[800]!;
        icon = Icons.psychology;
        displayText = 'LLM';
        break;
      case 'google':
        // Comprehensive Mode - Google Search API
        badgeColor = Colors.green[100]!;
        textColor = Colors.green[800]!;
        icon = Icons.search;
        displayText = 'Google';
        break;
      case 'api':
        badgeColor = Colors.blue[100]!;
        textColor = Colors.blue[800]!;
        icon = Icons.cloud;
        displayText = 'API';
        break;
      case 'unknown':
      default:
        // Handle unknown or legacy data sources
        badgeColor = Colors.grey[100]!;
        textColor = Colors.grey[800]!;
        icon = Icons.help_outline;
        displayText = 'Unknown';
        break;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: badgeColor,
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: textColor.withOpacity(0.3)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            icon,
            size: 12,
            color: textColor,
          ),
          const SizedBox(width: 4),
          Text(
            displayText,
            style: TextStyle(
              color: textColor,
              fontWeight: FontWeight.bold,
              fontSize: 10,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHotelCard(Hotel hotel) {
    return Card(
      elevation: 3,
      margin: const EdgeInsets.only(bottom: 12.0),
      color: Colors.purple[50], // Light purple background for hotels
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Hotel header
            Row(
              children: [
                const Icon(Icons.hotel, color: Colors.purple, size: 24),
                const SizedBox(width: 8),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        hotel.name,
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Text(
                        hotel.city,
                        style: TextStyle(
                          fontSize: 14,
                          color: Colors.grey[600],
                        ),
                      ),
                    ],
                  ),
                ),
                // Price badge first
                if (hotel.pricePerNight.isNotEmpty)
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.green[100],
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      '${hotel.pricePerNight}/night',
                      style: TextStyle(
                        color: Colors.green[800],
                        fontWeight: FontWeight.bold,
                        fontSize: 12,
                      ),
                    ),
                  ),
                // Data source indicator at rightmost position
                if (hotel.pricePerNight.isNotEmpty && hotel.dataSource != null)
                  const SizedBox(width: 8),
                if (hotel.dataSource != null)
                  _buildDataSourceBadge(hotel.dataSource!),
              ],
            ),
            
            const SizedBox(height: 12),
            
            // Hotel details
            Row(
              children: [
                // Rating
                if (hotel.rating > 0) ...[
                  const Icon(Icons.star, size: 16, color: Colors.amber),
                  const SizedBox(width: 4),
                  Text(
                    hotel.rating.toStringAsFixed(1),
                    style: const TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  const SizedBox(width: 16),
                ],
                
                // Address
                if (hotel.address != null)
                  Expanded(
                    child: Row(
                      children: [
                        const Icon(Icons.location_on, size: 16, color: Colors.grey),
                        const SizedBox(width: 4),
                        Expanded(
                          child: Text(
                            hotel.address!,
                            style: const TextStyle(fontSize: 12),
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                      ],
                    ),
                  ),
              ],
            ),
            
            // Amenities
            if (hotel.amenities.isNotEmpty) ...[
              const SizedBox(height: 8),
              Wrap(
                spacing: 6,
                runSpacing: 4,
                children: hotel.amenities.take(4).map((amenity) => Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(
                    color: Colors.purple[100],
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    amenity,
                    style: TextStyle(
                      fontSize: 10,
                      color: Colors.purple[800],
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                )).toList(),
              ),
            ],
            
            // Confidence indicator only (data source shown in header)
            if (hotel.confidence != null) ...[
              const SizedBox(height: 8),
              Row(
                children: [
                  const Icon(Icons.verified, size: 16, color: Colors.grey),
                  const SizedBox(width: 6),
                  Text(
                    'Confidence: ${(hotel.confidence! * 100).toStringAsFixed(0)}%',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.grey[600],
                    ),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }
}