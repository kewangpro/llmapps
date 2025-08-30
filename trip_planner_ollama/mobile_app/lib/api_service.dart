import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/simple_models.dart';

class ApiService {
  static const String baseUrl = 'http://localhost:8000'; // Change for production
  
  static Future<TripPlan> planTrip(TripRequest request) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/plan-trip'),
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode(request.toJson()),
      );
      
      if (response.statusCode == 200) {
        final jsonResponse = jsonDecode(response.body);
        return TripPlan.fromJson(jsonResponse);
      } else {
        throw Exception('Failed to plan trip: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error planning trip: $e');
    }
  }
  
  static Future<Map<String, dynamic>> checkHealth() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/'));
      
      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Health check failed: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error checking health: $e');
    }
  }
  
  static Future<Map<String, dynamic>> testOllama() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/test-ollama'));
      
      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Ollama test failed: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error testing Ollama: $e');
    }
  }
}