import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/simple_models.dart';

class ApiService {
  static const String baseUrl = 'http://localhost:8000'; // Change for production
  
  static Future<TripPlan> planTrip(TripRequest request) async {
    try {
      print('🚀 Sending trip request: ${jsonEncode(request.toJson())}');
      
      final response = await http.post(
        Uri.parse('$baseUrl/plan-trip'),
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode(request.toJson()),
      );
      
      print('📡 Response status: ${response.statusCode}');
      print('📄 Response body length: ${response.body.length} chars');
      
      if (response.statusCode == 200) {
        final jsonResponse = jsonDecode(response.body);
        print('✅ JSON decoded successfully');
        print('🔍 Response keys: ${jsonResponse.keys}');
        
        // Log daily_plans structure for debugging
        if (jsonResponse['daily_plans'] != null) {
          final dailyPlans = jsonResponse['daily_plans'] as List;
          print('📅 Daily plans count: ${dailyPlans.length}');
          if (dailyPlans.isNotEmpty) {
            final firstPlan = dailyPlans[0];
            print('📅 First plan structure: ${firstPlan.keys}');
            if (firstPlan['activities'] != null) {
              print('🎯 First plan activities type: ${firstPlan['activities'].runtimeType}');
              if (firstPlan['activities'] is List && (firstPlan['activities'] as List).isNotEmpty) {
                print('🎯 First activity type: ${(firstPlan['activities'] as List)[0].runtimeType}');
                print('🎯 First activity: ${(firstPlan['activities'] as List)[0]}');
              }
            }
          }
        }
        
        try {
          final tripPlan = TripPlan.fromJson(jsonResponse);
          print('✅ TripPlan created successfully');
          return tripPlan;
        } catch (parseError) {
          print('❌ TripPlan parsing failed: $parseError');
          // Create detailed error message for UI display
          final errorDetails = '''
PARSING ERROR: $parseError

RESPONSE STRUCTURE:
- Keys: ${jsonResponse.keys.join(', ')}
- Daily plans count: ${jsonResponse['daily_plans']?.length ?? 'null'}
- First daily plan keys: ${jsonResponse['daily_plans']?[0]?.keys?.join(', ') ?? 'null'}
- First activity type: ${jsonResponse['daily_plans']?[0]?['activities']?[0]?.runtimeType ?? 'null'}

RAW RESPONSE (first 500 chars):
${jsonEncode(jsonResponse).substring(0, 500)}...
          ''';
          throw Exception(errorDetails);
        }
      } else {
        final errorBody = response.body.isNotEmpty ? response.body : 'No error details';
        throw Exception('Server error (${response.statusCode}): $errorBody');
      }
    } catch (e) {
      print('❌ API Service error: $e');
      if (e.toString().contains('Failed to parse trip plan')) {
        rethrow; // Keep detailed parsing error
      }
      throw Exception('Network/API error: $e');
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