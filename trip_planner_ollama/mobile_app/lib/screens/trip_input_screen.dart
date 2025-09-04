import 'package:flutter/material.dart';
import '../models/simple_models.dart';
import '../services/api_service.dart';
import 'trip_result_screen.dart';

class TripInputScreen extends StatefulWidget {
  const TripInputScreen({Key? key}) : super(key: key);

  @override
  State<TripInputScreen> createState() => _TripInputScreenState();
}

class _TripInputScreenState extends State<TripInputScreen> {
  final _formKey = GlobalKey<FormState>();
  final _originController = TextEditingController();
  final _destinationsController = TextEditingController();
  final _startDateController = TextEditingController();
  final _durationController = TextEditingController();
  final _preferencesController = TextEditingController();
  
  String _selectedBudget = 'medium';
  bool _isLoading = false;
  bool _isComprehensiveMode = false;
  
  final List<String> _budgetOptions = ['low', 'medium', 'high'];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Plan Your Trip'),
        backgroundColor: Colors.blue[600],
        foregroundColor: Colors.white,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const Text(
                  'Multi-City Round Trip Planner',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 20),
                
                _buildTextField(
                  controller: _originController,
                  label: 'Origin City',
                  hint: 'e.g. San Francisco',
                  icon: Icons.home,
                ),
                const SizedBox(height: 16),
                
                _buildTextField(
                  controller: _destinationsController,
                  label: 'Destination Cities',
                  hint: 'e.g. Tokyo, Seoul, Bangkok (comma separated)',
                  icon: Icons.location_on,
                  maxLines: 2,
                ),
                const SizedBox(height: 16),
                
                _buildTextField(
                  controller: _startDateController,
                  label: 'Start Date',
                  hint: 'e.g. 2024-04-01',
                  icon: Icons.calendar_today,
                ),
                const SizedBox(height: 16),
                
                _buildTextField(
                  controller: _durationController,
                  label: 'Duration (days)',
                  hint: 'e.g. 10',
                  icon: Icons.schedule,
                  keyboardType: TextInputType.number,
                ),
                const SizedBox(height: 16),
                
                // Budget dropdown
                DropdownButtonFormField<String>(
                  value: _selectedBudget,
                  decoration: InputDecoration(
                    labelText: 'Budget Level',
                    prefixIcon: const Icon(Icons.attach_money),
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                  items: _budgetOptions.map((budget) {
                    return DropdownMenuItem<String>(
                      value: budget,
                      child: Text(budget.toUpperCase()),
                    );
                  }).toList(),
                  onChanged: (value) {
                    setState(() {
                      _selectedBudget = value!;
                    });
                  },
                ),
                const SizedBox(height: 16),
                
                _buildTextField(
                  controller: _preferencesController,
                  label: 'Preferences (optional)',
                  hint: 'e.g. love food, temples, nightlife',
                  icon: Icons.favorite,
                  maxLines: 3,
                  isRequired: false,
                ),
                const SizedBox(height: 20),
                
                // Comprehensive Mode Toggle
                Container(
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.grey[300]!),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Icon(
                            _isComprehensiveMode ? Icons.psychology : Icons.speed,
                            color: _isComprehensiveMode ? Colors.purple : Colors.blue,
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  _isComprehensiveMode ? 'Comprehensive Mode' : 'Simple Mode',
                                  style: const TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                                Text(
                                  _isComprehensiveMode 
                                    ? '5 specialized agents with Google Search (3+ minutes)'
                                    : 'Single master agent, fast results (30-60 seconds)',
                                  style: TextStyle(
                                    fontSize: 12,
                                    color: Colors.grey[600],
                                  ),
                                ),
                              ],
                            ),
                          ),
                          Switch(
                            value: _isComprehensiveMode,
                            onChanged: (value) {
                              setState(() {
                                _isComprehensiveMode = value;
                              });
                            },
                            activeTrackColor: Colors.purple[300],
                            activeThumbColor: Colors.purple,
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: _isComprehensiveMode ? Colors.purple[50] : Colors.blue[50],
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: Row(
                          children: [
                            Icon(
                              Icons.info_outline,
                              size: 16,
                              color: _isComprehensiveMode ? Colors.purple[700] : Colors.blue[700],
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                _isComprehensiveMode
                                  ? 'Detailed analysis with budget planning, specialized flight/hotel agents, and activity recommendations'
                                  : 'Quick trip planning with efficient single-agent processing',
                                style: TextStyle(
                                  fontSize: 12,
                                  color: _isComprehensiveMode ? Colors.purple[700] : Colors.blue[700],
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 32),
                
                ElevatedButton(
                  onPressed: _isLoading ? null : _planTrip,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue[600],
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                  child: _isLoading
                      ? Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(
                                color: Colors.white,
                                strokeWidth: 2,
                              ),
                            ),
                            const SizedBox(width: 12),
                            Text(_isComprehensiveMode 
                                ? 'Planning with 5 agents...' 
                                : 'Planning Trip...'),
                          ],
                        )
                      : Text(
                          _isComprehensiveMode 
                              ? 'Plan with Comprehensive Mode' 
                              : 'Plan My Trip',
                          style: const TextStyle(fontSize: 18),
                        ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildTextField({
    required TextEditingController controller,
    required String label,
    required String hint,
    required IconData icon,
    TextInputType keyboardType = TextInputType.text,
    int maxLines = 1,
    bool isRequired = true,
  }) {
    return TextFormField(
      controller: controller,
      keyboardType: keyboardType,
      maxLines: maxLines,
      decoration: InputDecoration(
        labelText: label,
        hintText: hint,
        prefixIcon: Icon(icon),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
        ),
      ),
      validator: isRequired ? (value) {
        if (value == null || value.isEmpty) {
          return 'Please enter $label';
        }
        return null;
      } : null,
    );
  }

  Future<void> _planTrip() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    setState(() {
      _isLoading = true;
    });

    try {
      // Parse destinations
      final destinations = _destinationsController.text
          .split(',')
          .map((city) => city.trim())
          .where((city) => city.isNotEmpty)
          .toList();

      if (destinations.isEmpty) {
        throw Exception('Please enter at least one destination city');
      }

      final request = TripRequest(
        origin: _originController.text.trim(),
        destinations: destinations,
        startDate: _startDateController.text.trim(),
        durationDays: int.parse(_durationController.text.trim()),
        budget: _selectedBudget,
        preferences: _preferencesController.text.trim(),
        collaborationMode: _isComprehensiveMode ? 'comprehensive' : 'simple',
      );

      final tripPlan = await ApiService.planTrip(request);

      if (!mounted) return;

      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => TripResultScreen(tripPlan: tripPlan),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      
      // Show both SnackBar (brief) and Dialog (persistent)
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error: ${e.toString()}'),
          backgroundColor: Colors.red,
          duration: const Duration(seconds: 8), // Longer duration
          action: SnackBarAction(
            label: 'Details',
            textColor: Colors.white,
            onPressed: () => _showErrorDialog(e.toString()),
          ),
        ),
      );
      
      // Also show persistent error dialog
      _showErrorDialog(e.toString());
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  void _showErrorDialog(String error) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Row(
            children: [
              Icon(Icons.error_outline, color: Colors.red),
              SizedBox(width: 8),
              Text('Trip Planning Error'),
            ],
          ),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Unable to create your trip plan. Please check the details below:',
                style: TextStyle(fontWeight: FontWeight.w500),
              ),
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.red[50],
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.red[200]!),
                ),
                child: SingleChildScrollView(
                  child: Text(
                    error,
                    style: TextStyle(
                      fontSize: 12,
                      fontFamily: 'monospace',
                      color: Colors.red[800],
                    ),
                  ),
                ),
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Close'),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.of(context).pop();
                // Optional: Retry the request
                // _planTrip();
              },
              child: const Text('Try Again'),
            ),
          ],
        );
      },
    );
  }


  @override
  void dispose() {
    _originController.dispose();
    _destinationsController.dispose();
    _startDateController.dispose();
    _durationController.dispose();
    _preferencesController.dispose();
    super.dispose();
  }
}