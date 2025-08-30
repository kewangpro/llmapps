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
                      ? const Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(
                                color: Colors.white,
                                strokeWidth: 2,
                              ),
                            ),
                            SizedBox(width: 12),
                            Text('Planning Trip...'),
                          ],
                        )
                      : const Text(
                          'Plan My Trip',
                          style: TextStyle(fontSize: 18),
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
      
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error: ${e.toString()}'),
          backgroundColor: Colors.red,
        ),
      );
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
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
