import 'package:flutter/material.dart';
import 'screens/trip_input_screen.dart';

void main() {
  runApp(const TripPlannerApp());
}

class TripPlannerApp extends StatelessWidget {
  const TripPlannerApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Trip Planner',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        appBarTheme: const AppBarTheme(
          elevation: 0,
        ),
      ),
      home: const TripInputScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}