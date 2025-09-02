#!/bin/bash

# Trip Planner Flutter iOS Runner
# Cleans, installs dependencies, and launches the Flutter iOS application in iOS Simulator

echo "📱 Starting Trip Planner iOS App"

# Clean Flutter project
echo "🧹 Cleaning Flutter project..."
flutter clean

# Get dependencies
echo "📦 Getting dependencies..."
flutter pub get

# Run Flutter iOS app in iOS Simulator
echo "🚀 Running on iOS Simulator..."
flutter run -d ios