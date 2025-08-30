#!/bin/bash

# Trip Planner Flutter Web Runner
# Cleans, installs dependencies, and launches the Flutter web application in Chrome

echo "🌍 Starting Trip Planner Web App"

# Clean Flutter project
echo "🧹 Cleaning Flutter project..."
flutter clean

# Get dependencies
echo "📦 Getting dependencies..."
flutter pub get

# Run Flutter web app in Chrome
echo "🚀 Running on Chrome..."
flutter run -d chrome
