#!/bin/bash

# Flutter iOS Development Runner with Debug Console
# This script runs the Flutter app on iOS simulator with console output

set -e

echo "🚀 Flutter iOS Development Runner"
echo "=================================="

# Check if Flutter is installed
if ! command -v flutter &> /dev/null; then
    echo "❌ Flutter not found. Please install Flutter first."
    echo "   Visit: https://flutter.dev/docs/get-started/install"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "pubspec.yaml" ]; then
    echo "❌ Not in a Flutter project directory!"
    echo "   Make sure to run this script from the mobile_app directory."
    exit 1
fi

# Get dependencies
echo "📦 Getting Flutter dependencies..."
flutter pub get

# Check for iOS setup
if ! command -v xcodebuild &> /dev/null; then
    echo "❌ Xcode not found. iOS development requires Xcode."
    exit 1
fi

# Open iOS Simulator if not already running
echo "📱 Starting iOS Simulator..."
open -a Simulator

# Wait a moment for simulator to boot
sleep 3

echo ""
echo "🔧 Development Options:"
echo "1. Run app with console output in terminal"
echo "2. Run app and open Xcode console"
echo "3. Run app with verbose output"
echo ""

read -p "Choose option (1-3, or press Enter for option 1): " choice

case $choice in
    2)
        echo "🖥️  Opening Xcode Device Console..."
        echo "   Instructions:"
        echo "   1. In Xcode: Window → Devices and Simulators"
        echo "   2. Select your simulator device"
        echo "   3. Click 'Open Console' button"
        echo ""
        flutter run --debug
        ;;
    3)
        echo "🔍 Running with verbose output..."
        flutter run --debug --verbose
        ;;
    *)
        echo "📱 Running with console output in terminal..."
        echo ""
        echo "🔍 Debug Console Output:"
        echo "========================"
        flutter run --debug | tee ios_debug.log
        ;;
esac

echo ""
echo "✅ Flutter iOS session ended"
echo "📄 Debug output saved to: ios_debug.log (if using option 1)"