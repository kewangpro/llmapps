# AI Trip Planner Mobile App

A Flutter mobile application for smart multi-city round trip planning using pure LLM reasoning with Ollama integration (Gemma3 model).

## 🎯 Features

- **Smart Trip Planning**: Pure LLM reasoning for multi-city round trip optimization
- **Unified Itinerary**: Chronological view combining flights and daily activities
- **Flight Integration**: Realistic airline suggestions with times and pricing
- **Beautiful UI**: Clean, modern interface with visual distinction between flights and activities
- **Cross-Platform**: Runs on iOS (iPhone/iPad Simulator), Android, and Web
- **Local AI**: Uses Ollama Gemma3 for privacy-focused trip generation
- **Hotel Selection**: AI selects best hotel per city based on rating and price
- **Clean UI**: Modern interface with AI agent badges and structured data display

## 🏗️ Architecture

- **Frontend**: Flutter (Dart)
- **Backend API**: FastAPI Python server (../main.py)
- **AI Engine**: Dual-mode LangChain ReAct system + Ollama Gemma3
- **Models**: Clean data models with JSON serialization
- **CLI Tool**: Command-line interface (../run.py) for direct API access

## 📱 UI Components

### Trip Input Screen
- Origin and destination cities input
- Date and duration selection
- Budget level (Low/Medium/High)
- Optional preferences (food, activities, etc.)

### Trip Results Screen
- **Complete Itinerary**: Unified chronological timeline with smart hotel selection
- **Flight Cards** (Blue): Airline, times, prices, route information
- **Daily Plan Cards** (Green): Activities, accommodations, local transportation
- **Hotel Cards**: Best hotel per city selected by rating and price optimization
- **AI Agent Badges**: Clean indicators showing "AI Agent" data sources
- **Travel Tips**: AI-generated recommendations

## 🚀 Getting Started

### Prerequisites
- Flutter SDK (3.10.0+)
- Dart SDK (3.0.0+)
- Backend API server running on localhost:8000
- **iOS Development**: Xcode, iOS Simulator (iOS 18.6+), CocoaPods
- **Android Development**: Android Studio, Android SDK (optional)
- **Web Development**: Chrome browser (recommended)

### Installation

1. **Install dependencies**:
   ```bash
   flutter pub get
   ```

2. **Ensure backend is running**:
   ```bash
   cd ..
   source .venv/bin/activate
   python main.py
   ```

3. **Run the app**:
   
   **Quick Start (Recommended)**:
   ```bash
   ./run.sh
   ```
   This script automatically cleans the project, gets dependencies, and runs on Chrome.
   
   **Manual Launch**:
   - **Web**: `flutter run -d chrome`
   - **iOS Simulator**: `flutter run -d ios` (requires iOS setup - see iOS Setup section below)
   - **Android**: `flutter run -d android` (requires Android Studio)

### iOS Setup (Required for iOS Simulator)

**First-time iOS Setup**:
1. **Install Xcode** from the Mac App Store
2. **Accept Xcode License**:
   ```bash
   sudo xcodebuild -license accept
   ```
3. **Install CocoaPods**:
   ```bash
   brew install cocoapods
   ```
4. **Generate iOS Platform Files** (if not present):
   ```bash
   flutter create --platforms=ios .
   ```
5. **Download iOS Simulator Runtime**:
   ```bash
   xcodebuild -downloadPlatform iOS
   ```

**Running on iOS**:
1. **Check available devices**:
   ```bash
   flutter devices
   ```
2. **Launch on specific iOS simulator**:
   ```bash
   flutter run -d ios
   ```
   Or target specific device:
   ```bash
   flutter run -d [DEVICE_ID]
   ```

**iOS Troubleshooting**:
- If no iOS devices appear: `open -a Simulator` then run `flutter devices`
- For "connection refused" errors: Ensure backend server is running on `http://localhost:8000`
- iOS App Transport Security is configured to allow localhost connections

### Automated Setup Script
The `run.sh` script provides a complete development workflow:
```bash
#!/bin/bash
echo "🌍 Starting Trip Planner Web App"
echo "🧹 Cleaning Flutter project..."
flutter clean
echo "📦 Getting dependencies..."
flutter pub get
echo "🚀 Running on Chrome..."
flutter run -d chrome
```

## 📂 Project Structure

```
lib/
├── main.dart                    # App entry point
├── models/
│   └── simple_models.dart      # Data models (Trip, Flight, DayPlan, ItineraryItem)
├── screens/
│   ├── trip_input_screen.dart  # Trip planning form
│   └── trip_result_screen.dart # Unified itinerary display
└── services/
    └── api_service.dart        # Backend API communication
```

## 🎨 Design Features

### Visual Distinction
- **Flight Cards**: Light blue background with airplane icons
- **Daily Plan Cards**: Light green background with activity icons
- **Unified Timeline**: Chronologically sorted itinerary items

### Smart Data Flow
```
User Input → API Request → LLM Agent ReAct → Structured Output → Data Extraction → Flutter UI
```

### Itinerary Model
```dart
class ItineraryItem {
  final String type;        // 'flight' or 'day'
  final String date;        // YYYY-MM-DD
  final Flight? flight;     // Flight details if type == 'flight'  
  final DayPlan? dayPlan;   // Daily activities if type == 'day'
}
```

## 🛠️ Development

### Build for Production
```bash
# Web deployment
flutter build web

# Android APK
flutter build apk

# iOS (requires Apple Developer account for App Store)
flutter build ios --release

# iOS IPA (for distribution)
flutter build ipa
```

### Testing
```bash
flutter test
```

### Code Generation (if needed)
```bash
flutter packages pub run build_runner build
```

## 🌐 API Integration

### Backend Endpoints
- `GET /` - Health check
- `GET /test-ollama` - Test Ollama connection  
- `POST /plan-trip` - Generate trip plan

### Request Format
```json
{
  "origin": "San Francisco",
  "destinations": ["Tokyo", "Seoul"],
  "start_date": "2024-04-01",
  "duration_days": 7,
  "budget": "medium",
  "preferences": "love food and temples"
}
```

### Response Format
```json
{
  "total_days": 7,
  "route_order": ["San Francisco", "Tokyo", "Seoul", "San Francisco"],
  "flights": [...],
  "daily_plans": [...],
  "estimated_budget": "Medium budget estimate",
  "travel_tips": [...]
}
```

## 🎯 Key Features

### Round Trip Logic
- Automatic return to origin city
- Proper date calculations for each day
- Realistic flight scheduling

### Flight Integration
- Multiple airline options (United, Delta, JAL, etc.)
- Realistic pricing estimates
- Departure/arrival time coordination

### AI-Powered Planning
- Dual-mode LangChain ReAct framework with autonomous reasoning
- Simple mode: Single master agent (25s execution, production-optimized)
- Comprehensive mode: 5 specialized agents with sequential collaboration  
- Ollama Gemma3 local model processing
- Structured output format with exact pattern matching
- Smart hotel selection algorithm (rating + price optimization)
- Pure LLM reasoning without fallback systems

## 🚧 Development Notes

- **Multi-Platform Support**: 
  - **Web**: Fully functional on Chrome/Safari with automated setup
  - **iOS**: Complete support with iOS 18.6 Simulator, iPhone 15+ device types
  - **Android**: Standard Android development setup required
- **Hot Reload**: Supports Flutter hot reload for rapid development across all platforms
- **API Dependencies**: Requires backend server for full functionality
- **Local AI**: All AI processing happens locally via Ollama (Gemma3 model) with dual collaboration modes
- **iOS Configuration**: App Transport Security configured for localhost API access
- **Clean Architecture**: Recent cleanup removed unused test files and dependencies
- **Hotel Logic**: Implements smart selection - best rating first, then lowest price for ties
- **Network Security**: HTTP localhost connections enabled for development on all platforms

## 📝 Configuration

### Backend URL
Update API endpoint in `lib/services/api_service.dart`:
```dart
static const String baseUrl = 'http://localhost:8000';
```

### Theme Customization
Modify theme in `lib/main.dart`:
```dart
theme: ThemeData(
  primarySwatch: Colors.blue,
  // ... other theme properties
),
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper testing
4. Submit a pull request

## 📄 License

This project is part of the Trip Planner AI system.