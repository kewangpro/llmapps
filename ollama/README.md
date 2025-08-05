# Ollama Desktop Chat - PyQt6 Edition

A native desktop chat application for interacting with Ollama models, built with Python and PyQt6. This application provides a clean, modern interface for chatting with local Ollama models and analyzing various file types.

## 🚀 Features

### 💬 **Chat Interface**
- Clean, native PyQt6 interface that respects system themes
- Real-time chat with Ollama models
- Comprehensive logging for debugging and monitoring
- Modern message formatting with markdown support

### 🤖 **Model Management**
- Automatic discovery of local Ollama models
- Smart model selection (prefers `gemma3`, then vision models)
- Automatic switching to vision models for image analysis
- Model refresh functionality

### 📄 **File Analysis**
- **Images**: JPG, PNG, GIF, BMP, WebP, SVG, TIFF
  - Base64 encoding for Ollama API
  - Automatic vision model switching
  - Image preview in sidebar
- **PDFs**: Text extraction with PyPDF2
  - Handles text-based and image-based PDFs
  - Page count and content analysis
- **Text Files**: TXT, MD, PY, JS, HTML, CSS, JSON, XML
  - Full content analysis with syntax awareness
- **Videos**: MP4, MOV, AVI, MKV, WebM, FLV, WMV
  - Metadata display and general information
- **Binary Files**: Safe handling with file type identification

### 🎨 **User Interface**
- **Native Styling**: Uses Qt6 native widgets and system themes
- **File Preview**: Toggle-able sidebar with detailed file content
- **Context Management**: Maintains conversation history with files
- **Responsive Layout**: Resizable panels and proper scaling

## 📋 Prerequisites

- Python 3.8 or higher
- Ollama installed and running locally
- Qt6 system libraries (installed with PyQt6)

## 🛠️ Installation

### 1. Clone or Download
Download the `ollama_pyqt.py` file to your desired directory.

### 2. Install Dependencies
```bash
pip install PyQt6>=6.6.0 aiohttp>=3.9.0 PyPDF2>=3.0.0
```

### 3. Start Ollama
Ensure Ollama is running:
```bash
ollama serve
```

### 4. Install Models
Install at least one model:
```bash
ollama pull gemma3
# Optional: Install a vision model for image analysis
ollama pull llama3.2-vision
```

## 📁 File Structure

### PyQt6 Application Files

```
ollama-pyqt/
├── 📄 ollama_pyqt.py          # Main PyQt6 application (~1,130 lines)
├── 📄 requirements-pyqt.txt   # Python dependencies
└── 📄 README-pyqt.md         # Documentation (this file)
```

| File | Description | Size | Purpose |
|------|-------------|------|---------|
| **`ollama_pyqt.py`** | Main application | ~1,130 lines | Complete desktop chat app with file analysis |
| **`requirements-pyqt.txt`** | Dependencies | 6 lines | PyQt6, aiohttp, PyPDF2, Pillow, pillow-heif |
| **`README-pyqt.md`** | Documentation | ~250 lines | Installation guide and usage instructions |

### Dependencies Overview

```python
# requirements-pyqt.txt
PyQt6>=6.6.0        # Native GUI framework
aiohttp>=3.9.0      # Async HTTP client for Ollama API  
PyPDF2>=3.0.0       # PDF text extraction
Pillow>=9.0.0       # Image processing
pillow-heif>=0.10.0 # HEIC/iPhone image support
```

## 🏃 Usage

### Quick Start
```bash
python ollama_pyqt.py
```

### First Time Setup
1. **Launch the application** - The interface will open with model selection
2. **Model Loading** - Models are automatically discovered and loaded
3. **Start Chatting** - Type messages in the input field
4. **File Analysis** - Click "Select File" to analyze documents, images, etc.
5. **Preview Files** - Use the "Preview" button to view file content in the sidebar

### File Analysis Workflow
1. Click **"📄 Select File"** button
2. Choose your file (images, PDFs, text files, videos, etc.)
3. The file is automatically analyzed and context is set
4. **For images**: Automatically switches to vision model
5. **For other files**: Content is extracted and made available
6. Ask questions about the file content
7. Use **"🗑️ Clear Context"** to reset and load a new file

## 🔧 Features in Detail

### Model Selection
- **Automatic Discovery**: Finds all available Ollama models
- **Smart Defaults**: Prefers `gemma3:latest`, falls back to vision models
- **Vision Detection**: Automatically identifies models with vision capabilities
- **Refresh**: Manual model list updates with the 🔄 button

### File Types Supported

| Type | Extensions | Features |
|------|------------|----------|
| **Images** | jpg, png, gif, bmp, webp, svg, tiff | Vision model auto-switch, preview |
| **PDFs** | pdf | Text extraction, page count, content analysis |
| **Text** | txt, md, py, js, html, css, json, xml | Full content analysis |
| **Videos** | mp4, mov, avi, mkv, webm, flv, wmv | Metadata and general info |
| **Binary** | exe, zip, etc. | Safe handling, type identification |

### Logging and Debugging
The application provides comprehensive console logging:
- 🚀 Application startup and initialization
- 🔄 Model loading and selection
- 📄 File operations and type detection
- 💬 Chat requests and responses
- ❌ Error handling and troubleshooting

## 🎛️ Interface Overview

### Main Window Layout
```
┌─────────────────────────────────────────────────────────┐
│ 🔧 Ollama Desktop Chat                                   │
├─────────────────────────────────────────────────────────┤
│ Model: [gemma3 (7.4GB)] [🔄]                            │
├─────────────────────────────────────────────────────────┤
│ 📄 File Analysis                                        │
│ [📄 Select File] Loaded: filename.pdf [🗑️] [👁️]        │
├─────────────────────────────────────────────────────────┤
│ 💬 File conversation active - Ask follow-up questions!   │
├─────────────────────────────────────────────────────────┤
│                                               │ Preview │
│  Chat Messages Area                           │ Sidebar │
│                                               │         │
├─────────────────────────────────────────────────────────┤
│ [Input Field...........................] [Send] [Clear] │
└─────────────────────────────────────────────────────────┘
```

### Controls
- **Model Dropdown**: Select and switch between available models
- **🔄 Refresh**: Update model list
- **📄 Select File**: Choose file for analysis
- **🗑️ Clear Context**: Reset file context
- **👁️ Preview**: Toggle file preview sidebar
- **Send**: Send chat message
- **Clear**: Clear chat history

## 🔍 Troubleshooting

### Common Issues

#### "No models available"
- Ensure Ollama is running: `ollama serve`
- Install a model: `ollama pull gemma3`
- Check API accessibility: `curl http://localhost:11434/api/tags`

#### Application won't start
- Check Python version: `python --version` (needs 3.8+)
- Install dependencies: `pip install PyQt6 aiohttp PyPDF2`
- Check console logs for specific errors

#### File loading issues
- Verify file permissions and accessibility
- Check supported file formats
- For PDFs, ensure file is not corrupted or password-protected

#### Chat not working
- Verify Ollama service is running on port 11434
- Check firewall settings for localhost connections
- Look at console logs for API errors

### Performance Tips
- **File Size**: Keep files under 10MB for best performance
- **Model Selection**: Use smaller models for faster responses
- **Preview**: Close file preview when not needed for better performance
- **Chat History**: Clear chat periodically for optimal performance

## 🏗️ Technical Details

### Architecture
- **Main Application**: `OllamaDesktopChat` - Central Qt6 window
- **API Handler**: `OllamaAPI` - Async Ollama communication
- **File Processing**: `FileHandler` - Multi-format file support
- **Worker Threads**: Background operations for UI responsiveness

### Threading Model
- **Main Thread**: UI operations and user interactions
- **Chat Worker**: Background API requests to Ollama
- **Model Loader**: Async model discovery and loading

### Dependencies
- **PyQt6**: Native desktop GUI framework
- **aiohttp**: Async HTTP client for Ollama API
- **PyPDF2**: PDF text extraction
- **Python Standard Library**: File operations, base64, JSON, etc.

## 🔄 Migration from Electron

This PyQt6 version maintains feature parity with the original Electron application while providing:

### Advantages
- **Native Performance**: True native application with better resource usage
- **System Integration**: Proper system theme support and OS integration  
- **Memory Efficiency**: Lower memory footprint compared to Electron
- **Python Ecosystem**: Easy integration with Python libraries and tools

### Feature Parity
- ✅ All file analysis capabilities
- ✅ Model management and switching
- ✅ Chat interface and formatting
- ✅ File preview functionality
- ✅ Context management and conversation history

## 📝 License

This project maintains the same license as the original Electron application.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application.

---

**Note**: This is a native PyQt6 conversion of the original Electron-based Ollama Desktop Chat, providing the same functionality with improved performance and system integration.
