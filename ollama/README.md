# Ollama Desktop Applications

A collection of native desktop applications for interacting with Ollama models, built with Python. Features a comprehensive PyQt6 chat application with multi-format file analysis capabilities and a PDF-to-PowerPoint conversion tool.

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
- **Images**: JPG, PNG, GIF, BMP, WebP, SVG, TIFF, HEIC, HEIF, AVIF
  - HEIC/iPhone image conversion support
  - Base64 encoding for Ollama API
  - Automatic vision model switching
  - Image preview in sidebar
- **Documents**: 
  - **PDFs**: Text extraction with PyPDF2, handles text-based and image-based PDFs
  - **Word Documents**: DOCX, DOC text extraction from paragraphs and tables
  - **Excel Files**: XLSX, XLS, XLSM data extraction from all sheets
  - **PowerPoint**: PPTX, PPT text extraction from slides, shapes, and tables
- **Text Files**: TXT, MD, PY, JS, HTML, CSS, JSON, XML, CSV
  - Full content analysis with syntax awareness
- **Videos**: MP4, MOV, AVI, MKV, WebM, FLV, WMV, M4V
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
pip install -r requirements.txt
```

Or install manually:
```bash
pip install PyQt6>=6.6.0 aiohttp>=3.9.0 PyPDF2>=3.0.0 Pillow>=9.0.0 pillow-heif>=0.10.0 python-docx>=0.8.11 openpyxl>=3.1.0 python-pptx>=0.6.0
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

## 📁 Applications

### Desktop Chat Application

**`ollama_pyqt.py`** - Full-featured PyQt6 desktop chat application with comprehensive file analysis capabilities.

### Document Processing Tools  

**`gen_ppt.py`** - PDF to PowerPoint converter that uses Ollama to generate presentation slides from PDF documents.

## 📁 File Structure

```
ollama/
├── 📄 ollama_pyqt.py          # Main PyQt6 chat application (1,572 lines)
├── 📄 gen_ppt.py              # PDF to PowerPoint converter (269 lines) 
├── 📄 requirements.txt        # Python dependencies (8 packages)
└── 📄 README.md               # Documentation (this file)
```

| File | Description | Size | Purpose |
|------|-------------|------|---------|
| **`ollama_pyqt.py`** | Chat application | 1,572 lines | Complete desktop chat app with file analysis |
| **`gen_ppt.py`** | PPT generator | 269 lines | Convert PDF documents to PowerPoint presentations |
| **`requirements.txt`** | Dependencies | 8 packages | PyQt6, aiohttp, PyPDF2, python-pptx, etc. |
| **`README.md`** | Documentation | This file | Installation guide and usage instructions |

### Dependencies Overview

```python
# requirements.txt
PyQt6>=6.6.0        # Native GUI framework (desktop chat app)
aiohttp>=3.9.0      # Async HTTP client for Ollama API
PyPDF2>=3.0.0       # PDF text extraction (both apps)
Pillow>=9.0.0       # Image processing
pillow-heif>=0.10.0 # HEIC/iPhone image support
python-docx>=0.8.11 # Word document parsing
openpyxl>=3.1.0     # Excel file processing
python-pptx>=0.6.0  # PowerPoint processing (both apps)
```

## 🏃 Usage

### Desktop Chat Application
```bash
python ollama_pyqt.py
```

**First Time Setup:**
1. **Launch the application** - The interface will open with model selection
2. **Model Loading** - Models are automatically discovered and loaded
3. **Start Chatting** - Type messages in the input field
4. **File Analysis** - Click "Select File" to analyze documents, images, etc.
5. **Preview Files** - Use the "Preview" button to view file content in the sidebar

### PDF to PowerPoint Converter

**GUI Mode:**
```bash
python gen_ppt.py
```

**CLI Mode:**
```bash
python gen_ppt.py --pdf_file path/to/document.pdf
```

**Features:**
- Automatically extracts text from PDF documents
- Uses Ollama to generate structured slide content
- Creates PowerPoint presentation with titles and bullet points
- Output filename automatically derived from PDF name

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
| **Images** | jpg, png, gif, bmp, webp, svg, tiff, heic, heif, avif | Vision model auto-switch, HEIC conversion, preview |
| **Documents** | pdf, docx, doc, xlsx, xls, xlsm, pptx, ppt | Text/data extraction, content analysis, preview |
| **Text** | txt, md, py, js, html, css, json, xml, csv | Full content analysis, syntax highlighting |
| **Videos** | mp4, mov, avi, mkv, webm, flv, wmv, m4v | Metadata and general info |
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
│ [📄 Select File] Loaded: presentation.pptx [🗑️] [👁️]   │
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
- Install dependencies: `pip install -r requirements.txt`
- Check console logs for specific errors

#### File loading issues
- Verify file permissions and accessibility
- Check supported file formats
- For PDFs, ensure file is not corrupted or password-protected
- For Word/Excel/PowerPoint files, ensure python-docx, openpyxl, and python-pptx are installed
- For HEIC images, ensure pillow-heif is installed

#### Chat not working
- Verify Ollama service is running on port 11434
- Check firewall settings for localhost connections
- Look at console logs for API errors

### Performance Tips
- **File Size**: Keep files under 10MB for best performance
- **Model Selection**: Use smaller models for faster responses
- **Preview**: Close file preview when not needed for better performance
- **Chat History**: Clear chat periodically for optimal performance

### Current Project Status
- **Main Chat App**: Fully functional with comprehensive file support
- **PDF Converter**: GUI and CLI modes both working
- **Dependencies**: All required packages specified in requirements.txt
- **Active Files**: 4 core files in working directory

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
- **PyQt6**: Native desktop GUI framework (chat app only)
- **aiohttp**: Async HTTP client for Ollama API (both apps)
- **PyPDF2**: PDF text extraction (both apps)
- **python-docx**: Word document parsing (chat app)
- **openpyxl**: Excel file processing (chat app)
- **python-pptx**: PowerPoint processing (both apps)
- **Pillow & pillow-heif**: Image processing and HEIC support (chat app)
- **Python Standard Library**: File operations, base64, JSON, etc.

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the applications.

## 🔧 Development

### Adding New Features
- Both applications use async/await patterns for Ollama API communication
- The chat app follows PyQt6 MVC architecture patterns
- File handlers are modular and can be extended for new formats

### Testing
- Test with various file formats and sizes
- Verify Ollama model compatibility
- Check UI responsiveness and error handling

## 📊 Project Statistics

- **Total Lines of Code**: ~1,840 lines across all files
- **Main Application**: 1,572 lines of PyQt6 GUI code
- **PDF Converter**: 269 lines with GUI and CLI support  
- **Dependencies**: 8 Python packages for comprehensive functionality
- **File Format Support**: 20+ file extensions across 5 categories
- **Logging**: Comprehensive console output with emoji indicators

---

*Complete desktop solution for Ollama model interaction*