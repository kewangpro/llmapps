#!/usr/bin/env python3
"""
Ollama Desktop Chat - PyQt6 Native Application
Converted from Electron app with all functionality preserved and native styling
"""

import sys
import os
import json
import asyncio
import aiohttp
import base64
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import mimetypes
import PyPDF2
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QComboBox, QLabel, QSplitter,
    QScrollArea, QFrame, QFileDialog, QMessageBox, QProgressBar,
    QGroupBox, QSizePolicy, QTextBrowser
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QUrl
from PyQt6.QtGui import QFont, QPixmap, QIcon, QTextDocument, QTextCursor, QAction, QPainter, QColor


class OllamaAPI:
    """Handle communication with Ollama API"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_models(self) -> List[Dict[str, Any]]:
        """Get available Ollama models"""
        logger.info("🔄 Fetching Ollama models...")
        try:
            timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout for model list
            async with self.session.get(f"{self.base_url}/api/tags", timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    models = []
                    for model in data.get('models', []):
                        models.append({
                            'name': model['name'],
                            'display_name': model['name'].replace(':latest', ''),
                            'size': model.get('size', 0),
                            'family': model.get('details', {}).get('family', 'unknown')
                        })
                    logger.info(f"✅ Found {len(models)} Ollama models: {', '.join([m['display_name'] for m in models])}")
                    return models
                else:
                    logger.error(f"❌ Failed to fetch models: HTTP {response.status}")
                    return []
        except Exception as e:
            logger.error(f"❌ Error getting models: {e}")
            return []
    
    async def chat(self, message: str, model: str, images: Optional[List[str]] = None, 
                   conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Send chat message to Ollama"""
        logger.info(f"💬 Chat request: model={model}, hasImages={bool(images)}, messageLength={len(message)}, historyLength={len(conversation_history or [])}")
        try:
            # Build messages array
            messages = conversation_history or []
            
            message_obj = {"role": "user", "content": message}
            if images:
                message_obj["images"] = images
                logger.info(f"🖼️ Processing {len(images)} image(s) with vision model")
            
            messages.append(message_obj)
            
            payload = {
                "model": model,
                "messages": messages,
                "stream": False
            }
            
            # Set timeout - longer for vision models processing images
            timeout_seconds = 300 if images else 120  # 5 min for images, 2 min for text
            timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            logger.info(f"⏱️ Request timeout set to {timeout_seconds} seconds")
            
            async with self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data.get("message", {}).get("content", "")
                    logger.info(f"✅ Chat response received: {len(content)} characters")
                    return {
                        "success": True,
                        "content": content
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Chat error: HTTP {response.status}: {error_text}")
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}"
                    }
        except asyncio.TimeoutError:
            timeout_msg = f"Request timed out after {timeout_seconds} seconds. The model may be taking longer than expected to process this request."
            logger.error(f"⏱️ Chat timeout: {timeout_msg}")
            return {"success": False, "error": timeout_msg}
        except Exception as e:
            import traceback
            error_details = f"{type(e).__name__}: {str(e)}"
            logger.error(f"❌ Chat exception: {error_details}")
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            return {"success": False, "error": error_details}


class ChatWorker(QThread):
    """Worker thread for chat requests"""
    response_received = pyqtSignal(dict)
    
    def __init__(self, message: str, model: str, images: Optional[List[str]] = None,
                 conversation_history: Optional[List[Dict]] = None):
        super().__init__()
        self.message = message
        self.model = model
        self.images = images
        self.conversation_history = conversation_history
    
    def run(self):
        """Run the chat request in a separate thread"""
        try:
            # Run async code in thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def chat_request():
                async with OllamaAPI() as api:
                    result = await api.chat(
                        self.message, self.model, self.images, self.conversation_history
                    )
                    return result
            
            result = loop.run_until_complete(chat_request())
            loop.close()
            
            self.response_received.emit(result)
            
        except Exception as e:
            import traceback
            error_details = f"{type(e).__name__}: {str(e)}"
            logger.error(f"❌ ChatWorker exception: {error_details}")
            logger.error(f"🔍 ChatWorker traceback: {traceback.format_exc()}")
            self.response_received.emit({"success": False, "error": error_details})


class ModelLoader(QThread):
    """Worker thread for loading models"""
    models_loaded = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    
    def run(self):
        """Load models in background"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def load_models():
                async with OllamaAPI() as api:
                    models = await api.get_models()
                    return models
            
            models = loop.run_until_complete(load_models())
            loop.close()
            
            self.models_loaded.emit(models)
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class FileHandler:
    """Handle different file types"""
    
    @staticmethod
    def is_image_file(file_path: str) -> bool:
        """Check if file is an image"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.tiff', '.ico', '.heic', '.heif', '.avif'}
        return Path(file_path).suffix.lower() in image_extensions
    
    @staticmethod
    def is_video_file(file_path: str) -> bool:
        """Check if file is a video"""
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
        return Path(file_path).suffix.lower() in video_extensions
    
    @staticmethod
    def is_pdf_file(file_path: str) -> bool:
        """Check if file is a PDF"""
        return Path(file_path).suffix.lower() == '.pdf'
    
    @staticmethod
    def is_text_file(file_path: str) -> bool:
        """Check if file is a text file"""
        text_extensions = {'.txt', '.md', '.py', '.js', '.ts', '.html', '.css', '.json', '.xml', '.csv'}
        return Path(file_path).suffix.lower() in text_extensions
    
    @staticmethod
    def is_binary_file(file_path: str) -> bool:
        """Check if file is binary (fallback for unknown types)"""
        return not (FileHandler.is_image_file(file_path) or 
                   FileHandler.is_video_file(file_path) or 
                   FileHandler.is_pdf_file(file_path) or 
                   FileHandler.is_text_file(file_path))
    
    @staticmethod
    def read_image_file(file_path: str) -> Dict[str, Any]:
        """Read image file and convert to base64"""
        logger.info(f"🖼️ Reading image file: {file_path}")
        try:
            # Check for HEIC files which need conversion for Ollama compatibility
            if Path(file_path).suffix.lower() in {'.heic', '.heif'}:
                logger.info("📱 HEIC format detected - converting for Ollama compatibility")
                try:
                    from PIL import Image
                    import pillow_heif
                    
                    # Register HEIF opener with Pillow
                    pillow_heif.register_heif_opener()
                    
                    # Convert HEIC to JPEG
                    with Image.open(file_path) as img:
                        # Convert to RGB if necessary (HEIC can have different color modes)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Save to bytes as JPEG
                        from io import BytesIO
                        output = BytesIO()
                        img.save(output, format='JPEG', quality=85)
                        image_data = output.getvalue()
                        
                    logger.info(f"✅ HEIC converted to JPEG: {len(image_data)} bytes")
                    
                except ImportError:
                    logger.warning("⚠️ HEIC conversion libraries not available - trying direct read")
                    logger.warning("💡 Install pillow-heif for HEIC support: pip install pillow-heif")
                    # Fallback to direct read (may not work with Ollama)
                    with open(file_path, 'rb') as f:
                        image_data = f.read()
                except Exception as e:
                    logger.error(f"❌ HEIC conversion failed: {e}")
                    # Fallback to direct read
                    with open(file_path, 'rb') as f:
                        image_data = f.read()
            else:
                # Standard image formats
                with open(file_path, 'rb') as f:
                    image_data = f.read()
                
            base64_data = base64.b64encode(image_data).decode('utf-8')
            logger.info(f"✅ Image file processed: {len(image_data)} bytes -> {len(base64_data)} base64 chars")
            return {"success": True, "base64": base64_data}
            
        except Exception as e:
            logger.error(f"❌ Image file read error: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def read_text_file(file_path: str) -> Dict[str, Any]:
        """Read text file"""
        logger.info(f"📄 Reading text file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info(f"✅ Text file read: {len(content)} characters")
                return {"success": True, "content": content}
        except Exception as e:
            logger.error(f"❌ Text file read error: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def read_pdf_file(file_path: str) -> Dict[str, Any]:
        """Read PDF file and extract text"""
        logger.info(f"📋 Reading PDF file: {file_path}")
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text_content = ""
                
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
                
                page_count = len(pdf_reader.pages)
                logger.info(f"📋 PDF parsed: {len(text_content)} characters, {page_count} pages")
                
                if text_content.strip():
                    logger.info(f"✅ PDF parsed successfully: {len(text_content.strip())} characters extracted")
                    return {
                        "success": True,
                        "content": text_content.strip(),
                        "page_count": page_count,
                        "is_text_based": True
                    }
                else:
                    # Image-based PDF
                    file_size = os.path.getsize(file_path)
                    size_kb = round(file_size / 1024)
                    logger.info(f"📋 Image-based PDF detected: {size_kb} KB, {page_count} pages")
                    return {
                        "success": True,
                        "content": f"PDF Document ({size_kb} KB, {page_count} pages)\n\nThis PDF appears to contain images or scanned content with no extractable text. You can still ask questions about it.",
                        "page_count": page_count,
                        "is_image_based": True
                    }
                    
        except Exception as e:
            logger.error(f"❌ PDF parsing error: {e}")
            try:
                # Fallback: provide basic info
                file_size = os.path.getsize(file_path)
                size_kb = round(file_size / 1024)
                logger.info(f"📋 PDF fallback info: {size_kb} KB")
                return {
                    "success": True,
                    "content": f"PDF File ({size_kb} KB)\n\nError extracting text from this PDF: {str(e)}\n\nYou can still ask questions about this file.",
                    "page_count": "?",
                    "has_error": True,
                    "error": str(e)
                }
            except:
                return {"success": False, "error": str(e)}


class OllamaDesktopChat(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        logger.info("💬 Starting Ollama Desktop Chat...")
        self.setWindowTitle("💬 Ollama Desktop Chat")
        self.setMinimumSize(1200, 800)
        
        # Application state
        self.available_models = []
        self.current_model = None
        self.vision_model = None
        self.previous_model = None
        self.file_context = {
            'file_path': None,
            'file_content': None,
            'is_image': False,
            'is_pdf': False,
            'is_video': False,
            'is_binary': False,
            'conversation_history': []
        }
        
        logger.info("📱 Main window created")
        
        # Setup UI
        self.setup_ui()
        logger.info("🎨 UI setup complete")
        
        # Load models on startup
        self.load_models()
        logger.info("✅ Application ready")
    
    def setup_ui(self):
        """Setup the user interface with native styling"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Header with model selection
        self.setup_header(main_layout)
        
        # Main content area with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)
        
        # Left side - Chat area
        self.setup_chat_area(splitter)
        
        # Right side - File preview (hidden by default)
        self.setup_file_preview(splitter)
        
        # Set splitter proportions
        splitter.setSizes([800, 400])
        
    def setup_header(self, parent_layout):
        """Setup header with model selection"""
        header_group = QGroupBox("💬 Ollama Desktop Chat")
        header_layout = QHBoxLayout(header_group)
        
        # Model selection
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        
        self.refresh_models_btn = QPushButton("🔄")
        self.refresh_models_btn.setFixedSize(30, 30)
        self.refresh_models_btn.clicked.connect(self.load_models)
        self.refresh_models_btn.setToolTip("Refresh model list")
        
        header_layout.addWidget(model_label)
        header_layout.addWidget(self.model_combo)
        header_layout.addWidget(self.refresh_models_btn)
        header_layout.addStretch()
        
        parent_layout.addWidget(header_group)
    
    def setup_chat_area(self, parent_splitter):
        """Setup the main chat area"""
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        
        # File controls
        self.setup_file_controls(chat_layout)
        
        # Context indicator
        self.context_indicator = QLabel("💬 File conversation active - Ask follow-up questions!")
        self.context_indicator.setStyleSheet("background-color: #e8f4f8; padding: 8px; border-radius: 4px; color: #0066cc;")
        self.context_indicator.setVisible(False)
        chat_layout.addWidget(self.context_indicator)
        
        # Chat messages area
        self.chat_display = QTextBrowser()
        self.chat_display.setMinimumHeight(400)
        
        # Set clean, native styling for chat that respects system theme
        self.chat_display.setStyleSheet("""
            QTextBrowser {
                border: 1px solid palette(mid);
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
                line-height: 1.4;
            }
        """)
        
        chat_layout.addWidget(self.chat_display, 1)
        
        # Input area
        self.setup_input_area(chat_layout)
        
        parent_splitter.addWidget(chat_widget)
    
    def setup_file_controls(self, parent_layout):
        """Setup file controls"""
        file_group = QGroupBox("📄 File Analysis")
        file_layout = QHBoxLayout(file_group)
        
        # File selection
        self.select_file_btn = QPushButton("📄 Select File")
        self.select_file_btn.clicked.connect(self.select_file)
        
        # File path display
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet("color: gray; font-style: italic;")
        
        # Clear context button
        self.clear_context_btn = QPushButton("🗑️ Clear Context")
        self.clear_context_btn.clicked.connect(self.clear_file_context)
        self.clear_context_btn.setVisible(False)
        
        # Toggle preview button
        self.toggle_preview_btn = QPushButton("👁️ Preview")
        self.toggle_preview_btn.setCheckable(True)
        self.toggle_preview_btn.clicked.connect(self.toggle_file_preview)
        self.toggle_preview_btn.setVisible(False)
        
        file_layout.addWidget(self.select_file_btn)
        file_layout.addWidget(self.file_path_label, 1)
        file_layout.addWidget(self.clear_context_btn)
        file_layout.addWidget(self.toggle_preview_btn)
        
        parent_layout.addWidget(file_group)
    
    def setup_input_area(self, parent_layout):
        """Setup input area"""
        input_widget = QWidget()
        input_layout = QHBoxLayout(input_widget)
        
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask me anything or select a file to analyze...")
        self.chat_input.returnPressed.connect(self.send_message)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)
        self.send_btn.setDefault(True)
        
        self.clear_chat_btn = QPushButton("Clear")
        self.clear_chat_btn.clicked.connect(self.clear_chat)
        
        input_layout.addWidget(self.chat_input, 1)
        input_layout.addWidget(self.send_btn)
        input_layout.addWidget(self.clear_chat_btn)
        
        parent_layout.addWidget(input_widget)
    
    def setup_file_preview(self, parent_splitter):
        """Setup file preview sidebar"""
        self.file_preview_widget = QWidget()
        preview_layout = QVBoxLayout(self.file_preview_widget)
        
        # Preview header
        preview_header = QWidget()
        header_layout = QHBoxLayout(preview_header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        preview_title = QLabel("📄 File Preview")
        preview_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(25, 25)
        close_btn.clicked.connect(self.hide_file_preview)
        
        header_layout.addWidget(preview_title)
        header_layout.addStretch()
        header_layout.addWidget(close_btn)
        
        preview_layout.addWidget(preview_header)
        
        # Preview content
        self.preview_content = QScrollArea()
        self.preview_content.setWidgetResizable(True)
        
        # Default content
        default_widget = QWidget()
        default_layout = QVBoxLayout(default_widget)
        default_label = QLabel("Select a file to preview its content here")
        default_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        default_label.setStyleSheet("color: gray; font-style: italic; padding: 50px;")
        default_layout.addWidget(default_label)
        self.preview_content.setWidget(default_widget)
        
        preview_layout.addWidget(self.preview_content, 1)
        
        # Initially hidden
        self.file_preview_widget.setVisible(False)
        parent_splitter.addWidget(self.file_preview_widget)
    
    def load_models(self):
        """Load available Ollama models"""
        self.refresh_models_btn.setText("⏳")
        self.refresh_models_btn.setEnabled(False)
        
        self.model_loader = ModelLoader()
        self.model_loader.models_loaded.connect(self.on_models_loaded)
        self.model_loader.error_occurred.connect(self.on_models_error)
        self.model_loader.start()
    
    def on_models_loaded(self, models):
        """Handle loaded models"""
        self.available_models = models
        self.model_combo.clear()
        
        if not models:
            self.model_combo.addItem("No models available")
            return
        
        # Filter out embedding models
        chat_models = [m for m in models if 'embed' not in m['family'] and 'embed' not in m['name']]
        
        # Find vision model
        self.vision_model = None
        for model in chat_models:
            if 'vision' in model['name'].lower() or model['family'] == 'mllama':
                self.vision_model = model
                break
        
        # Populate combo box
        for model in chat_models:
            display_name = model['display_name']
            size_gb = model['size'] / (1024 * 1024 * 1024)
            display_text = f"{display_name} ({size_gb:.1f}GB)"
            
            if model == self.vision_model:
                display_text += " 👁️"
            
            self.model_combo.addItem(display_text, model)
        
        # Set default model (prefer gemma3, then vision, then first)
        default_index = 0
        gemma_found = False
        for i, model in enumerate(chat_models):
            if 'gemma3' in model['name'].lower():
                default_index = i
                gemma_found = True
                break
        
        if not gemma_found and self.vision_model:
            for i, model in enumerate(chat_models):
                if model == self.vision_model:
                    default_index = i
                    break
        
        self.model_combo.setCurrentIndex(default_index)
        self.current_model = chat_models[default_index] if chat_models else None
        
        # Re-enable refresh button
        self.refresh_models_btn.setText("🔄")
        self.refresh_models_btn.setEnabled(True)
        
        if self.current_model:
            print(f"Using model: {self.current_model['name']}")
    
    def on_models_error(self, error):
        """Handle model loading error"""
        self.model_combo.clear()
        self.model_combo.addItem("❌ Failed to load models")
        self.refresh_models_btn.setText("🔄")
        self.refresh_models_btn.setEnabled(True)
        
        QMessageBox.warning(self, "Error", f"Failed to load models: {error}")
    
    def select_file(self):
        """Open file selection dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            "",
            "All Files (*);;Images (*.jpg *.jpeg *.png *.gif *.bmp *.webp *.svg *.tiff);;Videos (*.mp4 *.mov *.avi *.mkv *.webm *.flv *.wmv *.m4v);;Documents (*.pdf *.txt *.md *.doc *.docx);;Code Files (*.py *.js *.ts *.html *.css *.json *.xml)"
        )
        
        if file_path:
            self.load_file(file_path)
    
    def load_file(self, file_path: str):
        """Load and analyze file"""
        logger.info(f"📄 Loading file: {file_path}")
        try:
            # Reset file context
            self.file_context = {
                'file_path': file_path,
                'file_content': None,
                'is_image': FileHandler.is_image_file(file_path),
                'is_pdf': FileHandler.is_pdf_file(file_path),
                'is_video': FileHandler.is_video_file(file_path),
                'is_binary': FileHandler.is_binary_file(file_path),
                'conversation_history': []
            }
            
            # Log file type detection
            file_type = "unknown"
            if self.file_context['is_image']:
                file_type = "image"
            elif self.file_context['is_pdf']:
                file_type = "PDF"
            elif FileHandler.is_text_file(file_path):
                file_type = "text"
            elif self.file_context['is_video']:
                file_type = "video"
            else:
                file_type = "binary"
            logger.info(f"📊 File type detected: {file_type}")
            
            # Update UI
            file_name = Path(file_path).name
            self.file_path_label.setText(f"Loaded: {file_name}")
            self.file_path_label.setStyleSheet("color: #0066cc; font-weight: bold;")
            
            # Show context controls
            self.clear_context_btn.setVisible(True)
            self.toggle_preview_btn.setVisible(True)
            self.context_indicator.setVisible(True)
            
            # Load file content based on type
            if self.file_context['is_image']:
                self.load_image_file(file_path)
            elif self.file_context['is_pdf']:
                self.load_pdf_file(file_path)
            elif FileHandler.is_text_file(file_path):
                self.load_text_file(file_path)
            elif self.file_context['is_video']:
                self.load_video_file(file_path)
            else:
                self.load_binary_file(file_path)
                
            # Update input placeholder
            self.chat_input.setPlaceholderText("Ask something about the file...")
            logger.info(f"✅ File loaded successfully: {file_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load file: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
    
    def load_image_file(self, file_path: str):
        """Load image file"""
        result = FileHandler.read_image_file(file_path)
        if result['success']:
            self.file_context['file_content'] = result['base64']
            self.switch_to_vision_model()
            self.add_system_message(f"📷 Image loaded: {Path(file_path).name}. You can now ask questions about this image!")
            self.show_image_preview(file_path, result['base64'])
        else:
            QMessageBox.warning(self, "Error", f"Failed to load image: {result['error']}")
    
    def load_pdf_file(self, file_path: str):
        """Load PDF file"""
        result = FileHandler.read_pdf_file(file_path)
        if result['success']:
            self.file_context['file_content'] = result['content']
            self.switch_back_to_previous_model()
            self.add_system_message(f"📋 PDF loaded: {Path(file_path).name}. You can now ask questions about this PDF!")
            self.show_pdf_preview(file_path, result)
        else:
            QMessageBox.warning(self, "Error", f"Failed to load PDF: {result['error']}")
    
    def load_text_file(self, file_path: str):
        """Load text file"""
        result = FileHandler.read_text_file(file_path)
        if result['success']:
            self.file_context['file_content'] = result['content']
            self.switch_back_to_previous_model()
            self.add_system_message(f"📄 Text file loaded: {Path(file_path).name}. You can now ask questions about this file!")
            self.show_text_preview(file_path, result['content'])
        else:
            QMessageBox.warning(self, "Error", f"Failed to load text file: {result['error']}")
    
    def load_video_file(self, file_path: str):
        """Load video file (metadata only)"""
        self.switch_back_to_previous_model()
        self.add_system_message(f"🎥 Video file loaded: {Path(file_path).name}. Note: Video content analysis requires specialized models. You can ask general questions about video files.")
        self.show_video_preview(file_path)
    
    def load_binary_file(self, file_path: str):
        """Load binary file (metadata only)"""
        self.switch_back_to_previous_model()
        self.add_system_message(f"📦 Binary file loaded: {Path(file_path).name}. This file contains binary data. You can ask questions about the file format or general information.")
        self.show_binary_preview(file_path)
    
    def switch_to_vision_model(self):
        """Switch to vision model for image analysis"""
        if self.vision_model:
            current_model = self.model_combo.currentData()
            if current_model != self.vision_model:
                self.previous_model = current_model
                # Find vision model index
                for i in range(self.model_combo.count()):
                    if self.model_combo.itemData(i) == self.vision_model:
                        self.model_combo.setCurrentIndex(i)
                        self.current_model = self.vision_model
                        self.add_system_message(f"🔄 Switched to {self.vision_model['display_name']} for image analysis")
                        break
    
    def switch_back_to_previous_model(self):
        """Switch back to previous model"""
        if self.previous_model:
            # Find previous model index
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == self.previous_model:
                    self.model_combo.setCurrentIndex(i)
                    self.current_model = self.previous_model
                    self.add_system_message(f"🔄 Switched back to {self.previous_model['display_name']}")
                    self.previous_model = None
                    break
    
    def show_file_preview(self):
        """Show file preview sidebar"""
        if not self.file_preview_widget.isVisible():
            self.file_preview_widget.setVisible(True)
            self.toggle_preview_btn.setChecked(True)
    
    def hide_file_preview(self):
        """Hide file preview sidebar"""
        self.file_preview_widget.setVisible(False)
        self.toggle_preview_btn.setChecked(False)
    
    def toggle_file_preview(self):
        """Toggle file preview visibility"""
        if self.toggle_preview_btn.isChecked():
            self.show_file_preview()
        else:
            self.hide_file_preview()
    
    def show_image_preview(self, file_path: str, base64_data: str):
        """Show image in preview pane"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # File info
        file_name = Path(file_path).name
        file_size = Path(file_path).stat().st_size / 1024  # KB
        info_label = QLabel(f"🖼️ {file_name} ({file_size:.1f} KB)")
        info_label.setStyleSheet("font-weight: bold; color: #0066cc; padding: 10px;")
        layout.addWidget(info_label)
        
        # Image display
        try:
            image_data = base64.b64decode(base64_data)
            pixmap = QPixmap()
            pixmap.loadFromData(image_data)
            
            if not pixmap.isNull():
                # Scale image to fit preview
                scaled_pixmap = pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                
                image_label = QLabel()
                image_label.setPixmap(scaled_pixmap)
                image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(image_label)
        except Exception as e:
            error_label = QLabel(f"Error displaying image: {str(e)}")
            error_label.setStyleSheet("color: red; padding: 10px;")
            layout.addWidget(error_label)
        
        layout.addStretch()
        self.preview_content.setWidget(widget)
        self.show_file_preview()
    
    def show_text_preview(self, file_path: str, content: str):
        """Show text content in preview pane"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # File info
        file_name = Path(file_path).name
        file_size = Path(file_path).stat().st_size / 1024  # KB
        char_count = len(content)
        info_label = QLabel(f"📄 {file_name} ({file_size:.1f} KB, {char_count:,} characters)")
        info_label.setStyleSheet("font-weight: bold; color: #0066cc; padding: 10px;")
        layout.addWidget(info_label)
        
        # Text content
        text_widget = QTextEdit()
        text_widget.setReadOnly(True)
        text_widget.setPlainText(content[:3000])  # Limit preview size
        
        if len(content) > 3000:
            text_widget.append("\n\n... [Content truncated for preview]")
        
        layout.addWidget(text_widget)
        
        self.preview_content.setWidget(widget)
        self.show_file_preview()
    
    def show_pdf_preview(self, file_path: str, pdf_result: Dict):
        """Show PDF content in preview pane"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # File info
        file_name = Path(file_path).name
        file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
        pages = pdf_result.get('page_count', 0)
        
        info_text = f"📋 {file_name} ({file_size:.1f} MB, {pages} pages)"
        if pdf_result.get('is_image_based'):
            info_text += " - Image-based PDF"
        elif pdf_result.get('is_text_based'):
            info_text += " - Text-based PDF"
        elif pdf_result.get('has_error'):
            info_text += " - Parse Error"
        
        info_label = QLabel(info_text)
        info_label.setStyleSheet("font-weight: bold; color: #0066cc; padding: 10px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Content preview
        if pdf_result.get('success') and pdf_result.get('content'):
            content_widget = QTextEdit()
            content_widget.setReadOnly(True)
            content = pdf_result['content']
            preview_content = content[:2000]  # Show first 2000 chars
            content_widget.setPlainText(preview_content)
            
            if len(content) > 2000:
                content_widget.append("\n\n... [Content truncated for preview]")
            
            layout.addWidget(content_widget)
        
        self.preview_content.setWidget(widget)
        self.show_file_preview()
    
    def show_video_preview(self, file_path: str):
        """Show video info in preview pane"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # File info
        file_name = Path(file_path).name
        file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
        
        info_label = QLabel(f"🎥 {file_name} ({file_size:.1f} MB)")
        info_label.setStyleSheet("font-weight: bold; color: #0066cc; padding: 10px;")
        layout.addWidget(info_label)
        
        # Video icon and info
        video_icon = QLabel("🎬")
        video_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_icon.setStyleSheet("font-size: 48px; padding: 20px;")
        layout.addWidget(video_icon)
        
        help_text = QLabel("Video file detected\n\nNote: Video content analysis requires specialized models.\nYou can ask questions about the video file itself.")
        help_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        help_text.setWordWrap(True)
        help_text.setStyleSheet("color: gray; font-style: italic; padding: 10px;")
        layout.addWidget(help_text)
        
        layout.addStretch()
        self.preview_content.setWidget(widget)
        self.show_file_preview()
    
    def show_binary_preview(self, file_path: str):
        """Show binary file info in preview pane"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # File info
        file_name = Path(file_path).name
        file_size = Path(file_path).stat().st_size / 1024  # KB
        
        info_label = QLabel(f"📦 {file_name} ({file_size:.1f} KB)")
        info_label.setStyleSheet("font-weight: bold; color: #0066cc; padding: 10px;")
        layout.addWidget(info_label)
        
        # Binary file icon and info
        binary_icon = QLabel("📦")
        binary_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        binary_icon.setStyleSheet("font-size: 48px; padding: 20px;")
        layout.addWidget(binary_icon)
        
        help_text = QLabel("Binary file detected\n\nThis file contains binary data that cannot be displayed as text.\nYou can ask questions about the file format or general information.")
        help_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        help_text.setWordWrap(True)
        help_text.setStyleSheet("color: gray; font-style: italic; padding: 10px;")
        layout.addWidget(help_text)
        
        layout.addStretch()
        self.preview_content.setWidget(widget)
        self.show_file_preview()
    
    def clear_file_context(self):
        """Clear file context and reset UI"""
        # Switch back from vision model if currently using one
        current_model = self.model_combo.currentData()
        if current_model and self.vision_model and current_model == self.vision_model and self.previous_model:
            logger.info(f"🔄 Auto-switching back from vision model to previous model")
            self.switch_back_to_previous_model()
        
        self.file_context = {
            'file_path': None,
            'file_content': None,
            'is_image': False,
            'is_pdf': False,
            'is_video': False,
            'is_binary': False,
            'conversation_history': []
        }
        
        self.file_path_label.setText("No file selected")
        self.file_path_label.setStyleSheet("color: gray; font-style: italic;")
        
        self.clear_context_btn.setVisible(False)
        self.toggle_preview_btn.setVisible(False)
        self.context_indicator.setVisible(False)
        
        self.chat_input.setPlaceholderText("Ask me anything or select a file to analyze...")
        
        self.hide_file_preview()
        
        self.add_system_message("File context cleared. You can now chat normally or select a new file.")
    
    def send_message(self):
        """Send chat message"""
        message = self.chat_input.text().strip()
        if not message:
            return
        
        logger.info(f"📝 User message: {message[:100]}{'...' if len(message) > 100 else ''}")
        
        current_model_data = self.model_combo.currentData()
        if not current_model_data:
            QMessageBox.warning(self, "Error", "No model selected")
            return
        
        model_name = current_model_data['name']
        logger.info(f"🤖 Using model: {model_name}")
        
        # Add user message to chat
        self.add_user_message(message)
        self.chat_input.clear()
        
        # Show loading message
        loading_msg = self.add_system_message("🤔 Thinking...")
        
        # Prepare message for API
        api_message = message
        images = None
        conversation_history = []
        
        # Add file context if available
        if self.file_context['file_path']:
            logger.info(f"📂 Using file context: {Path(self.file_context['file_path']).name}")
            conversation_history = self.file_context['conversation_history'].copy()
            
            if self.file_context['is_image'] and self.file_context['file_content']:
                images = [self.file_context['file_content']]
                if not conversation_history:
                    # First message about image
                    api_message = f"I'm looking at an image file. {message}"
                logger.info(f"🖼️ Including image in request")
            elif self.file_context['file_content']:
                if not conversation_history:
                    # First message about file
                    api_message = f"I'm analyzing this file content:\n\n{self.file_context['file_content'][:4000]}\n\nUser question: {message}"
                logger.info(f"📄 Including file content in request: {len(self.file_context['file_content'])} characters")
        
        # Send chat request
        logger.info("💬 Starting chat request...")
        self.chat_worker = ChatWorker(api_message, model_name, images, conversation_history)
        self.chat_worker.response_received.connect(lambda response: self.handle_chat_response(response, message, loading_msg))
        self.chat_worker.start()
    
    def handle_chat_response(self, response, original_message, loading_msg):
        """Handle chat response"""
        # Remove loading message if it exists
        if loading_msg:
            try:
                loading_msg.deleteLater()
            except:
                pass
        
        if response['success']:
            self.add_assistant_message(response['content'])
            
            # Update conversation history if in file context
            if self.file_context['file_path']:
                self.file_context['conversation_history'].append({
                    "role": "user", 
                    "content": original_message
                })
                self.file_context['conversation_history'].append({
                    "role": "assistant", 
                    "content": response['content']
                })
        else:
            self.add_assistant_message(f"❌ Error: {response['error']}")
    
    def add_user_message(self, message: str):
        """Add user message to chat"""
        self.chat_display.append(f"<p><b>You:</b> {message}</p>")
        self.scroll_to_bottom()
    
    def add_assistant_message(self, message: str):
        """Add assistant message to chat"""
        formatted_message = self.format_message(message)
        self.chat_display.append(f"<p><b>Assistant:</b></p><p>{formatted_message}</p>")
        self.scroll_to_bottom()
    
    def add_system_message(self, message: str):
        """Add system message to chat"""
        self.chat_display.append(f"<p><i style='color: palette(bright-text); background-color: palette(alternate-base); padding: 4px 8px; border-radius: 4px; display: inline-block;'>📢 {message}</i></p>")
        self.scroll_to_bottom()
        return None
    
    def format_message(self, content: str) -> str:
        """Format message with theme-aware formatting"""
        import re
        
        # Code blocks ```code``` - theme-aware styling
        content = re.sub(r'```(.*?)```', r'<pre style="background-color: palette(alternate-base); color: palette(text); padding: 8px; border-radius: 4px; margin: 4px 0; border: 1px solid palette(mid);">\1</pre>', content, flags=re.DOTALL)
        
        # Inline code `code` - theme-aware styling
        content = re.sub(r'`([^`]+)`', r'<code style="background-color: palette(alternate-base); color: palette(text); padding: 2px 4px; border-radius: 2px; border: 1px solid palette(mid);">\1</code>', content)
        
        # Bold text **text**
        content = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', content)
        
        # Italic text *text*
        content = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', content)
        
        # Headers - use system colors
        content = re.sub(r'^### (.+)$', r'<h4 style="color: palette(text); margin: 8px 0 4px 0;">\1</h4>', content, flags=re.MULTILINE)
        content = re.sub(r'^## (.+)$', r'<h3 style="color: palette(text); margin: 8px 0 4px 0;">\1</h3>', content, flags=re.MULTILINE)
        content = re.sub(r'^# (.+)$', r'<h2 style="color: palette(text); margin: 8px 0 4px 0;">\1</h2>', content, flags=re.MULTILINE)
        
        # List items
        content = re.sub(r'^[\*\-] (.+)$', r'• \1', content, flags=re.MULTILINE)
        
        # Line breaks
        content = content.replace('\n\n', '</p><p>')
        content = content.replace('\n', '<br>')
        
        return content
    
    def scroll_to_bottom(self):
        """Scroll chat to bottom"""
        QTimer.singleShot(100, lambda: self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        ))
    
    def clear_chat(self):
        """Clear chat messages"""
        self.chat_display.clear()
        self.add_system_message("Chat cleared. Start a new conversation!")


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Ollama Desktop Chat")
    app.setApplicationVersion("1.0.0")
    
    # Set application icon to Ollama-themed icon
    try:
        # Create a custom Ollama-inspired icon
        pixmap = QPixmap(64, 64)
        pixmap.fill(QColor(0, 0, 0, 0))  # Transparent background
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw Ollama "O" with gradient-like effect
        painter.setPen(QColor(70, 130, 180))  # Steel blue outline
        painter.setBrush(QColor(100, 149, 237))  # Cornflower blue fill
        
        # Draw main circle (representing the "O" in Ollama)
        painter.drawEllipse(8, 8, 48, 48)
        
        # Add inner highlight
        painter.setBrush(QColor(135, 206, 250))  # Light sky blue
        painter.drawEllipse(16, 16, 32, 32)
        
        # Add center dot
        painter.setBrush(QColor(255, 255, 255))  # White center
        painter.drawEllipse(28, 28, 8, 8)
        
        painter.end()
        
        icon = QIcon(pixmap)
        app.setWindowIcon(icon)
        logger.info("🦙 Set application icon to custom Ollama-themed icon")
    except Exception as e:
        logger.warning(f"⚠️ Could not set application icon: {e}")
        # Fallback to system icon
        try:
            style = app.style()
            icon = style.standardIcon(style.StandardPixmap.SP_ComputerIcon)
            app.setWindowIcon(icon)
        except:
            pass
    
    # Set native style
    app.setStyle('Fusion')  # Cross-platform native-like style
    
    window = OllamaDesktopChat()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()