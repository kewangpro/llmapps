#!/usr/bin/env python3
"""
Stock Analysis AI Platform - Main Entry Point
"""

import sys
import os
import logging
from pathlib import Path

# Add src to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import panel as pn
from src.config import Config
from src.ui.components import create_app

def setup_logging():
    """Setup basic logging configuration"""
    Config.ensure_directories()
    
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL.upper()),
        format=Config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Config.LOG_DIR / "app.log")
        ]
    )
    
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("bokeh").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def main():
    """Main application entry point"""
    logger = setup_logging()
    logger.info("Starting Stock Analysis AI Platform")
    
    try:
        # Configure Panel
        pn.config.console_output = 'disable'
        pn.extension('plotly')
        
        # Create the application
        logger.info("Creating Panel application")
        app = create_app()
        
        # Serve the application
        logger.info(f"Starting server on {Config.PANEL_HOST}:{Config.PANEL_PORT}")
        print(f"\n🚀 Stock Analysis AI Platform")
        print(f"📊 Server starting at: http://{Config.PANEL_HOST}:{Config.PANEL_PORT}")
        print("🔍 Ready to analyze stocks with AI!")
        print("💡 Try queries like: 'Analyze AAPL' or 'Predict GOOGL'\n")
        
        pn.serve(
            app,
            port=Config.PANEL_PORT,
            allow_websocket_origin=Config.PANEL_ALLOW_WEBSOCKET_ORIGIN,
            show=True,
            autoreload=False,  # Disable for stability
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        print("\n👋 Stock Analysis AI Platform stopped")
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        print(f"\n❌ Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
