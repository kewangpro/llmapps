#!/usr/bin/env python3
"""
Stock Analysis and Trading AI Platform - Main Entry Point
"""

import sys
import os
import logging
import warnings
from pathlib import Path

# Suppress known warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')
warnings.filterwarnings('ignore', message='Dropping a patch because it contains a previously known reference')

# Add src to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import panel as pn
from src.config import Config
from src.ui.app import create_app
from src.ui.design_system import Colors
from src.rl.session_manager import get_session_manager

class DropPatchFilter(logging.Filter):
    """Filter to suppress 'Dropping a patch' warnings from Panel"""
    def filter(self, record):
        return 'Dropping a patch' not in record.getMessage()

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
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("bokeh").setLevel(logging.WARNING)
    logging.getLogger("tornado.access").setLevel(logging.WARNING)

    # Add filter to suppress Panel "Dropping a patch" warnings
    root_logger = logging.getLogger()
    root_logger.addFilter(DropPatchFilter())

    return logging.getLogger(__name__)

def main():
    """Main application entry point"""
    logger = setup_logging()
    logger.info("Starting Stock Analysis and Trading AI Platform")
    
    try:
        # Configure Panel with custom CSS for light theme
        pn.config.console_output = 'disable'
        pn.config.raw_css.append(f"""
            body {{
                background: {Colors.BG_PRIMARY} !important;
                color: {Colors.TEXT_PRIMARY} !important;
            }}
            .bk-root {{
                background: {Colors.BG_PRIMARY} !important;
            }}
            /* Hide theme toggle */
            .theme-toggle {{
                display: none !important;
            }}
        """)
        pn.extension('plotly', notifications=True)

        # Defer creating the Panel application to the server (callable).
        # Passing the `create_app` callable to `pn.serve` makes Panel call it
        # per-session/document which prevents reusing the same models across
        # multiple documents (avoids ImportedStyleSheet already-in-doc errors).
        logger.info("Serving Panel application (create_app will be called per-session)")
        print(f"\n🚀 Stock Analysis and Trading AI Platform")
        print(f"📊 Server starting at: http://{Config.PANEL_HOST}:{Config.PANEL_PORT}")
        print("🔍 Ready to analyze stocks with AI!")
        print("💡 Try queries like: 'Analyze AAPL' or 'Predict GOOGL'\n")

        server_thread = pn.serve(
            create_app,
            port=Config.PANEL_PORT,
            allow_websocket_origin=Config.PANEL_ALLOW_WEBSOCKET_ORIGIN,
            show=True,
            autoreload=False,  # Disable for stability
            threaded=True,
            websocket_max_message_size=100*1024*1024,  # 100MB for large messages
            check_unused_sessions_milliseconds=3600000,  # Check every hour
            unused_session_lifetime_milliseconds=86400000  # 24 hours session lifetime
        )
        while server_thread.is_alive():
            server_thread.join(timeout=1)
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        session_manager = get_session_manager()
        session_manager.shutdown_all()
        print("\n👋 Stock Analysis and Trading AI Platform stopped")
        os._exit(0)
    except Exception as e:
        logger.error(f"Application failed to start: {e}", exc_info=True)
        print(f"\n❌ Failed to start application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
