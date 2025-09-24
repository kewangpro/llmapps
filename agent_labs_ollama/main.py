#!/usr/bin/env python3
"""
Agent Labs - Cloud Deployment Entry Point
"""
import os
import sys
import logging
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# Set environment variables for cloud deployment
os.environ["HOST"] = "0.0.0.0"
os.environ["PORT"] = os.environ.get("PORT", "3000")

# Configure logging for cloud
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    import uvicorn
    from backend.main import app
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    # Update CORS for cloud deployment
    from fastapi.middleware.cors import CORSMiddleware

    # Remove existing CORS middleware and add cloud-compatible one
    app.user_middleware = []
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins in cloud
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve Next.js static export files
    frontend_path = Path(__file__).parent / "frontend" / "out"
    if frontend_path.exists():
        # Mount static assets
        static_assets = frontend_path / "_next"
        if static_assets.exists():
            app.mount("/_next", StaticFiles(directory=str(static_assets)), name="next_static")

        # Serve the main page
        @app.get("/")
        async def serve_frontend():
            index_path = frontend_path / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            else:
                return {"message": "Agent Labs API - Frontend not available", "status": "API only"}

        # Serve other static files
        app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")

    port = int(os.environ.get("PORT", 3000))

    print(f"🚀 Starting Agent Labs on port {port}")
    print(f"🌐 Environment: {'PRODUCTION' if os.environ.get('GAE_ENV') else 'DEVELOPMENT'}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )