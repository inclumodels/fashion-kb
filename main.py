"""
main.py — starts the FastAPI server with the scheduler running in background.
Usage: python main.py
"""

import sys
import uvicorn
from api.server import app
from config import API_HOST, API_PORT

if __name__ == "__main__":
    print("=" * 50)
    print("  Fashion KB — Phase 1")
    print("=" * 50)
    print(f"  API:      http://{API_HOST}:{API_PORT}")
    print(f"  Docs:     http://localhost:{API_PORT}/docs")
    print(f"  Frontend: run ->  streamlit run frontend/app.py")
    print("=" * 50)
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")
