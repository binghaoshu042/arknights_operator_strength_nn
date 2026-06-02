import os
import sys

# Set KMP_DUPLICATE_LIB_OK to avoid OpenMP crash in PyTorch on Windows/Linux environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add current directory to path to enable clean absolute imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the Flask application from app/server.py
from app.server import app

if __name__ == "__main__":
    # Hugging Face Spaces binds the app to port 7860 by default
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting Flask App for Hugging Face Spaces on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
