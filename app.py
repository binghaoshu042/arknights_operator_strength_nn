import os
import sys

# Set KMP_DUPLICATE_LIB_OK to avoid OpenMP crash in PyTorch on Windows/Linux environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add current directory to path to enable clean absolute imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
import gradio as gr

# Import our Flask app
from web_app.server import app as flask_app

# Read the HTML content for Gradio rendering
html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_app", "templates", "index.html")
with open(html_path, "r", encoding="utf-8") as f:
    html_content = f.read()

# Replace prediction endpoint in JS to route through the mounted Flask app
html_content = html_content.replace('"/api/predict"', '"/api_server/api/predict"')

# Create the Gradio interface
with gr.Blocks(title="你是啥杯 - 明日方舟干员神经网络评估仪") as demo:
    gr.HTML(html_content)

# Mount Gradio and Flask under a single FastAPI app
fastapi_app = FastAPI()
fastapi_app.mount("/api_server", WSGIMiddleware(flask_app))

# Mount Gradio app at root "/" so that ModelScope can find Gradio's /config route!
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting Gradio + Flask App on port {port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
