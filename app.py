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

# Create the Gradio interface with an iframe pointing to our Flask app!
# This forces the browser to render the Flask page as a standalone document,
# executing the JavaScript script tag perfectly instead of silencing it (which happens inside gr.HTML)
with gr.Blocks(title="你是啥杯 - 明日方舟干员神经网络评估仪") as demo:
    gr.HTML(
        '<iframe src="api_server/" style="width: 100%; height: 950px; border: none; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.05);"></iframe>'
    )

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
