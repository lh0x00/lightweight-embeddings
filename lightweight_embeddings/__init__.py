# filename: __init__.py

"""
LightweightEmbeddings - FastAPI Application Entry Point

This application provides text and image embeddings using multiple text models and one image model.

Supported text model IDs:
- "multilingual-e5-small"
- "multilingual-e5-base"
- "multilingual-e5-large"
- "snowflake-arctic-embed-l-v2.0"
- "paraphrase-multilingual-MiniLM-L12-v2"
- "paraphrase-multilingual-mpnet-base-v2"
- "bge-m3"

Supported image model ID:
- "siglip-base-patch16-256-multilingual"
"""

import gradio as gr
import requests
import json
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gradio.routes import mount_gradio_app


# Filter out /v1 requests from the access log
class LogFilter(logging.Filter):
    def filter(self, record):
        if record.args and len(record.args) >= 3:
            if "/v1" in str(record.args[2]):
                return True
        return False


logger = logging.getLogger("uvicorn.access")
logger.addFilter(LogFilter())

# Application metadata
__version__ = "1.0.0"
__author__ = "lamhieu"
__description__ = "Fast, lightweight, multilingual embeddings solution."
__metadata__ = {
    "project": "Lightweight Embeddings Service",
    "version": __version__,
    "description": (
        "Fast and efficient multilingual text and image embeddings service "
        "powered by sentence-transformers, supporting 100+ languages and multi-modal inputs"
    ),
    "docs": "https://lamhieu-lightweight-embeddings.hf.space/docs",
    "github": "https://github.com/lh0x00/lightweight-embeddings",
    "spaces": "https://huggingface.co/spaces/lamhieu/lightweight-embeddings",
}

# Set your embeddings API URL here (change host/port if needed)
EMBEDDINGS_API_URL = "http://localhost:7860/v1/embeddings"

# Markdown description for the main interface
APP_DESCRIPTION = f"""
## 🚀 **Lightweight Embeddings API**  

The **Lightweight Embeddings API** is a fast, free, and multilingual service designed for generating embeddings and reranking with support for both **text** and **image** inputs. Get started below by exploring our interactive playground or using the cURL examples provided.

### ✨ Key Features

- **Free, Unlimited, and Multilingual**: A fully free API service with no usage limits, capable of processing text in over 100+ languages to support global applications seamlessly.  
- **Advanced Embedding and Reranking**: Generate high-quality text and image-text embeddings using state-of-the-art models, alongside robust reranking capabilities for enhanced results.  
- **Optimized and Flexible**: Built for speed with lightweight transformer models, efficient backends for rapid inference on low-resource systems, and support for diverse use cases with models.
- **Production-Ready with Ease of Use**: Deploy effortlessly using Docker for a hassle-free setup, and experiment interactively through a **Gradio-powered playground** with comprehensive REST API documentation.  

### 🔗 Links
- [Documentation]({__metadata__["docs"]}) | [GitHub]({__metadata__["github"]}) | [Playground]({__metadata__["spaces"]})
"""

# Initialize FastAPI application
app = FastAPI(
    title="Lightweight Embeddings API",
    description=__description__,
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust if needed for specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include your existing router (which provides /v1/embeddings, /v1/rank, etc.)
from .router import router

app.include_router(router, prefix="/v1")


def call_embeddings_api(user_input: str, selected_model: str) -> str:
    """
    Send a request to the /v1/embeddings endpoint with the given model and input.
    Return a pretty-printed JSON response or an error message.
    """
    payload = {
        "model": selected_model,
        "input": user_input,
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            EMBEDDINGS_API_URL, json=payload, headers=headers, timeout=20
        )
    except requests.exceptions.RequestException as e:
        return f"❌ Network Error: {str(e)}"

    if response.status_code != 200:
        # Provide detailed error message
        return f"❌ API Error {response.status_code}: {response.text}"

    try:
        data = response.json()
        return json.dumps(data, indent=2)
    except ValueError:
        return "❌ Failed to parse JSON from API response."


def create_main_interface():
    """
    Creates a Gradio Blocks interface showing project info and an embeddings playground.
    """
    # Available model options for the dropdown
    model_options = [
        "snowflake-arctic-embed-l-v2.0",
        "multilingual-e5-small",
        "multilingual-e5-base",
        "multilingual-e5-large",
        "paraphrase-multilingual-MiniLM-L12-v2",
        "paraphrase-multilingual-mpnet-base-v2",
        "bge-m3",
        "siglip-base-patch16-256-multilingual",
    ]

    with gr.Blocks(title="Lightweight Embeddings", theme="default") as demo:
        # Project Info
        gr.Markdown(APP_DESCRIPTION)

        # Split Layout: Playground and cURL Examples
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔬 Try the Embeddings Playground")
                input_text = gr.Textbox(
                    label="Input Text or Image URL",
                    placeholder="Enter text or an image URL...",
                    lines=3,
                )
                model_dropdown = gr.Dropdown(
                    choices=model_options,
                    value=model_options[0],
                    label="Select Model",
                )
                generate_btn = gr.Button("Generate Embeddings")
                output_json = gr.Textbox(
                    label="Embeddings API Response",
                    lines=10,
                    interactive=False,
                )

                # Link button to inference function
                generate_btn.click(
                    fn=call_embeddings_api,
                    inputs=[input_text, model_dropdown],
                    outputs=output_json,
                )

            with gr.Column():
                gr.Markdown(
                    """
                  ### 🛠️ cURL Examples

                  **Generate Embeddings**
                  ```bash
                  curl -X 'POST' \\
                    'https://lamhieu-lightweight-embeddings.hf.space/v1/embeddings' \\
                    -H 'accept: application/json' \\
                    -H 'Content-Type: application/json' \\
                    -d '{
                    "model": "snowflake-arctic-embed-l-v2.0",
                    "input": "That is a happy person"
                  }'
                  ```

                  **Perform Ranking**
                  ```bash
                  curl -X 'POST' \\
                    'https://lamhieu-lightweight-embeddings.hf.space/v1/rank' \\
                    -H 'accept: application/json' \\
                    -H 'Content-Type: application/json' \\
                    -d '{
                    "model": "snowflake-arctic-embed-l-v2.0",
                    "queries": "That is a happy person",
                    "candidates": [
                      "That is a happy dog",
                      "That is a very happy person",
                      "Today is a sunny day"
                    ]
                  }'
                  ```
                  """
                )

    return demo


# Create and mount the Gradio Blocks at the root path
main_interface = create_main_interface()
mount_gradio_app(app, main_interface, path="/")


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """
    Initialize resources (like model loading) when the application starts.
    """
    pass


@app.on_event("shutdown")
async def shutdown_event():
    """
    Perform cleanup before the application shuts down.
    """
    pass
