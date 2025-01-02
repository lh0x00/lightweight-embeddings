# filename: __init__.py

"""
LightweightEmbeddings - FastAPI Application Entry Point

This application provides text and image embeddings using multiple text models and one image model.

Supported text model IDs:
- "multilingual-e5-small"
- "paraphrase-multilingual-MiniLM-L12-v2"
- "bge-m3"

Supported image model ID:
- "google/siglip-base-patch16-256-multilingual"
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import requests
import json
from gradio.routes import mount_gradio_app

# Application metadata
__version__ = "1.0.0"
__author__ = "lamhieu"
__description__ = "Fast, lightweight, multilingual embeddings solution."

# Set your embeddings API URL here (change host/port if needed)
EMBEDDINGS_API_URL = "http://localhost:8000/v1/embeddings"

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
    # Metadata to be displayed
    root_data = {
        "project": "Lightweight Embeddings Service",
        "version": "1.0.0",
        "description": (
            "Fast and efficient multilingual text and image embeddings service "
            "powered by sentence-transformers, supporting 100+ languages and multi-modal inputs"
        ),
        "docs": "https://lamhieu-lightweight-embeddings.hf.space/docs",
        "github": "https://github.com/lh0x00/lightweight-embeddings",
        "spaces": "https://huggingface.co/spaces/lamhieu/lightweight-embeddings",
    }

    # Available model options for the dropdown
    model_options = [
        "multilingual-e5-small",
        "paraphrase-multilingual-MiniLM-L12-v2",
        "bge-m3",
        "google/siglip-base-patch16-256-multilingual",
    ]

    with gr.Blocks(title="Lightweight Embeddings", theme="default") as demo:
        # Project Info
        gr.Markdown(
            """
            # 🎉 **Lightweight Embeddings Service** 🎉
            
            Welcome to the **Lightweight Embeddings** API, a blazing-fast and flexible service 
            supporting **text** and **image** embeddings. Below you'll find key project details:
            """
        )
        gr.Markdown(
            f"""
            **Project**: {root_data["project"]} 🚀  
            **Version**: {root_data["version"]}  
            **Description**: {root_data["description"]}  

            **Docs**: [Click here]({root_data["docs"]}) 😎  
            **GitHub**: [Check it out]({root_data["github"]}) 🐙  
            **Spaces**: [Explore]({root_data["spaces"]}) 🤗  
            """
        )
        gr.Markdown(
            """
            ---
            ### 💡 How to Use
            - Visit **/docs** or **/redoc** for interactive API documentation.
            - Check out **/v1/embeddings** and **/v1/rank** endpoints for direct usage.
            - Or try the simple playground below! Enjoy exploring a multilingual, multi-modal world! 🌏🌐
            """
        )

        # Embeddings Playground
        with gr.Accordion("🔬 Try the Embeddings Playground", open=True):
            gr.Markdown(
                "Enter your **text** or an **image URL**, pick a model, "
                "then click **Generate** to get embeddings from the `/v1/embeddings` API."
            )
            input_text = gr.Textbox(
                label="Input Text or Image URL",
                placeholder="Type some text or paste an image URL...",
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
                lines=15,
                interactive=False,
            )

            # Link the button to the inference function
            generate_btn.click(
                fn=call_embeddings_api,
                inputs=[input_text, model_dropdown],
                outputs=output_json,
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
