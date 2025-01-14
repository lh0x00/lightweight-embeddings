import gradio as gr
import requests
import json
import logging
import pandas as pd
from typing import Tuple

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
# 🚀 **Lightweight Embeddings API**  

The **Lightweight Embeddings API** is a fast, free, and multilingual service designed for generating embeddings and reranking with support for both **text** and **image** inputs.

### ✨ Features & Privacy

- **Free & Multilingual**: Unlimited API service supporting 100+ languages with no usage restrictions
- **Advanced Processing**: High-quality text and image-text embeddings using state-of-the-art models with reranking capabilities
- **Privacy-First**: No storage of input data (text/images), only anonymous usage statistics for service improvement
- **Production-Ready**: Docker deployment, interactive Gradio playground, and comprehensive REST API documentation
- **Open & Efficient**: Fully open-source codebase using lightweight transformer models for rapid inference

### 🔗 Resources
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
        return json.dumps(data, indent=2, ensure_ascii=False)
    except ValueError:
        return "❌ Failed to parse JSON from API response."


def call_stats_api_df() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calls the /v1/stats endpoint to retrieve analytics data.
    Returns two DataFrames (access_df, tokens_df) constructed from the JSON response.
    """
    url = "https://lamhieu-lightweight-embeddings.hf.space/v1/stats"

    # Fetch stats
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch stats: {response.text}")

    data = response.json()
    access_data = data["access"]
    tokens_data = data["tokens"]

    def build_stats_df(bucket: dict) -> pd.DataFrame:
        """
        Helper to build a DataFrame with columns: Model, total, daily, weekly, monthly, yearly.
        bucket is a dictionary like data["access"] or data["tokens"] in the stats response.
        """
        all_models = set()
        for time_range in ["total", "daily", "weekly", "monthly", "yearly"]:
            all_models.update(bucket[time_range].keys())

        # Prepare a data structure for DataFrame creation
        result_dict = {
            "Model": [],
            "Total": [],
            "Daily": [],
            "Weekly": [],
            "Monthly": [],
            "Yearly": [],
        }

        for model in sorted(all_models):
            result_dict["Model"].append(model)
            result_dict["Total"].append(bucket["total"].get(model, 0))
            result_dict["Daily"].append(bucket["daily"].get(model, 0))
            result_dict["Weekly"].append(bucket["weekly"].get(model, 0))
            result_dict["Monthly"].append(bucket["monthly"].get(model, 0))
            result_dict["Yearly"].append(bucket["yearly"].get(model, 0))

        df = pd.DataFrame(result_dict)
        return df

    access_df = build_stats_df(access_data)
    tokens_df = build_stats_df(tokens_data)
    return access_df, tokens_df


def create_main_interface():
    """
    Creates a Gradio Blocks interface showing project info and an embeddings playground.
    """
    # Available model options for the dropdown
    model_options = [
        "snowflake-arctic-embed-l-v2.0",
        "bge-m3",
        "gte-multilingual-base",
        "paraphrase-multilingual-MiniLM-L12-v2",
        "paraphrase-multilingual-mpnet-base-v2",
        "multilingual-e5-small",
        "multilingual-e5-base",
        "multilingual-e5-large",
        "siglip-base-patch16-256-multilingual",
    ]

    with gr.Blocks(title="Lightweight Embeddings", theme="default") as demo:
        gr.Markdown(APP_DESCRIPTION)
        with gr.Tab("Embeddings Playground"):
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

                    generate_btn.click(
                        fn=call_embeddings_api,
                        inputs=[input_text, model_dropdown],
                        outputs=output_json,
                    )

                with gr.Column():
                    gr.Markdown(
                        """
                    ### 🛠️ cURL Examples

                    **Generate Embeddings (OpenAI compatible)**
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

        # STATS SECTION: display stats in tables
        with gr.Tab("Analytics Stats"):
            stats_btn = gr.Button("Get Stats")
            access_df = gr.DataFrame(
                label="Access Stats",
                headers=["Model", "Total", "Daily", "Weekly", "Monthly", "Yearly"],
                interactive=False,
            )
            tokens_df = gr.DataFrame(
                label="Token Stats",
                headers=["Model", "Total", "Daily", "Weekly", "Monthly", "Yearly"],
                interactive=False,
            )
            stats_btn.click(
                fn=call_stats_api_df, inputs=[], outputs=[access_df, tokens_df]
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
