"""Minimal, elegant Gradio playground.

Design principles:

* **In-process** — no HTTP loopback, no ``requests`` dependency. The UI
  reaches the embedding service and analytics directly via ``app.state``.
* **Two tabs** — *Playground* (try it) and *Stats* (observe). No marketing,
  no scrollwall.
* **Sane defaults** — model list reflects the registry, separated by kind.
* **Honest output** — we show the dimension, token usage, and a small
  preview of the vector instead of dumping 1024 floats into the textbox.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .. import __version__
from ..analytics.periods import current_period_keys
from ..core.registry import REGISTRY, ModelKind

logger = logging.getLogger(__name__)


_HEADER_MD = f"""
# Lightweight Embeddings

A multilingual text + image embedding & reranking service.
Model weights are loaded **lazily** on first use.

`v{__version__}` · [API docs](/docs) · [Health](/healthz) · [Metrics](/metrics)
""".strip()


_CURL_MD = """
### HTTP

```bash
# Embeddings (OpenAI-compatible)
curl -X POST :PORT/v1/embeddings \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"multilingual-e5-small","input":"Hello world"}'

# Rerank
curl -X POST :PORT/v1/rank \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"multilingual-e5-small",
       "queries":"happy person",
       "candidates":["happy dog","sunny day"]}'

# List models / health / quota
curl :PORT/v1/models
curl :PORT/healthz
curl :PORT/v1/quota -H 'Authorization: Bearer <TOKEN>'
```

> Add `Authorization: Bearer <TOKEN>` if `LWE_ACCESS_TOKEN` is set.
""".strip()


def _model_choices() -> tuple[list[str], list[str]]:
    text = [s.name for s in REGISTRY if s.kind is ModelKind.TEXT]
    image = [s.name for s in REGISTRY if s.kind is ModelKind.IMAGE]
    return text, image


def _matryoshka_for(model: str) -> list[int]:
    spec = REGISTRY.get(model) if model in REGISTRY else None
    return list(spec.matryoshka_dims) if spec else []


def build_demo(app):
    """Construct the Gradio Blocks UI tied to a FastAPI ``app``."""
    import gradio as gr

    text_models, image_models = _model_choices()
    all_models = [*text_models, *image_models]

    # ---------- handlers ----------
    async def embed_handler(text: str, model: str, dimensions: int | None) -> str:
        text = (text or "").strip()
        if not text:
            return _err("Please enter text or an image URL.")
        service = app.state.service
        try:
            embeddings, spec = await service.generate_embeddings(
                model=model,
                inputs=text,
                normalize=True,
                dimensions=int(dimensions) if dimensions and dimensions > 0 else None,
            )
            tokens = service.count_tokens(spec.name, text) if spec.kind is ModelKind.TEXT else 0
            preview = embeddings[0, :8].round(4).tolist()
            payload: dict[str, Any] = {
                "model": spec.name,
                "kind": spec.kind.value,
                "dimensions": int(embeddings.shape[-1]),
                "tokens": tokens,
                "preview": preview,
                "note": "vector L2-normalised; preview shows first 8 values",
            }
            return json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("UI embed failed: %s", exc)
            return _err(str(exc))

    async def rank_handler(query: str, candidates_blob: str, model: str) -> str:
        query = (query or "").strip()
        candidates = [c.strip() for c in (candidates_blob or "").splitlines() if c.strip()]
        if not query or not candidates:
            return _err("Provide a query and at least one candidate (one per line).")
        service = app.state.service
        try:
            result = await service.rank(model, query, candidates)
            scores = result["cosine_similarities"][0]
            ranked = sorted(
                zip(candidates, scores, strict=False),
                key=lambda x: x[1],
                reverse=True,
            )
            payload = {
                "model": model,
                "tokens": result["usage"]["total_tokens"],
                "ranked": [
                    {"candidate": c, "cosine": round(float(s), 4)} for c, s in ranked
                ],
                "softmax": [round(float(p), 4) for p in result["probabilities"][0]],
            }
            return json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("UI rank failed: %s", exc)
            return _err(str(exc))

    async def stats_handler():
        analytics = app.state.analytics
        try:
            raw = await analytics.stats()
        except Exception as exc:
            logger.warning("UI stats failed: %s", exc)
            return [], []
        keys = current_period_keys()

        def to_table(kind: str) -> list[list[Any]]:
            data = raw.get(kind, {})
            models = sorted({m for periods in data.values() for m in periods})
            return [
                [
                    m,
                    data.get(keys.total, {}).get(m, 0),
                    data.get(keys.day, {}).get(m, 0),
                    data.get(keys.week, {}).get(m, 0),
                    data.get(keys.month, {}).get(m, 0),
                    data.get(keys.year, {}).get(m, 0),
                ]
                for m in models
            ]

        return to_table("access"), to_table("tokens")

    def matryoshka_update(model: str):
        dims = _matryoshka_for(model)
        if not dims:
            return gr.update(visible=False, choices=[], value=None)
        return gr.update(visible=True, choices=[str(d) for d in dims], value=str(dims[-1]))

    def _to_int(value: str | None) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except ValueError:
            return None

    def embed_sync(text, model, dim_str):
        return asyncio.run(embed_handler(text, model, _to_int(dim_str)))

    def rank_sync(query, candidates_blob, model):
        return asyncio.run(rank_handler(query, candidates_blob, model))

    def stats_sync():
        return asyncio.run(stats_handler())

    # ---------- layout ----------
    with gr.Blocks(
        title="Lightweight Embeddings",
        theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
        css=_CSS,
    ) as demo:
        gr.Markdown(_HEADER_MD, elem_classes=["lwe-header"])

        with gr.Tabs():
            # ------------------------------ Embeddings
            with gr.Tab("Embeddings"):
                with gr.Row():
                    with gr.Column(scale=3):
                        embed_input = gr.Textbox(
                            label="Input",
                            placeholder="Text or image URL — one per line for batches",
                            lines=4,
                        )
                        with gr.Row():
                            embed_model = gr.Dropdown(
                                choices=all_models,
                                value=text_models[0] if text_models else None,
                                label="Model",
                                scale=3,
                            )
                            embed_dim = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="Dimensions (Matryoshka)",
                                visible=False,
                                allow_custom_value=False,
                                scale=2,
                            )
                        embed_btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=2):
                        embed_output = gr.Code(
                            label="Response",
                            language="json",
                            interactive=False,
                            lines=14,
                        )

                embed_model.change(matryoshka_update, inputs=embed_model, outputs=embed_dim)
                embed_btn.click(
                    embed_sync,
                    inputs=[embed_input, embed_model, embed_dim],
                    outputs=embed_output,
                )

            # ------------------------------ Rank
            with gr.Tab("Rerank"):
                with gr.Row():
                    with gr.Column(scale=3):
                        rank_query = gr.Textbox(label="Query", placeholder="happy person")
                        rank_candidates = gr.Textbox(
                            label="Candidates (one per line)",
                            placeholder="happy dog\nsunny day\nThat is a very happy person",
                            lines=6,
                        )
                        rank_model = gr.Dropdown(
                            choices=text_models,
                            value=text_models[0] if text_models else None,
                            label="Model",
                        )
                        rank_btn = gr.Button("Rank", variant="primary")
                    with gr.Column(scale=2):
                        rank_output = gr.Code(
                            label="Ranked",
                            language="json",
                            interactive=False,
                            lines=14,
                        )
                rank_btn.click(
                    rank_sync,
                    inputs=[rank_query, rank_candidates, rank_model],
                    outputs=rank_output,
                )

            # ------------------------------ Stats
            with gr.Tab("Stats"):
                stats_btn = gr.Button("Refresh", variant="secondary")
                with gr.Row():
                    access_df = gr.Dataframe(
                        headers=["Model", "Total", "Daily", "Weekly", "Monthly", "Yearly"],
                        interactive=False,
                        wrap=True,
                        label="Requests",
                    )
                    tokens_df = gr.Dataframe(
                        headers=["Model", "Total", "Daily", "Weekly", "Monthly", "Yearly"],
                        interactive=False,
                        wrap=True,
                        label="Tokens",
                    )
                stats_btn.click(stats_sync, inputs=[], outputs=[access_df, tokens_df])

            # ------------------------------ API
            with gr.Tab("API"):
                gr.Markdown(_CURL_MD)

    return demo


def mount_demo(app, *, path: str = "/") -> None:
    """Mount the Gradio Blocks onto a FastAPI app at ``path``."""
    from gradio.routes import mount_gradio_app

    demo = build_demo(app)
    mount_gradio_app(app, demo, path=path)


def _err(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False, indent=2)


_CSS = """
.lwe-header h1 { margin-bottom: 4px; }
.lwe-header p { color: var(--body-text-color-subdued); }
footer { display: none !important; }
"""
