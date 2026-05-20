"""FastAPI server for Constitutional BioGuard bio-safety classifier.

Usage:
    python scripts/serve.py [--model-dir PATH] [--host HOST] [--port PORT]

Environment variables:
    BIOGUARD_MODEL_DIR  Override default model directory
    BIOGUARD_PORT       Override default port (8000)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from constitutional_bioguard.config import MODELS_DIR

# ── Pydantic models ──────────────────────────────────────────────────────────

from pydantic import BaseModel, model_validator


class ClassifyRequest(BaseModel):
    """Single-text classification request.

    Accepts either:
    - ``{"text": "..."}`` — raw combined text
    - ``{"query": "...", "response": "..."}`` — query/response pair (joined with [SEP])
    """

    text: Optional[str] = None
    query: Optional[str] = None
    response: Optional[str] = None

    @model_validator(mode="after")
    def check_at_least_one(self) -> "ClassifyRequest":
        if self.text is None and self.query is None:
            raise ValueError("Provide either 'text' or 'query' (+ optional 'response')")
        return self

    def to_pair(self) -> tuple[str, str] | None:
        """Return (query, response) if both are available, else None."""
        if self.query is not None:
            return (self.query or "", self.response or "")
        return None

    def to_text(self) -> str:
        if self.text is not None:
            return self.text
        q = self.query or ""
        r = self.response or ""
        if r:
            return f"{q} [SEP] {r}"
        return q


class ClassifyResponse(BaseModel):
    label: str  # "SAFE" or "UNSAFE"
    p_unsafe: float
    latency_ms: float


class BatchClassifyRequest(BaseModel):
    texts: list[str]


class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


# ── Global model state (populated at startup) ────────────────────────────────

_STATE: dict[str, Any] = {}


def _get_model_dir(cli_model_dir: Optional[Path] = None) -> Path:
    env_dir = os.environ.get("BIOGUARD_MODEL_DIR")
    if cli_model_dir is not None:
        return cli_model_dir
    if env_dir:
        return Path(env_dir)
    return MODELS_DIR / "deberta_bioguard_v1"


def _build_app(model_dir: Optional[Path] = None):
    """Build and return the FastAPI application, optionally with a custom model_dir."""
    import torch
    from fastapi import FastAPI, Query

    from constitutional_bioguard.evaluation.evaluate_classifier import (
        load_model_and_tokenizer,
        predict_batch,
    )
    from constitutional_bioguard.preprocessing import normalize_text

    resolved_model_dir = _get_model_dir(model_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load model once at startup, clean up on shutdown."""
        print(f"Loading model from {resolved_model_dir} …")
        model, tokenizer = load_model_and_tokenizer(resolved_model_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        _STATE["model"] = model
        _STATE["tokenizer"] = tokenizer
        _STATE["device"] = device
        _STATE["model_dir"] = str(resolved_model_dir)
        print(f"Model loaded on {device}. Ready.")
        yield
        _STATE.clear()

    app = FastAPI(
        title="Constitutional BioGuard",
        description="Bio-safety content classifier endpoint",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── Helper ────────────────────────────────────────────────────────────────

    def _classify(
        texts: list[str] | None = None,
        do_normalize: bool = True,
        *,
        queries: list[str] | None = None,
        responses: list[str] | None = None,
    ) -> list[ClassifyResponse]:
        if do_normalize:
            if queries is not None:
                queries = [normalize_text(q) for q in queries]
                responses = [normalize_text(r) for r in responses]
            elif texts is not None:
                texts = [normalize_text(t) for t in texts]

        results = predict_batch(
            texts=texts,
            model=_STATE["model"],
            tokenizer=_STATE["tokenizer"],
            device=_STATE["device"],
            normalize=False,
            queries=queries,
            responses=responses,
        )
        responses = []
        for _label, _conf, p_unsafe in results:
            label_str = "UNSAFE" if _label == 1 else "SAFE"
            responses.append(
                ClassifyResponse(label=label_str, p_unsafe=p_unsafe, latency_ms=0.0)
            )
        return responses

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model="deberta_bioguard_v1",
            device=_STATE.get("device", "unknown"),
        )

    @app.post("/classify", response_model=ClassifyResponse)
    async def classify(
        req: ClassifyRequest,
        normalize: bool = Query(default=True, description="Apply text normalization"),
    ) -> ClassifyResponse:
        t0 = time.perf_counter()
        pair = req.to_pair()
        if pair:
            results = _classify(
                do_normalize=normalize, queries=[pair[0]], responses=[pair[1]],
            )
        else:
            results = _classify([req.to_text()], do_normalize=normalize)
        latency_ms = (time.perf_counter() - t0) * 1000
        result = results[0]
        return ClassifyResponse(
            label=result.label,
            p_unsafe=result.p_unsafe,
            latency_ms=round(latency_ms, 2),
        )

    @app.post("/classify/batch", response_model=BatchClassifyResponse)
    async def classify_batch(
        req: BatchClassifyRequest,
        normalize: bool = Query(default=True, description="Apply text normalization"),
    ) -> BatchClassifyResponse:
        t0 = time.perf_counter()
        results = _classify(req.texts, do_normalize=normalize)
        total_ms = (time.perf_counter() - t0) * 1000
        per_ms = total_ms / max(len(req.texts), 1)
        enriched = [
            ClassifyResponse(label=r.label, p_unsafe=r.p_unsafe, latency_ms=round(per_ms, 2))
            for r in results
        ]
        return BatchClassifyResponse(results=enriched)

    return app


# ── CLI entry point ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    default_port = int(os.environ.get("BIOGUARD_PORT", "8000"))
    parser = argparse.ArgumentParser(
        description="Serve Constitutional BioGuard via FastAPI"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to trained model directory (overrides BIOGUARD_MODEL_DIR env var)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help=f"Bind port (default: {default_port}, overrides BIOGUARD_PORT)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    app = _build_app(model_dir=args.model_dir)
    uvicorn.run(app, host=args.host, port=args.port)
