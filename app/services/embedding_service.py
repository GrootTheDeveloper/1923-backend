"""Embedding providers behind one interface so the pipeline is not vendor-locked.

- GeminiEmbeddingProvider: real semantic embeddings via the Gemini embedContent
  REST API (same pattern as gemini_extraction). Uses matching task types for
  documents vs. queries and truncates to EMBEDDING_DIM (Matryoshka).
- HashingEmbeddingProvider: dependency-free deterministic fallback for offline
  dev/tests. Low quality but keeps retrieval functional without a network call.

Select via EMBEDDING_PROVIDER=gemini|hashing. Any failure returns None so callers
degrade to non-vector retrieval instead of crashing.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional, Protocol

from app.config import (
    EMBEDDING_DIM,
    EMBEDDING_PROVIDER,
    GEMINI_API_KEY,
    GEMINI_EMBED_MODEL,
    GEMINI_EMBED_URL,
)

Vector = List[float]


class EmbeddingProvider(Protocol):
    dim: int

    async def embed_query(self, text: str) -> Optional[Vector]:
        ...

    async def embed_documents(self, texts: List[str]) -> List[Optional[Vector]]:
        ...


def _normalize(values: List[float]) -> Optional[Vector]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0:
        return None
    return [v / norm for v in values]


class GeminiEmbeddingProvider:
    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def _embed_sync(self, text: str, task_type: str) -> Optional[Vector]:
        url = GEMINI_EMBED_URL.format(model=urllib.parse.quote(GEMINI_EMBED_MODEL, safe=""))
        url = f"{url}?key={urllib.parse.quote(GEMINI_API_KEY)}"
        payload = {
            "model": f"models/{GEMINI_EMBED_MODEL}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
            "outputDimensionality": self.dim,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        values = data["embedding"]["values"]
        # Truncated Matryoshka dims are not unit-norm; normalize for cosine.
        return _normalize(values)

    async def _embed_one(self, text: str, task_type: str) -> Optional[Vector]:
        if not GEMINI_API_KEY or not text.strip():
            return None
        try:
            return await asyncio.to_thread(self._embed_sync, text, task_type)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            print(f"[embedding] gemini {task_type} failed ({exc}).")
            return None

    async def embed_query(self, text: str) -> Optional[Vector]:
        return await self._embed_one(text, "RETRIEVAL_QUERY")

    async def embed_documents(self, texts: List[str]) -> List[Optional[Vector]]:
        return [await self._embed_one(text, "RETRIEVAL_DOCUMENT") for text in texts]


class HashingEmbeddingProvider:
    """Deterministic hashed bag-of-words vector. No network, no heavy deps."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def _embed(self, text: str) -> Optional[Vector]:
        if not text.strip():
            return None
        vector = [0.0] * self.dim
        for token in re.findall(r"[a-zA-Z0-9+#.]{2,}", text.lower()):
            bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vector[bucket] += 1.0
        return _normalize(vector)

    async def embed_query(self, text: str) -> Optional[Vector]:
        return self._embed(text)

    async def embed_documents(self, texts: List[str]) -> List[Optional[Vector]]:
        return [self._embed(text) for text in texts]


_provider: Optional[EmbeddingProvider] = None


def get_embedding_provider() -> EmbeddingProvider:
    global _provider
    if _provider is None:
        if EMBEDDING_PROVIDER == "hashing":
            _provider = HashingEmbeddingProvider()
        else:
            _provider = GeminiEmbeddingProvider()
    return _provider
