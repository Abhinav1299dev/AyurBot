"""Retrieval over knowledge-base entries.

Two independent retrievers are provided:

* :class:`SemanticRetriever` uses a Sentence Transformers model. The model is
  loaded once, document embeddings are computed once, and both are reused for
  every query. If loading fails, the failure is recorded and *never retried*
  per request.
* :class:`LexicalRetriever` is a pure-Python BM25 index with no dependency on
  PyTorch or the transformer model. It is the fallback.

:class:`RetrievalEngine` chooses between them and always reports which one
answered. Scores are relevance indicators only. They say how closely the
query's wording or meaning matches an entry, not whether the content is
medically accurate.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .text_utils import tokenize

log = logging.getLogger(__name__)

SEMANTIC = "semantic"
LEXICAL = "lexical"

SEMANTIC_SCORE_MEANING = (
    "Cosine similarity between query and entry embeddings (range -1 to 1). "
    "Higher means closer in meaning. It is not a measure of medical accuracy."
)
LEXICAL_SCORE_MEANING = (
    "Share of the query's informative words (weighted by rarity) found in the entry, 0 to 1. "
    "It is not a measure of medical accuracy."
)


@dataclass(frozen=True)
class RetrievalResult:
    entry_id: str
    score: float
    method: str  # SEMANTIC or LEXICAL
    score_meaning: str


@dataclass
class RetrievalOutput:
    results: list[RetrievalResult]
    mode: str
    note: str = ""


# ====================================================================== lexical
class LexicalRetriever:
    """Okapi BM25 ranking with an interpretable coverage score for filtering."""

    def __init__(self, documents: Sequence[tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        self.ids = [doc_id for doc_id, _ in documents]
        self.k1, self.b = k1, b
        self._tfs = [Counter(tokenize(text)) for _, text in documents]
        self._lengths = [sum(tf.values()) for tf in self._tfs]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        n = len(self._tfs)
        df: Counter = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._max_idf = math.log(1 + (n + 0.5) / 0.5) if n else 1.0

    def search(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[RetrievalResult]:
        q_terms = list(dict.fromkeys(tokenize(query)))
        if not q_terms or not self._tfs:
            return []
        q_weight = sum(self._idf.get(t, self._max_idf) for t in q_terms)
        scored: list[tuple[float, float, int]] = []
        for i, tf in enumerate(self._tfs):
            bm25 = 0.0
            covered = 0.0
            norm = self.k1 * (1 - self.b + self.b * self._lengths[i] / (self._avg_len or 1.0))
            for term in q_terms:
                freq = tf.get(term, 0)
                if freq:
                    idf = self._idf[term]
                    bm25 += idf * freq * (self.k1 + 1) / (freq + norm)
                    covered += idf
            if bm25 > 0:
                scored.append((bm25, covered / q_weight, i))
        scored.sort(key=lambda s: (-s[0], self.ids[s[2]]))
        results = [
            RetrievalResult(self.ids[i], round(coverage, 4), LEXICAL, LEXICAL_SCORE_MEANING)
            for _, coverage, i in scored
            if coverage >= min_score
        ]
        return results[:top_k]


# ===================================================================== semantic
ModelFactory = Callable[[], tuple[Any, str]]


def default_model_factory(model_name: str, cache_dir: Path | None, device: str, offline: bool) -> ModelFactory:
    """Return a factory that loads a SentenceTransformer, preferring the local cache."""

    def factory() -> tuple[Any, str]:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            from sentence_transformers import SentenceTransformer  # heavy import, done once
        except Exception as exc:  # ImportError, or a broken torch install
            raise RuntimeError(
                f"sentence-transformers/PyTorch could not be imported ({exc.__class__.__name__}: {exc})."
            ) from exc

        resolved_device = device
        if device == "auto":
            try:
                import torch

                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                resolved_device = "cpu"
        kwargs: dict[str, Any] = {"device": resolved_device}
        if cache_dir:
            kwargs["cache_folder"] = str(cache_dir)
        try:
            return SentenceTransformer(model_name, local_files_only=True, **kwargs), "loaded from local cache"
        except Exception as local_exc:
            if offline:
                raise RuntimeError(
                    f"Offline mode is on and '{model_name}' is not in the local cache ({local_exc})."
                ) from local_exc
            log.info("Model not in local cache; downloading '%s' (first run only).", model_name)
        return SentenceTransformer(model_name, **kwargs), "downloaded and cached"

    return factory


@dataclass
class SemanticStatus:
    state: str = "not_started"  # not_started | loading | ready | failed | disabled
    detail: str = ""
    load_seconds: float = 0.0
    errors_at_query_time: int = 0


class SemanticRetriever:
    """Sentence-embedding retriever. Call :meth:`initialize` once at startup."""

    MAX_QUERY_ERRORS = 3

    def __init__(self, documents: Sequence[tuple[str, str]], model_factory: ModelFactory,
                 model_name: str = "model"):
        self.ids = [doc_id for doc_id, _ in documents]
        self._texts = [text for _, text in documents]
        self._factory = model_factory
        self.model_name = model_name
        self.status = SemanticStatus()
        self._model: Any = None
        self._matrix: Any = None  # numpy array (n_docs, dim), L2-normalised
        self._init_lock = threading.Lock()
        self._encode_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.status.state == "ready"

    def disable(self, reason: str) -> None:
        self.status = SemanticStatus(state="disabled", detail=reason)

    def initialize(self) -> bool:
        """Load the model and embed all documents. Runs at most once."""
        with self._init_lock:
            if self.status.state != "not_started":
                return self.ready
            self.status.state = "loading"
            started = time.perf_counter()
            try:
                import numpy as np

                if not self._texts:
                    raise RuntimeError("there are no documents to embed")
                model, how = self._factory()
                matrix = model.encode(
                    self._texts, batch_size=32, convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=False,
                )
                matrix = np.asarray(matrix, dtype="float32")
                if matrix.ndim != 2 or matrix.shape[0] != len(self._texts):
                    raise RuntimeError(f"unexpected embedding shape {getattr(matrix, 'shape', None)}")
                self._model, self._matrix = model, matrix
                elapsed = time.perf_counter() - started
                self.status = SemanticStatus(
                    "ready", f"{self.model_name} {how}; {matrix.shape[0]} entries embedded ({matrix.shape[1]} dims).",
                    round(elapsed, 2),
                )
                log.info("Semantic retrieval ready in %.1fs (%s).", elapsed, how)
            except Exception as exc:
                self._model = self._matrix = None
                self.status = SemanticStatus(
                    "failed", f"{exc.__class__.__name__}: {exc}", round(time.perf_counter() - started, 2)
                )
                log.warning("Semantic retrieval unavailable, using lexical fallback: %s", self.status.detail)
            return self.ready

    def search(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[RetrievalResult]:
        if not self.ready:
            raise RuntimeError("semantic retriever is not ready")
        if not query or not query.strip():
            return []
        import numpy as np

        with self._encode_lock:  # keep model use single-threaded and predictable
            vec = self._model.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )
        vec = np.asarray(vec, dtype="float32").reshape(-1)
        scores = self._matrix @ vec
        order = np.argsort(-scores)[: max(top_k, 0)]
        return [
            RetrievalResult(self.ids[i], round(float(scores[i]), 4), SEMANTIC, SEMANTIC_SCORE_MEANING)
            for i in order
            if float(scores[i]) >= min_score
        ]

    def record_query_error(self, exc: Exception) -> None:
        self.status.errors_at_query_time += 1
        if self.status.errors_at_query_time >= self.MAX_QUERY_ERRORS:
            self.status.state = "failed"
            self.status.detail = f"disabled after repeated query errors (last: {exc.__class__.__name__}: {exc})"
            self._model = self._matrix = None


# ======================================================================= engine
@dataclass
class RetrievalEngine:
    lexical: LexicalRetriever
    semantic: SemanticRetriever | None = None
    top_k: int = 3
    min_semantic_score: float = 0.30
    min_lexical_score: float = 0.25
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def mode(self) -> str:
        return SEMANTIC if self.semantic is not None and self.semantic.ready else LEXICAL

    def search(self, query: str, top_k: int | None = None) -> RetrievalOutput:
        k = top_k or self.top_k
        if not isinstance(query, str) or not query.strip():
            return RetrievalOutput([], self.mode, "Empty query.")
        if self.mode == SEMANTIC:
            try:
                results = self.semantic.search(query, k, self.min_semantic_score)
                return RetrievalOutput(results, SEMANTIC)
            except Exception as exc:  # fall through to lexical for this query
                log.exception("Semantic search failed; using lexical fallback for this query.")
                with self._lock:
                    self.semantic.record_query_error(exc)
                results = self.lexical.search(query, k, self.min_lexical_score)
                return RetrievalOutput(results, LEXICAL, f"Semantic search failed for this query ({exc}).")
        return RetrievalOutput(self.lexical.search(query, k, self.min_lexical_score), LEXICAL)
