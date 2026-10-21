"""Shared fixtures. All fixtures work offline; no model is downloaded."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.knowledge_base import KnowledgeBase  # noqa: E402
from src.ner_engine import NEREngine  # noqa: E402
from src.retrieval_engine import LexicalRetriever, RetrievalEngine, SemanticRetriever  # noqa: E402
from src.response_engine import ResponseEngine  # noqa: E402
from src.text_utils import tokenize  # noqa: E402

KB_PATH = ROOT / "data" / "herbs.json"


class FakeEncoder:
    """Deterministic bag-of-words 'embedding' model with the SentenceTransformer encode API.

    Texts that share words get similar vectors, which is enough to test the
    semantic pipeline (ranking, thresholds, caching) without PyTorch.
    """

    dim = 4096

    def __init__(self):
        self.encode_calls = 0

    def encode(self, texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True,
               show_progress_bar=False):
        self.encode_calls += 1
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                out[row, h % self.dim] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out = out / norms
        return out


@pytest.fixture(scope="session")
def kb() -> KnowledgeBase:
    return KnowledgeBase.from_file(KB_PATH)


@pytest.fixture(scope="session")
def ner(kb) -> NEREngine:
    return NEREngine(kb)


@pytest.fixture()
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture()
def fake_factory(fake_encoder):
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return fake_encoder, "fake model for tests"

    factory.calls = calls
    return factory


@pytest.fixture()
def failing_factory():
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        raise OSError("simulated: could not download model (no internet)")

    factory.calls = calls
    return factory


@pytest.fixture()
def lexical_engine(kb, ner) -> ResponseEngine:
    retrieval = RetrievalEngine(lexical=LexicalRetriever(kb.documents), semantic=None)
    return ResponseEngine(kb, ner, retrieval)


@pytest.fixture()
def semantic_engine(kb, ner, fake_factory) -> ResponseEngine:
    sem = SemanticRetriever(kb.documents, fake_factory, "fake")
    assert sem.initialize()
    retrieval = RetrievalEngine(lexical=LexicalRetriever(kb.documents), semantic=sem, min_semantic_score=0.2)
    return ResponseEngine(kb, ner, retrieval)


def minimal_kb_dict(**overrides):
    """Small valid knowledge base for focused tests."""
    data = {
        "entries": [
            {"id": "basil", "type": "herb", "canonical_name": "Basil", "summary": "Basil summary.",
             "synonyms": [], "botanical_name": "Ocimum basilicum",
             "traditional_uses": [], "traditional_properties": {}, "dosha_associations": {"pacifies": [], "may_aggravate": []}},
            {"id": "holy_basil", "type": "herb", "canonical_name": "Holy Basil", "summary": "Holy basil summary.",
             "synonyms": ["Tulsi"], "botanical_name": "Ocimum tenuiflorum",
             "traditional_uses": [], "traditional_properties": {}, "dosha_associations": {"pacifies": [], "may_aggravate": []}},
            {"id": "vata", "type": "dosha", "canonical_name": "Vata", "summary": "Vata summary."},
        ]
    }
    data.update(overrides)
    return data
