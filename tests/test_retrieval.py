"""Retrieval tests. Semantic tests use a fake encoder, so no download is needed."""

from __future__ import annotations

import os
import sys

import pytest

from src.retrieval_engine import (
    LEXICAL,
    SEMANTIC,
    LexicalRetriever,
    RetrievalEngine,
    SemanticRetriever,
    default_model_factory,
)


# ----------------------------------------------------------------- lexical
def test_lexical_finds_herb_by_english_name(kb):
    results = LexicalRetriever(kb.documents).search("holy basil", top_k=3)
    assert results and results[0].entry_id == "tulsi"
    assert results[0].method == LEXICAL
    assert 0 <= results[0].score <= 1


def test_lexical_finds_herbs_for_condition_gloss(kb):
    ids = [r.entry_id for r in LexicalRetriever(kb.documents).search("cough", top_k=5)]
    assert "kasa" in ids and "tulsi" in ids


def test_lexical_respects_top_k(kb):
    assert len(LexicalRetriever(kb.documents).search("kapha pitta vata herb", top_k=2)) <= 2


def test_lexical_threshold_filters_unrelated(kb):
    assert LexicalRetriever(kb.documents).search("weather forecast football", top_k=3, min_score=0.25) == []


@pytest.mark.parametrize("query", ["", "   ", "the and of"])
def test_lexical_empty_or_stopword_query(kb, query):
    assert LexicalRetriever(kb.documents).search(query) == []


def test_lexical_empty_corpus():
    assert LexicalRetriever([]).search("tulsi") == []


def test_lexical_score_meaning_is_not_medical(kb):
    r = LexicalRetriever(kb.documents).search("neem")[0]
    assert "not a measure of medical accuracy" in r.score_meaning


# ----------------------------------------------------------------- semantic
def test_semantic_embeds_documents_once(kb, fake_factory, fake_encoder):
    sem = SemanticRetriever(kb.documents, fake_factory, "fake")
    assert sem.initialize() and sem.ready
    calls_after_init = fake_encoder.encode_calls
    for _ in range(5):
        sem.search("tulsi cough")
    assert fake_factory.calls["n"] == 1
    assert fake_encoder.encode_calls == calls_after_init + 5  # one encode per query, docs not re-encoded
    assert sem.initialize() is True and fake_factory.calls["n"] == 1  # initialize is idempotent


def test_semantic_ranks_relevant_entry_first(kb, fake_factory):
    sem = SemanticRetriever(kb.documents, fake_factory, "fake")
    sem.initialize()
    results = sem.search("Bacopa monnieri water hyssop", top_k=3)
    assert results[0].entry_id == "brahmi" and results[0].method == SEMANTIC


def test_semantic_threshold(kb, fake_factory):
    sem = SemanticRetriever(kb.documents, fake_factory, "fake")
    sem.initialize()
    assert sem.search("zebra xylophone quantum", top_k=3, min_score=0.3) == []


def test_semantic_empty_documents(fake_factory):
    sem = SemanticRetriever([], fake_factory, "fake")
    assert sem.initialize() is False
    assert sem.status.state == "failed"


# ----------------------------------------------------------------- fallback
def test_model_failure_falls_back_to_lexical_without_retrying(kb, failing_factory):
    sem = SemanticRetriever(kb.documents, failing_factory, "broken")
    assert sem.initialize() is False
    assert sem.status.state == "failed" and "simulated" in sem.status.detail
    engine = RetrievalEngine(LexicalRetriever(kb.documents), sem)
    assert engine.mode == LEXICAL
    for _ in range(5):
        out = engine.search("holy basil")
        assert out.mode == LEXICAL and out.results[0].entry_id == "tulsi"
    sem.initialize()
    assert failing_factory.calls["n"] == 1  # the failed model is never reloaded


def test_disabled_semantic_uses_lexical(kb, fake_factory):
    sem = SemanticRetriever(kb.documents, fake_factory, "fake")
    sem.disable("disabled in test")
    engine = RetrievalEngine(LexicalRetriever(kb.documents), sem)
    assert engine.mode == LEXICAL and sem.initialize() is False
    assert fake_factory.calls["n"] == 0


def test_query_time_error_falls_back_and_eventually_disables(kb, fake_factory, fake_encoder):
    sem = SemanticRetriever(kb.documents, fake_factory, "fake")
    sem.initialize()
    engine = RetrievalEngine(LexicalRetriever(kb.documents), sem)

    def boom(*args, **kwargs):
        raise RuntimeError("encode exploded")

    fake_encoder.encode = boom
    for i in range(SemanticRetriever.MAX_QUERY_ERRORS):
        out = engine.search("neem")
        assert out.mode == LEXICAL and "Semantic search failed" in out.note
    assert sem.status.state == "failed" and engine.mode == LEXICAL


def test_engine_empty_query(kb):
    out = RetrievalEngine(LexicalRetriever(kb.documents)).search("  ")
    assert out.results == [] and out.note == "Empty query."


def test_default_factory_reports_missing_library(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # makes the import fail
    factory = default_model_factory("any-model", None, "cpu", offline=False)
    with pytest.raises(RuntimeError, match="could not be imported"):
        factory()


# ----------------------------------------------------------------- real model (optional)
@pytest.mark.model
@pytest.mark.skipif(os.environ.get("AYURBOT_RUN_MODEL_TESTS") != "1",
                    reason="set AYURBOT_RUN_MODEL_TESTS=1 to run tests that load the real model")
def test_real_model_semantic_retrieval(kb):
    factory = default_model_factory("sentence-transformers/all-MiniLM-L6-v2", None, "cpu", offline=False)
    sem = SemanticRetriever(kb.documents, factory, "all-MiniLM-L6-v2")
    assert sem.initialize(), sem.status.detail
    results = sem.search("Which herb is known as holy basil?", top_k=3)
    assert results and results[0].entry_id == "tulsi"
    assert sem.search("Which herb is used for memory?", top_k=3)[0].entry_id in {"brahmi", "medhya", "smriti_daurbalya"}
