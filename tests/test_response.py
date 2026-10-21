"""Tests for grounded response generation and safety behaviour."""

from __future__ import annotations

import pytest

from src.knowledge_base import KnowledgeBase
from src.ner_engine import NEREngine
from src.response_engine import ENTITY_COLUMNS, EVIDENCE_COLUMNS, SHORT_DISCLAIMER, ResponseEngine
from src.retrieval_engine import LexicalRetriever, RetrievalEngine
from tests.conftest import minimal_kb_dict


def test_known_herb_answer_is_grounded(lexical_engine, kb):
    r = lexical_engine.respond("What is Tulsi?")
    assert r.status == "ok"
    assert "Ocimum tenuiflorum" in r.answer
    assert kb.get("tulsi").summary in r.answer
    assert "Traditional uses" in r.answer and "not clinical evidence" in r.answer
    assert SHORT_DISCLAIMER in r.answer


def test_relationship_pacifies(lexical_engine):
    r = lexical_engine.respond("Does Haridra pacify Kapha?")
    assert "**Haridra** is traditionally described as pacifying **Kapha**." in r.answer


def test_relationship_aggravates(lexical_engine):
    r = lexical_engine.respond("Neem and Vata")
    assert "possibly aggravating **Vata**" in r.answer


def test_relationship_missing_is_stated_not_invented(lexical_engine):
    r = lexical_engine.respond("Does Haridra affect Vata?")
    assert "does not record an effect of **Haridra** on **Vata**" in r.answer


def test_condition_query_lists_herbs(lexical_engine):
    r = lexical_engine.respond("Which herbs are described for cough?")
    assert "**Kasa**: Tulsi" in r.answer
    assert "approximate" in r.answer.lower()


def test_cure_language_gets_caveat(lexical_engine):
    r = lexical_engine.respond("Can Neem cure Kushtha?")
    assert "cannot confirm that any herb treats, cures or prevents" in r.answer


def test_dosage_mention_is_not_endorsed(lexical_engine):
    r = lexical_engine.respond("Is 5 g of Haridra good?")
    assert "'5 g'" in r.answer and "does not recommend doses" in r.answer
    assert any(e.label == "DOSAGE_MENTION" for e in r.entities)


def test_dose_question_without_number(lexical_engine):
    r = lexical_engine.respond("What is the dosage of Ashwagandha?")
    assert "does not provide dosage recommendations" in r.answer


def test_personal_medical_question(lexical_engine):
    r = lexical_engine.respond("I have a fever, should I take Guduchi?")
    assert "can't give personal medical advice" in r.answer


def test_emergency_language(lexical_engine):
    r = lexical_engine.respond("I swallowed neem oil and feel sick")
    assert "emergency" in r.answer.lower() and "112" in r.answer


def test_safety_notes_present_for_herbs(lexical_engine, kb):
    r = lexical_engine.respond("Tell me about Neem")
    assert kb.get("neem").string_list("safety_notes")[0] in r.answer


def test_no_match(lexical_engine):
    r = lexical_engine.respond("What is the capital of France?")
    assert r.status == "no_match"
    assert "won't guess" in r.answer and r.evidence_rows == []
    assert SHORT_DISCLAIMER in r.answer


@pytest.mark.parametrize("query", ["", "   ", None, "\x00\x01"])
def test_empty_input(lexical_engine, query):
    r = lexical_engine.respond(query)
    assert r.status == "empty" and "Please type a question" in r.answer


def test_too_long_input(kb, ner):
    engine = ResponseEngine(kb, ner, RetrievalEngine(LexicalRetriever(kb.documents)), max_query_chars=50)
    r = engine.respond("Tulsi " * 30)
    assert r.status == "too_long" and "50 characters" in r.answer


def test_tables_have_fixed_columns(lexical_engine):
    r = lexical_engine.respond("Brahmi and Pitta 2 g")
    assert list(r.entity_table().columns) == ENTITY_COLUMNS
    assert list(r.evidence_table().columns) == EVIDENCE_COLUMNS
    assert len(r.entity_table()) == 3
    empty = lexical_engine.respond("hello there")
    assert list(empty.entity_table().columns) == ENTITY_COLUMNS and empty.entity_table().empty


def test_evidence_table_explains_scores(lexical_engine):
    r = lexical_engine.respond("water hyssop")
    row = r.evidence_rows[0]
    assert row["Entry"] == "Brahmi"
    assert "not a measure of medical accuracy" in row["Score meaning"]
    assert row["Data status"] == "demonstration"


def test_highlight_covers_full_query(lexical_engine):
    q = "Guduchi for Jwara"
    r = lexical_engine.respond(q)
    assert "".join(t for t, _ in r.highlighted) == q


def test_semantic_engine_reports_mode(semantic_engine):
    r = semantic_engine.respond("Bacopa monnieri")
    assert r.retrieval_mode == "semantic" and r.status == "ok"


def test_missing_evidence_and_safety_fields_are_explained():
    kb = KnowledgeBase.from_dict(minimal_kb_dict())
    engine = ResponseEngine(kb, NEREngine(kb), RetrievalEngine(LexicalRetriever(kb.documents)))
    r = engine.respond("Tell me about Basil")
    assert "No safety information is recorded" in r.answer
    assert "No evidence note is recorded" in r.answer
    assert "Not available in this knowledge base" in r.answer
    assert r.evidence_rows[0]["Source"] == "Not available"


def test_answers_are_fast(lexical_engine):
    r = lexical_engine.respond("Does Tulsi pacify Kapha and help Kasa at 2 g twice a day?")
    assert r.elapsed_ms < 500
