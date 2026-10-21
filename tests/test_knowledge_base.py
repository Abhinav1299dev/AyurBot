"""Knowledge-base loading and validation tests."""

from __future__ import annotations

import json

import pytest

from src.knowledge_base import KnowledgeBase, KnowledgeBaseError
from tests.conftest import minimal_kb_dict

REQUIRED = ["tulsi", "haridra", "ashwagandha", "brahmi", "guduchi", "neem", "vata", "pitta", "kapha"]


def test_bundled_kb_is_valid_and_complete(kb):
    for entry_id in REQUIRED:
        assert entry_id in kb.by_id, entry_id
    assert kb.warnings == []
    counts = kb.counts()
    assert counts["herb"] >= 6 and counts["dosha"] == 3
    assert counts["condition_term"] > 0 and counts["traditional_property"] > 0 and counts["concept"] > 0


def test_every_herb_has_required_fields(kb):
    for herb in kb.entries_of_type("herb"):
        for key in ("botanical_name", "english_name", "synonyms", "traditional_uses", "traditional_properties",
                    "dosha_associations", "source", "evidence_note", "safety_notes", "data_status"):
            assert herb.get(key), f"{herb.id} missing {key}"


def test_no_fabricated_references(kb):
    for entry in kb.entries:
        source = entry.get("source") or {}
        assert source.get("reference") is None, f"{entry.id} has a reference; verify it before adding"
        assert entry.get("data_status") == "demonstration"


def test_condition_terms_have_mapping_notes(kb):
    for term in kb.entries_of_type("condition_term"):
        assert "not" in term.get("mapping_note", "").lower(), term.id


def test_documents_built_once_per_entry(kb):
    docs = kb.documents
    assert len(docs) == len(kb.entries)
    tulsi = dict(docs)["tulsi"]
    assert "Holy Basil" in tulsi and "Kasa" in tulsi and "cough" in tulsi


def test_missing_file(tmp_path):
    with pytest.raises(KnowledgeBaseError, match="not found"):
        KnowledgeBase.from_file(tmp_path / "missing.json")


def test_empty_file(tmp_path):
    p = tmp_path / "kb.json"
    p.write_text("   ", encoding="utf-8")
    with pytest.raises(KnowledgeBaseError, match="empty"):
        KnowledgeBase.from_file(p)


def test_invalid_json_reports_line(tmp_path):
    p = tmp_path / "kb.json"
    p.write_text('{"entries": [\n {"id": "x",}\n]}', encoding="utf-8")
    with pytest.raises(KnowledgeBaseError, match="line 2"):
        KnowledgeBase.from_file(p)


def test_utf8_bom_is_accepted(tmp_path):
    p = tmp_path / "kb.json"
    p.write_text(json.dumps(minimal_kb_dict()), encoding="utf-8-sig")
    assert len(KnowledgeBase.from_file(p).entries) == 3


@pytest.mark.parametrize("data,message", [
    ({"entries": []}, "empty"),
    ({}, "no \"entries\""),
    ({"entries": {}}, "must be a JSON list"),
    ("text", "JSON object"),
])
def test_structural_errors(data, message):
    with pytest.raises(KnowledgeBaseError, match=message):
        KnowledgeBase.from_dict(data)


def test_all_entry_problems_reported_together():
    data = minimal_kb_dict()
    data["entries"].append({"id": "Bad ID", "type": "plant", "canonical_name": "", "summary": ""})
    data["entries"].append({"id": "basil", "type": "herb", "canonical_name": "Dup", "summary": "x"})
    data["entries"].append({"id": "syn", "type": "dosha", "canonical_name": "S", "summary": "x", "synonyms": "oops"})
    with pytest.raises(KnowledgeBaseError) as info:
        KnowledgeBase.from_dict(data)
    text = str(info.value)
    assert "lower-case" in text and "\"type\" must be one of" in text and "canonical_name" in text
    assert "duplicate id 'basil'" in text and "list of strings" in text
    assert len(info.value.problems) >= 5


def test_invalid_herb_structures():
    data = minimal_kb_dict()
    data["entries"][0]["traditional_uses"] = [{"term": "x"}]
    data["entries"][0]["dosha_associations"] = {"pacifies": "vata"}
    with pytest.raises(KnowledgeBaseError) as info:
        KnowledgeBase.from_dict(data)
    assert "traditional_uses" in str(info.value) and "dosha_associations" in str(info.value)


def test_dangling_references_become_warnings():
    data = minimal_kb_dict()
    data["entries"][0]["traditional_uses"] = [{"term_id": "nowhere", "description": "x"}]
    data["entries"][0]["dosha_associations"] = {"pacifies": ["vata", "ghost"], "may_aggravate": []}
    kb = KnowledgeBase.from_dict(data)
    assert any("nowhere" in w for w in kb.warnings) and any("ghost" in w for w in kb.warnings)


def test_lookup_helpers(kb):
    assert [h.id for h in kb.herbs_for_term("kasa")] == ["tulsi"]
    assert "neem" in [h.id for h in kb.herbs_aggravating("vata")]
    assert kb.find_by_name("giloy").id == "guduchi"
    assert [h.id for h in kb.herbs_with_property(kb.get("medhya"))] == ["brahmi"]
