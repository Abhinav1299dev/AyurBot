"""Tests for the rule-based NER engine (fully offline)."""

from __future__ import annotations

import pytest

from src.knowledge_base import KnowledgeBase
from src.ner_engine import DOSAGE_METHOD, NEREngine, highlight_spans
from tests.conftest import minimal_kb_dict


def labels(entities):
    return [(e.text, e.label, e.canonical) for e in entities]


def assert_offsets(text, entities):
    for e in entities:
        assert text[e.start:e.end] == e.text, (e, text[e.start:e.end])


@pytest.mark.parametrize("herb", ["Tulsi", "Haridra", "Ashwagandha", "Brahmi", "Guduchi", "Neem"])
def test_all_required_herbs_detected(ner, herb):
    ents = ner.extract(f"Tell me about {herb} please")
    assert labels(ents) == [(herb, "HERB", herb)]


@pytest.mark.parametrize("dosha", ["Vata", "Pitta", "Kapha"])
def test_doshas_detected(ner, dosha):
    ents = ner.extract(f"What is {dosha}?")
    assert [(e.label, e.canonical) for e in ents] == [("DOSHA", dosha)]


@pytest.mark.parametrize("text", ["TULSI", "tulsi", "TuLsI"])
def test_case_insensitive(ner, text):
    ents = ner.extract(text)
    assert len(ents) == 1 and ents[0].canonical == "Tulsi" and ents[0].text == text


@pytest.mark.parametrize("surface,canonical", [
    ("Giloy", "Guduchi"), ("Amrita", "Guduchi"), ("Holy Basil", "Tulsi"), ("Turmeric", "Haridra"),
    ("Nimba", "Neem"), ("Shleshma", "Kapha"), ("Jvara", "Jwara"), ("Kushta", "Kushtha"),
])
def test_synonyms(ner, surface, canonical):
    ents = ner.extract(f"Is {surface} mentioned here?")
    assert [e.canonical for e in ents] == [canonical]
    assert "dictionary match" in ents[0].method.lower()


@pytest.mark.parametrize("botanical,herb", [
    ("Withania somnifera", "Ashwagandha"), ("Curcuma longa", "Haridra"),
    ("Ocimum sanctum", "Tulsi"), ("Bacopa monnieri", "Brahmi"), ("Azadirachta indica", "Neem"),
])
def test_botanical_names(ner, botanical, herb):
    ents = ner.extract(f"Facts about {botanical}.")
    assert labels(ents) == [(botanical, "BOTANICAL_NAME", herb)]


def test_botanical_name_extra_spaces(ner):
    ents = ner.extract("withania   somnifera")
    assert ents and ents[0].label == "BOTANICAL_NAME" and ents[0].end == len("withania   somnifera")


def test_longest_overlapping_match_wins():
    kb = KnowledgeBase.from_dict(minimal_kb_dict())
    ner = NEREngine(kb)
    ents = ner.extract("Holy Basil and basil are different")
    assert labels(ents) == [("Holy Basil", "HERB", "Holy Basil"), ("basil", "HERB", "Basil")]


def test_no_overlapping_spans(ner):
    text = "Tulsi Holy Basil Ocimum tenuiflorum Kapha Kasa cough 3 g"
    ents = ner.extract(text)
    for a, b in zip(ents, ents[1:]):
        assert a.end <= b.start


def test_repeated_term_gives_one_entity_per_occurrence(ner):
    text = "Neem, neem and NEEM"
    ents = ner.extract(text)
    assert [e.start for e in ents] == [0, 6, 15]
    assert len({(e.start, e.end) for e in ents}) == len(ents)


def test_offsets_are_exact(ner):
    text = "  Can Haridra reduce Kapha and treat Kushta in a dose of 5 g?"
    ents = ner.extract(text)
    assert_offsets(text, ents)
    assert [e.label for e in ents] == ["HERB", "DOSHA", "CONDITION_TERM", "DOSAGE_MENTION"]


def test_whole_word_only(ner):
    assert ner.extract("That was a pittance, not a neemrana trip") == []
    assert ner.extract("tulsis") == []


def test_unicode_devanagari_and_offsets(ner):
    text = "🌿 तुलसी और वात"
    ents = ner.extract(text)
    assert [e.canonical for e in ents] == ["Tulsi", "Vata"]
    assert_offsets(text, ents)


def test_devanagari_vowel_sign_is_not_a_boundary(ner):
    # "वाता" continues the word with a vowel sign; it must not match "वात".
    assert [e.canonical for e in ner.extract("वाता")] == []


def test_latin_diacritics_precomposed_and_decomposed(ner):
    for text in ("Tulasī and Vāta", "Tulasi\u0304 and Va\u0304ta"):
        ents = ner.extract(text)
        assert [e.canonical for e in ents] == ["Tulsi", "Vata"]
        assert_offsets(text, ents)


@pytest.mark.parametrize("text,expected", [
    ("take 5 g daily", "5 g"),
    ("about 500mg", "500mg"),
    ("2-3 grams", "2-3 grams"),
    ("1/2 tsp", "1/2 tsp"),
    ("half a teaspoon", "half a teaspoon"),
    ("10 ml", "10 ml"),
    ("2 capsules", "2 capsules"),
    ("1 tola", "1 tola"),
    ("twice a day", "twice a day"),
    ("3 times per day", "3 times per day"),
])
def test_dosage_mentions(ner, text, expected):
    ents = [e for e in ner.extract(text) if e.label == "DOSAGE_MENTION"]
    assert [e.text for e in ents] == [expected]
    assert ents[0].method == DOSAGE_METHOD
    assert "does not judge" in ents[0].evidence


def test_numbers_without_units_are_not_dosages(ner):
    assert [e for e in ner.extract("There are 3 doshas and 7 dhatus") if e.label == "DOSAGE_MENTION"] == []


@pytest.mark.parametrize("text", ["", "   ", None, 42])
def test_empty_or_invalid_input(ner, text):
    assert ner.extract(text) == []


def test_unknown_query(ner):
    assert ner.extract("What is the weather in Pune today?") == []


def test_english_gloss_is_labelled_approximate(ner):
    ents = ner.extract("herbs for cough")
    assert ents[0].canonical == "Kasa"
    assert "approximate" in ents[0].method


def test_entities_have_all_fields(ner):
    ents = ner.extract("Brahmi is Medhya and relates to Agni")
    assert [e.label for e in ents] == ["HERB", "TRADITIONAL_PROPERTY", "AYURVEDIC_CONCEPT"]
    for e in ents:
        d = e.to_dict()
        assert set(d) == {"text", "canonical", "entry_id", "label", "start", "end", "method", "evidence"}
        assert d["evidence"]
        assert "%" not in d["method"]  # no fake confidence percentages


def test_results_are_stable(ner):
    text = "Guduchi, Neem and Pitta with 2 g"
    assert ner.extract(text) == ner.extract(text)


def test_highlight_spans_reconstruct_text(ner):
    text = "Does Tulsi pacify Kapha?"
    spans = highlight_spans(text, ner.extract(text))
    assert "".join(s for s, _ in spans) == text
    assert ("Tulsi", "HERB") in spans and ("Kapha", "DOSHA") in spans


def test_new_herb_from_json_is_recognised_without_code_changes():
    data = minimal_kb_dict()
    data["entries"].append({
        "id": "amalaki", "type": "herb", "canonical_name": "Amalaki", "summary": "Demo.",
        "synonyms": ["Amla"], "botanical_name": "Phyllanthus emblica",
        "traditional_uses": [], "traditional_properties": {}, "dosha_associations": {"pacifies": [], "may_aggravate": []},
    })
    ner = NEREngine(KnowledgeBase.from_dict(data))
    assert [(e.canonical, e.label) for e in ner.extract("Amla or Phyllanthus emblica")] == [
        ("Amalaki", "HERB"), ("Amalaki", "BOTANICAL_NAME")]
