"""Rule-based Named Entity Recognition for Ayurvedic text.

This is a *rule-based* NER engine: a dictionary built from the knowledge base
plus regular expressions for dosage mentions. It is not a trained statistical
model, and it does not produce probability scores. Every entity carries a
plain-language explanation of exactly which rule matched.

Matching rules
--------------
* Case-insensitive and Latin-diacritic-insensitive (``Tulasī`` == ``tulasi``).
* Whole words only (``pitta`` is not found inside ``pittance``).
* Multi-word terms tolerate extra spaces or hyphens (``Holy  Basil``).
* When matches overlap, the longest span wins; ties prefer the more specific
  category (botanical name > herb > dosha > ...).
* Offsets always refer to the original, unmodified query.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

from .knowledge_base import KnowledgeBase, LexiconTerm
from .text_utils import BOUNDARY_AFTER, BOUNDARY_BEFORE, fold, fold_with_map

LABEL_PRIORITY = {
    "BOTANICAL_NAME": 0,
    "HERB": 1,
    "DOSHA": 2,
    "CONDITION_TERM": 3,
    "TRADITIONAL_PROPERTY": 4,
    "AYURVEDIC_CONCEPT": 5,
    "DOSAGE_MENTION": 6,
}
KIND_PRIORITY = {
    "canonical name": 0,
    "botanical name": 0,
    "Sanskrit name": 1,
    "synonym": 2,
    "botanical synonym": 2,
    "Devanagari name": 3,
    "English name": 4,
    "approximate English gloss": 5,
}

DOSAGE_METHOD = "Dosage pattern match (regular expression)"

_NUMBER = r"(?:\d+(?:[.,]\d+)?(?:/\d+)?|half(?:\s+an?)?|one|two|three|four|five|a)"
_RANGE = rf"{_NUMBER}(?:\s*(?:-|–|to)\s*{_NUMBER})?"
_UNITS = (
    "milligrams|milligram|micrograms|microgram|grams|gram|gms|gm|kilograms|kg|mg|mcg|μg|g|"
    "millilitres|millilitre|milliliters|milliliter|ml|litres|litre|liters|liter|"
    "teaspoons|teaspoon|tsp|tablespoons|tablespoon|tbsp|cups|cup|drops|drop|"
    "capsules|capsule|tablets|tablet|tabs|tab|pills|pill|ounces|ounce|oz|"
    "tolas|tola|mashas|masha|rattis|ratti|karshas|karsha|palas|pala"
)
DOSAGE_PATTERNS = (
    (re.compile(rf"{BOUNDARY_BEFORE}{_RANGE}\s*(?:{_UNITS}){BOUNDARY_AFTER}"), "quantity with unit"),
    (re.compile(
        rf"{BOUNDARY_BEFORE}(?:(?:once|twice|thrice|\d+\s*times)\s+(?:a|per|each)\s+day|"
        rf"(?:once|twice|thrice)\s+daily|every\s+\d+\s+hours?){BOUNDARY_AFTER}"
    ), "frequency"),
)


@dataclass(frozen=True)
class Entity:
    """One recognised entity. ``start``/``end`` index the original query."""

    text: str
    canonical: str
    entry_id: str | None
    label: str
    start: int
    end: int
    method: str
    evidence: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _CompiledTerm:
    pattern: re.Pattern
    terms: tuple[LexiconTerm, ...]  # several entries may share a surface form


@dataclass(frozen=True)
class _Candidate:
    start: int
    end: int
    label: str
    priority: tuple[int, int]
    term: LexiconTerm | None
    alternatives: tuple[LexiconTerm, ...]
    dosage_kind: str | None = None


def _term_regex(surface: str) -> re.Pattern:
    tokens = [re.escape(t) for t in re.split(r"[\s\-]+", fold(surface)) if t]
    body = r"[\s\-]+".join(tokens)
    return re.compile(f"{BOUNDARY_BEFORE}{body}{BOUNDARY_AFTER}")


class NEREngine:
    """Dictionary + regex entity recogniser built once from a knowledge base."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        grouped: dict[str, list[LexiconTerm]] = {}
        for term in kb.lexicon():
            folded = fold(term.surface)
            if folded.strip():
                grouped.setdefault(folded, []).append(term)
        self._compiled: list[_CompiledTerm] = []
        for folded, terms in grouped.items():
            terms.sort(key=lambda t: (LABEL_PRIORITY[t.label], KIND_PRIORITY.get(t.kind, 9)))
            self._compiled.append(_CompiledTerm(_term_regex(terms[0].surface), tuple(terms)))
        self.term_count = sum(len(c.terms) for c in self._compiled)

    # ------------------------------------------------------------------
    def extract(self, text: object) -> list[Entity]:
        """Return non-overlapping entities sorted by position."""
        if not isinstance(text, str) or not text.strip():
            return []
        folded, index_map = fold_with_map(text)
        if not folded:
            return []

        candidates: list[_Candidate] = []
        for compiled in self._compiled:
            best = compiled.terms[0]
            for match in compiled.pattern.finditer(folded):
                start, end = self._original_span(text, index_map, match.start(), match.end())
                candidates.append(_Candidate(
                    start, end, best.label,
                    (LABEL_PRIORITY[best.label], KIND_PRIORITY.get(best.kind, 9)),
                    best, compiled.terms[1:],
                ))
        for pattern, kind in DOSAGE_PATTERNS:
            for match in pattern.finditer(folded):
                start, end = self._original_span(text, index_map, match.start(), match.end())
                candidates.append(_Candidate(
                    start, end, "DOSAGE_MENTION", (LABEL_PRIORITY["DOSAGE_MENTION"], 0), None, (), kind,
                ))

        # Longest span first, then most specific category, then earliest.
        candidates.sort(key=lambda c: (-(c.end - c.start), c.priority, c.start))
        chosen: list[_Candidate] = []
        for cand in candidates:
            if any(cand.start < c.end and cand.end > c.start for c in chosen):
                continue
            chosen.append(cand)
        chosen.sort(key=lambda c: c.start)
        return [self._to_entity(text, c) for c in chosen]

    # ------------------------------------------------------------------
    @staticmethod
    def _original_span(text: str, index_map: list[int], fstart: int, fend: int) -> tuple[int, int]:
        start = index_map[fstart]
        end = index_map[fend - 1] + 1
        # Include trailing combining accents that folding removed (e.g. "a" + U+0304).
        while end < len(text) and unicodedata.combining(text[end]) and 0x0300 <= ord(text[end]) < 0x0370:
            end += 1
        return start, end

    def _to_entity(self, text: str, cand: _Candidate) -> Entity:
        span = text[cand.start:cand.end]
        if cand.term is None:
            return Entity(
                text=span,
                canonical=" ".join(span.split()),
                entry_id=None,
                label="DOSAGE_MENTION",
                start=cand.start,
                end=cand.end,
                method=DOSAGE_METHOD,
                evidence=(
                    f"Matched the {cand.dosage_kind} pattern. Recorded only as a mention in the query; "
                    "AyurBot does not judge whether any amount is safe, effective or recommended."
                ),
            )
        term = cand.term
        entry = self.kb.get(term.entry_id)
        canonical = entry.canonical_name if entry else term.surface
        if term.kind == "approximate English gloss":
            method = "Dictionary match (approximate English gloss)"
        else:
            method = f"Exact dictionary match ({term.kind})"
        evidence = (
            f"'{span}' matches the {term.kind} '{term.surface}' of knowledge-base entry "
            f"'{term.entry_id}' ({canonical}). Whole-word match, ignoring case and accents."
        )
        if term.label == "BOTANICAL_NAME":
            evidence += f" Botanical name of the herb {canonical}."
        if term.kind == "approximate English gloss" and entry is not None:
            evidence += " The English word is only an approximate gloss of the classical term."
        if cand.alternatives:
            others = ", ".join(sorted({a.entry_id for a in cand.alternatives if a.entry_id != term.entry_id}))
            if others:
                evidence += f" Ambiguous: also listed for {others}; the more specific entry was chosen."
        return Entity(
            text=span,
            canonical=canonical,
            entry_id=term.entry_id,
            label=term.label,
            start=cand.start,
            end=cand.end,
            method=method,
            evidence=evidence,
        )


def highlight_spans(text: str, entities: list[Entity]) -> list[tuple[str, str | None]]:
    """Split ``text`` into ``(segment, label_or_None)`` pairs for display."""
    segments: list[tuple[str, str | None]] = []
    cursor = 0
    for ent in sorted(entities, key=lambda e: e.start):
        if ent.start < cursor:  # defensive: never emit overlapping segments
            continue
        if ent.start > cursor:
            segments.append((text[cursor:ent.start], None))
        segments.append((text[ent.start:ent.end], ent.label))
        cursor = ent.end
    if cursor < len(text):
        segments.append((text[cursor:], None))
    return segments
