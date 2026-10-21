"""Grounded, template-based response generation.

No generative language model is used. Every sentence in an answer is either a
fixed template or text copied from a knowledge-base entry, so every claim can
be traced back to ``data/herbs.json``. When the knowledge base has nothing
relevant, the engine says so instead of guessing.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import pandas as pd

from .knowledge_base import Entry, KnowledgeBase, herb_property_values, herb_use_ids
from .ner_engine import Entity, NEREngine, highlight_spans
from .retrieval_engine import RetrievalEngine, RetrievalResult
from .text_utils import clean_user_text, fold

DISCLAIMER = (
    "AyurBot is an educational prototype. It describes traditional Ayurvedic knowledge, "
    "which is not the same as clinical evidence, and it has not been clinically validated. "
    "It cannot diagnose, treat or recommend doses. Please consult a qualified healthcare "
    "professional about any health concern."
)
SHORT_DISCLAIMER = (
    "_Educational use only. Traditional descriptions are not clinical evidence; "
    "consult a qualified healthcare professional for medical advice._"
)

ENTITY_COLUMNS = ["Text in query", "Category", "Canonical entry", "Offsets", "Detection method", "Evidence"]
EVIDENCE_COLUMNS = ["Rank", "Entry", "Type", "Why selected", "Score", "Score meaning", "Data status", "Source"]

_EMERGENCY = re.compile(
    r"\b(suicid\w*|kill (?:my ?self|myself)|end my life|self[- ]harm|overdos\w*|poison(?:ed|ing)?|"
    r"chest pain|can'?t breathe|cannot breathe|unconscious|fainted|seizure (?:now|right now)|"
    r"severe bleeding|swallowed)\b",
    re.I,
)
_PERSONAL = re.compile(
    r"\b(should i|can i (?:take|use|give)|how much (?:should|can|do) i|is it safe for (?:me|my)|"
    r"i have|i am suffering|i'm suffering|i suffer|my (?:dose|child|son|daughter|baby|mother|father|wife|husband)|"
    r"i am pregnant|i'm pregnant|while pregnant|breastfeeding|my doctor|my medication|my medicine)\b",
    re.I,
)
_DOSE_QUESTION = re.compile(r"\b(dose|doses|dosage|how much|how many)\b", re.I)
_CURE_WORDS = re.compile(r"\b(cure|cures|curing|treat|treats|treatment|heal|heals|prevent|prevents)\b", re.I)


@dataclass
class ChatResponse:
    """Everything the UI needs for one turn."""

    query: str
    answer: str
    status: str  # ok | no_match | empty | too_long | error
    entities: list[Entity] = field(default_factory=list)
    highlighted: list[tuple[str, str | None]] = field(default_factory=list)
    entity_rows: list[dict] = field(default_factory=list)
    evidence_rows: list[dict] = field(default_factory=list)
    retrieval_mode: str = ""
    retrieval_note: str = ""
    elapsed_ms: float = 0.0

    def entity_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.entity_rows, columns=ENTITY_COLUMNS)

    def evidence_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.evidence_rows, columns=EVIDENCE_COLUMNS)


@dataclass
class _Selected:
    entry: Entry
    reason: str
    result: RetrievalResult | None = None
    named: bool = False


class ResponseEngine:
    """Combines NER output and retrieval results into a readable answer."""

    def __init__(self, kb: KnowledgeBase, ner: NEREngine, retrieval: RetrievalEngine,
                 max_query_chars: int = 1000, max_cards: int = 3):
        self.kb = kb
        self.ner = ner
        self.retrieval = retrieval
        self.max_query_chars = max_query_chars
        self.max_cards = max_cards

    # ================================================================ public
    def respond(self, raw_query: object) -> ChatResponse:
        started = time.perf_counter()
        query = clean_user_text(raw_query)
        if not query:
            return self._finish(ChatResponse(
                query, "Please type a question, for example *What is Tulsi?*", "empty"), started)
        if len(query) > self.max_query_chars:
            return self._finish(ChatResponse(
                query[: self.max_query_chars],
                f"Your message is {len(query)} characters long. Please shorten it to "
                f"{self.max_query_chars} characters or fewer and ask one question at a time.",
                "too_long"), started)

        entities = self.ner.extract(query)
        retrieval = self.retrieval.search(query)
        selected = self._select_entries(entities, retrieval.results)

        response = ChatResponse(
            query=query,
            answer="",
            status="ok" if selected else "no_match",
            entities=entities,
            highlighted=highlight_spans(query, entities),
            entity_rows=[self._entity_row(e) for e in entities],
            evidence_rows=[self._evidence_row(i, s) for i, s in enumerate(selected, 1)],
            retrieval_mode=retrieval.mode,
            retrieval_note=retrieval.note,
        )
        response.answer = self._compose(query, entities, selected)
        return self._finish(response, started)

    # ============================================================ selection
    def _select_entries(self, entities: list[Entity], results: list[RetrievalResult]) -> list[_Selected]:
        selected: list[_Selected] = []
        seen: set[str] = set()
        # Entities named in the query come first, herbs before other categories.
        named = sorted(
            (e for e in entities if e.entry_id),
            key=lambda e: (0 if e.label in ("HERB", "BOTANICAL_NAME") else 1, e.start),
        )
        for ent in named:
            if ent.entry_id in seen:
                continue
            entry = self.kb.get(ent.entry_id)
            if entry:
                seen.add(entry.id)
                selected.append(_Selected(entry, f"Named in the query as '{ent.text}' ({ent.label})", named=True))
        for res in results:
            if res.entry_id in seen:
                for sel in selected:
                    if sel.entry.id == res.entry_id and sel.result is None:
                        sel.result = res
                continue
            entry = self.kb.get(res.entry_id)
            if entry:
                seen.add(entry.id)
                selected.append(_Selected(entry, f"Retrieved by {res.method} search", res))
        return selected

    # =========================================================== composing
    def _compose(self, query: str, entities: list[Entity], selected: list[_Selected]) -> str:
        parts: list[str] = []
        safety = self._safety_notices(query, entities)
        if safety:
            parts.append(safety)

        if not selected:
            parts.append(
                "I couldn't find anything relevant to that in the AyurBot knowledge base, so I won't guess.\n\n"
                "Try asking about an herb (Tulsi, Haridra, Ashwagandha, Brahmi, Guduchi, Neem), a dosha "
                "(Vata, Pitta, Kapha), a classical term such as Kasa or Jwara, or a concept such as Agni or Prakriti."
            )
            parts.append(SHORT_DISCLAIMER)
            return "\n\n".join(parts)

        relations = self._relations(entities)
        if relations:
            parts.append("**What the knowledge base records**\n\n" + "\n".join(f"- {r}" for r in relations))

        if _CURE_WORDS.search(query):
            parts.append(
                "> This knowledge base records *traditional descriptions* only. It cannot confirm that any "
                "herb treats, cures or prevents a disease."
            )

        # Entries the user named get full cards. Retrieved entries get cards only
        # when nothing was named, so answers stay focused on the question.
        named = [s for s in selected if s.named]
        primary = (named or selected)[: self.max_cards]
        parts.extend(self._card(s.entry) for s in primary)
        others = [s for s in selected if s not in primary]
        if others:
            extra = ", ".join(s.entry.canonical_name for s in others)
            parts.append(f"_Related entries found by search: {extra}. See the evidence table for scores._")
        parts.append(SHORT_DISCLAIMER)
        return "\n\n".join(parts)

    def _safety_notices(self, query: str, entities: list[Entity]) -> str:
        notes: list[str] = []
        if _EMERGENCY.search(query):
            notes.append(
                "**If you or someone else may be in danger or having a medical emergency, contact local "
                "emergency services now (112 in India) or go to the nearest emergency department. If you are "
                "thinking about harming yourself, please reach out to a crisis line or someone you trust right away.**"
            )
        if _PERSONAL.search(query):
            notes.append(
                "AyurBot can't give personal medical advice, diagnose conditions or say whether something is "
                "right for you. A qualified healthcare professional can take your health history and "
                "medications into account."
            )
        has_dosage = any(e.label == "DOSAGE_MENTION" for e in entities)
        if has_dosage:
            mentions = ", ".join(f"'{e.text}'" for e in entities if e.label == "DOSAGE_MENTION")
            notes.append(
                f"I noticed a quantity in your message ({mentions}). AyurBot only records it as a mention; "
                "it does not judge whether any amount is safe, effective or appropriate, and it does not "
                "recommend doses."
            )
        elif _DOSE_QUESTION.search(query):
            notes.append(
                "AyurBot does not provide dosage recommendations. Appropriate amounts depend on the product, "
                "the person and their medications, and should come from a qualified professional."
            )
        return "\n\n".join(f"> {n}" for n in notes)

    def _relations(self, entities: list[Entity]) -> list[str]:
        def unique(label_set: set[str]) -> list[Entry]:
            out: list[Entry] = []
            for e in entities:
                if e.label in label_set and e.entry_id and all(x.id != e.entry_id for x in out):
                    entry = self.kb.get(e.entry_id)
                    if entry:
                        out.append(entry)
            return out

        herbs = unique({"HERB", "BOTANICAL_NAME"})
        doshas = [d for d in unique({"DOSHA"}) if d.type == "dosha"]
        terms = [t for t in unique({"CONDITION_TERM"}) if t.type == "condition_term"]
        props = [p for p in unique({"TRADITIONAL_PROPERTY"}) if p.type == "traditional_property"]
        lines: list[str] = []

        for herb in herbs:
            assoc = herb.get("dosha_associations") or {}
            for dosha in doshas:
                if dosha.id in assoc.get("pacifies", []):
                    lines.append(f"**{herb.canonical_name}** is traditionally described as pacifying **{dosha.canonical_name}**.")
                elif dosha.id in assoc.get("may_aggravate", []):
                    lines.append(f"**{herb.canonical_name}** is traditionally described as possibly aggravating **{dosha.canonical_name}**.")
                else:
                    lines.append(f"This knowledge base does not record an effect of **{herb.canonical_name}** on **{dosha.canonical_name}**.")
            for term in terms:
                use = next((u for u in herb.get("traditional_uses") or [] if u.get("term_id") == term.id), None)
                if use:
                    lines.append(f"**{herb.canonical_name}**: {use['description']}")
                else:
                    lines.append(f"This knowledge base has no traditional-use record linking **{herb.canonical_name}** to **{term.canonical_name}**.")
            for prop in props:
                values = {fold(v) for v in herb_property_values(herb)}
                names = {fold(prop.canonical_name), *(fold(s) for s in prop.string_list("synonyms"))}
                if values & names:
                    lines.append(f"**{herb.canonical_name}** is traditionally classed as **{prop.canonical_name}**.")
                else:
                    lines.append(f"This knowledge base does not list **{prop.canonical_name}** among the properties of **{herb.canonical_name}**.")

        if not herbs:
            for term in terms:
                found = self.kb.herbs_for_term(term.id)
                if found:
                    lines.append(
                        f"Herbs in this knowledge base traditionally described in the context of **{term.canonical_name}**: "
                        + ", ".join(h.canonical_name for h in found) + "."
                    )
            for dosha in doshas:
                pac = self.kb.herbs_pacifying(dosha.id)
                agg = self.kb.herbs_aggravating(dosha.id)
                if pac:
                    lines.append(f"Herbs traditionally described as pacifying **{dosha.canonical_name}**: "
                                 + ", ".join(h.canonical_name for h in pac) + ".")
                if agg:
                    lines.append(f"Herbs traditionally described as possibly aggravating **{dosha.canonical_name}**: "
                                 + ", ".join(h.canonical_name for h in agg) + ".")
            for prop in props:
                found = self.kb.herbs_with_property(prop)
                if found:
                    lines.append(f"Herbs traditionally classed as **{prop.canonical_name}**: "
                                 + ", ".join(h.canonical_name for h in found) + ".")
        return lines

    # ================================================================ cards
    def _card(self, entry: Entry) -> str:
        if entry.type == "herb":
            return self._herb_card(entry)
        lines = [f"#### {entry.canonical_name} ({entry.type_display})", entry.summary]
        glosses = entry.string_list("approximate_english_terms")
        if glosses:
            lines.append(f"**Approximate English terms:** {', '.join(glosses)}")
        if entry.string_list("elements"):
            lines.append(f"**Elements:** {', '.join(entry.string_list('elements'))}")
        if entry.string_list("functions"):
            lines.append(f"**Functions in Ayurvedic theory:** {'; '.join(entry.string_list('functions'))}")
        if entry.get("mapping_note"):
            lines.append(f"**Terminology note:** {entry.get('mapping_note')}")
        lines.append(self._evidence_line(entry))
        return "\n\n".join(lines)

    def _herb_card(self, herb: Entry) -> str:
        title = f"#### {herb.canonical_name}"
        bits = [b for b in (herb.get("english_name"), f"*{herb.get('botanical_name')}*" if herb.get("botanical_name") else None) if b]
        if bits:
            title += " (" + ", ".join(bits) + ")"
        lines = [title, herb.summary]

        uses = []
        for use in herb.get("traditional_uses") or []:
            term = self.kb.get(use.get("term_id", ""))
            name = term.canonical_name if term else use.get("term_id", "unknown term")
            uses.append(f"- **{name}**: {use.get('description', 'No description available.')}")
        lines.append("**Traditional uses** (classical descriptions, not clinical evidence)\n\n"
                     + ("\n".join(uses) if uses else "- Not available in this knowledge base."))

        props = herb.get("traditional_properties") or {}
        labels = {"rasa": "Rasa (taste)", "guna": "Guna (qualities)", "virya": "Virya (potency)",
                  "vipaka": "Vipaka (post-digestive effect)", "karma": "Karma (traditional actions)"}
        prop_lines = []
        for key, label in labels.items():
            value = props.get(key)
            if value:
                prop_lines.append(f"{label}: {', '.join(value) if isinstance(value, list) else value}")
        lines.append("**Traditional properties:** " + ("; ".join(prop_lines) if prop_lines else "Not available."))

        assoc = herb.get("dosha_associations") or {}
        pac = [self.kb.get(d).canonical_name for d in assoc.get("pacifies", []) if self.kb.get(d)]
        agg = [self.kb.get(d).canonical_name for d in assoc.get("may_aggravate", []) if self.kb.get(d)]
        dosha_text = (f"traditionally said to pacify {', '.join(pac)}" if pac else "no pacifying effect recorded")
        if agg:
            dosha_text += f"; may aggravate {', '.join(agg)}"
        if assoc.get("note"):
            dosha_text += f". {assoc['note']}"
        lines.append(f"**Dosha associations:** {dosha_text}")

        if herb.get("ambiguity_note"):
            lines.append(f"**Name note:** {herb.get('ambiguity_note')}")
        safety = herb.string_list("safety_notes")
        lines.append("**Safety notes**\n\n" + ("\n".join(f"- {s}" for s in safety) if safety
                     else "- No safety information is recorded for this entry. Absence of a warning does not mean it is safe."))
        lines.append(self._evidence_line(herb))
        return "\n\n".join(lines)

    @staticmethod
    def _evidence_line(entry: Entry) -> str:
        note = entry.get("evidence_note") or "No evidence note is recorded for this entry."
        status = entry.get("data_status") or "unspecified"
        return f"<sub>Evidence: {note} Data status: {status}.</sub>"

    # ================================================================ tables
    @staticmethod
    def _entity_row(e: Entity) -> dict:
        return {
            "Text in query": e.text,
            "Category": e.label,
            "Canonical entry": e.canonical,
            "Offsets": f"{e.start}–{e.end}",
            "Detection method": e.method,
            "Evidence": e.evidence,
        }

    @staticmethod
    def _evidence_row(rank: int, sel: _Selected) -> dict:
        source = sel.entry.get("source") or {}
        if isinstance(source, dict):
            source_text = source.get("reference") or source.get("description") or "Not available"
        else:
            source_text = "Not available"
        res = sel.result
        return {
            "Rank": rank,
            "Entry": sel.entry.canonical_name,
            "Type": sel.entry.type_display,
            "Why selected": sel.reason,
            "Score": f"{res.score:.3f} ({res.method})" if res else "n/a (direct entity match)",
            "Score meaning": res.score_meaning if res else "Selected because the query names this entry.",
            "Data status": sel.entry.get("data_status") or "unspecified",
            "Source": source_text,
        }

    @staticmethod
    def _finish(response: ChatResponse, started: float) -> ChatResponse:
        response.elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        if not response.highlighted and response.query:
            response.highlighted = [(response.query, None)]
        return response
