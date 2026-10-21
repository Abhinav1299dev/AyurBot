"""Loading and validating the local JSON knowledge base.

The knowledge base is a single JSON file (``data/herbs.json`` by default).
New herbs, doshas, terms or concepts can be added by editing that file; no
Python changes are needed. The loader validates the whole file at startup and
reports *all* problems it finds in one readable error.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .text_utils import fold

ENTRY_TYPES = {
    "herb": "HERB",
    "dosha": "DOSHA",
    "condition_term": "CONDITION_TERM",
    "traditional_property": "TRADITIONAL_PROPERTY",
    "concept": "AYURVEDIC_CONCEPT",
}

ENTITY_LABELS = (
    "HERB",
    "BOTANICAL_NAME",
    "DOSHA",
    "CONDITION_TERM",
    "TRADITIONAL_PROPERTY",
    "DOSAGE_MENTION",
    "AYURVEDIC_CONCEPT",
)

TYPE_DISPLAY = {
    "herb": "Herb",
    "dosha": "Dosha",
    "condition_term": "Classical condition term",
    "traditional_property": "Traditional property",
    "concept": "Ayurvedic concept",
}

_ID_RE = re.compile(r"^[a-z0-9_]+$")
_STRING_LIST_FIELDS = (
    "synonyms",
    "sanskrit_names",
    "devanagari",
    "approximate_english_terms",
    "botanical_synonyms",
    "safety_notes",
    "elements",
    "functions",
)
_OPTIONAL_STRING_FIELDS = (
    "english_name",
    "botanical_name",
    "family",
    "evidence_note",
    "mapping_note",
    "ambiguity_note",
    "data_status",
    "property_kind",
)


class KnowledgeBaseError(Exception):
    """Raised when the knowledge base cannot be loaded or is invalid."""

    def __init__(self, message: str, problems: Iterable[str] | None = None):
        self.problems = list(problems or [])
        full = message
        if self.problems:
            shown = self.problems[:25]
            full += "\n" + "\n".join(f"  - {p}" for p in shown)
            if len(self.problems) > len(shown):
                full += f"\n  ... and {len(self.problems) - len(shown)} more problem(s)."
        super().__init__(full)


@dataclass(frozen=True)
class LexiconTerm:
    """One surface form that the NER engine can recognise."""

    surface: str
    entry_id: str
    label: str
    kind: str  # e.g. "canonical name", "synonym", "botanical name"


@dataclass
class Entry:
    """A validated knowledge-base entry. ``raw`` keeps every original field."""

    id: str
    type: str
    canonical_name: str
    summary: str
    raw: dict[str, Any] = field(repr=False)

    @property
    def label(self) -> str:
        return ENTRY_TYPES[self.type]

    @property
    def type_display(self) -> str:
        return TYPE_DISPLAY[self.type]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def string_list(self, key: str) -> list[str]:
        value = self.raw.get(key) or []
        return [v for v in value if isinstance(v, str) and v.strip()]


class KnowledgeBase:
    """Validated, read-only collection of entries plus lookup helpers."""

    def __init__(self, entries: list[Entry], metadata: dict[str, Any] | None = None,
                 warnings: list[str] | None = None, path: Path | None = None):
        if not entries:
            raise KnowledgeBaseError("The knowledge base contains no entries.")
        self.entries = entries
        self.metadata = metadata or {}
        self.warnings = warnings or []
        self.path = path
        self.by_id = {e.id: e for e in entries}
        self._name_index: dict[str, str] = {}
        for entry in entries:
            for name in [entry.canonical_name, *entry.string_list("synonyms")]:
                self._name_index.setdefault(fold(name), entry.id)
        self._documents = [(e.id, self._document_text(e)) for e in entries]

    # ----------------------------------------------------------- loading
    @classmethod
    def from_file(cls, path: str | Path) -> "KnowledgeBase":
        path = Path(path)
        if not path.exists():
            raise KnowledgeBaseError(
                f"Knowledge base file not found: {path}. "
                "Check AYURBOT_KB_PATH or restore data/herbs.json from the project ZIP."
            )
        if not path.is_file():
            raise KnowledgeBaseError(f"Knowledge base path is not a file: {path}.")
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            raise KnowledgeBaseError(f"Could not read {path} as UTF-8 text: {exc}.") from exc
        if not text.strip():
            raise KnowledgeBaseError(f"Knowledge base file is empty: {path}.")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise KnowledgeBaseError(
                f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}. "
                "Common causes are a missing comma or a trailing comma before ] or }."
            ) from exc
        return cls.from_dict(data, path=path)

    @classmethod
    def from_dict(cls, data: Any, path: Path | None = None) -> "KnowledgeBase":
        entries, metadata, warnings = validate_kb_data(data)
        return cls(entries, metadata=metadata, warnings=warnings, path=path)

    # ----------------------------------------------------------- lookups
    def get(self, entry_id: str) -> Entry | None:
        return self.by_id.get(entry_id)

    def find_by_name(self, name: str) -> Entry | None:
        entry_id = self._name_index.get(fold(name))
        return self.by_id.get(entry_id) if entry_id else None

    def entries_of_type(self, entry_type: str) -> list[Entry]:
        return [e for e in self.entries if e.type == entry_type]

    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for entry in self.entries:
            result[entry.type] = result.get(entry.type, 0) + 1
        return result

    @property
    def documents(self) -> list[tuple[str, str]]:
        """``(entry_id, searchable_text)`` pairs used for retrieval."""
        return list(self._documents)

    def herbs_pacifying(self, dosha_id: str) -> list[Entry]:
        return [h for h in self.entries_of_type("herb") if dosha_id in _dosha_list(h, "pacifies")]

    def herbs_aggravating(self, dosha_id: str) -> list[Entry]:
        return [h for h in self.entries_of_type("herb") if dosha_id in _dosha_list(h, "may_aggravate")]

    def herbs_for_term(self, term_id: str) -> list[Entry]:
        return [h for h in self.entries_of_type("herb") if term_id in herb_use_ids(h)]

    def herbs_with_property(self, property_entry: Entry) -> list[Entry]:
        names = {fold(property_entry.canonical_name), *(fold(s) for s in property_entry.string_list("synonyms"))}
        return [h for h in self.entries_of_type("herb") if names & {fold(p) for p in herb_property_values(h)}]

    def property_entry(self, value: str) -> Entry | None:
        entry = self.find_by_name(value)
        return entry if entry and entry.type == "traditional_property" else None

    # ----------------------------------------------------------- lexicon
    def lexicon(self) -> list[LexiconTerm]:
        """All surface forms for the rule-based NER engine, de-duplicated."""
        terms: list[LexiconTerm] = []
        seen: set[tuple[str, str, str]] = set()

        def add(surface: Any, entry: Entry, label: str, kind: str) -> None:
            if not isinstance(surface, str) or not surface.strip():
                return
            key = (fold(surface.strip()), entry.id, label)
            if key in seen:
                return
            seen.add(key)
            terms.append(LexiconTerm(surface.strip(), entry.id, label, kind))

        for entry in self.entries:
            label = entry.label
            add(entry.canonical_name, entry, label, "canonical name")
            for s in entry.string_list("sanskrit_names"):
                add(s, entry, label, "Sanskrit name")
            for s in entry.string_list("synonyms"):
                add(s, entry, label, "synonym")
            for s in entry.string_list("devanagari"):
                add(s, entry, label, "Devanagari name")
            if entry.get("english_name"):
                add(entry.get("english_name"), entry, label, "English name")
            for s in entry.string_list("approximate_english_terms"):
                add(s, entry, label, "approximate English gloss")
            if entry.type == "herb":
                add(entry.get("botanical_name"), entry, "BOTANICAL_NAME", "botanical name")
                for s in entry.string_list("botanical_synonyms"):
                    add(s, entry, "BOTANICAL_NAME", "botanical synonym")
        return terms

    # ----------------------------------------------------------- documents
    def _document_text(self, entry: Entry) -> str:
        parts = [f"{entry.canonical_name} ({entry.type_display})."]
        names = entry.string_list("sanskrit_names") + entry.string_list("synonyms")
        if names:
            parts.append("Also known as " + ", ".join(dict.fromkeys(names)) + ".")
        if entry.get("english_name"):
            parts.append(f"English name: {entry.get('english_name')}.")
        if entry.get("botanical_name"):
            parts.append(f"Botanical name: {entry.get('botanical_name')}.")
        glosses = entry.string_list("approximate_english_terms")
        if glosses:
            parts.append("Approximate English terms: " + ", ".join(glosses) + ".")
        parts.append(entry.summary)
        if entry.type == "herb":
            for use in entry.get("traditional_uses") or []:
                term = self.by_id.get(use.get("term_id", ""))
                if term:
                    gloss = ", ".join(term.string_list("approximate_english_terms"))
                    parts.append(f"Traditional use: {term.canonical_name}" + (f" ({gloss})." if gloss else "."))
                parts.append(use.get("description", ""))
            props = herb_property_values(entry)
            if props:
                parts.append("Traditional properties: " + ", ".join(props) + ".")
            pac = [self.by_id[d].canonical_name for d in _dosha_list(entry, "pacifies") if d in self.by_id]
            agg = [self.by_id[d].canonical_name for d in _dosha_list(entry, "may_aggravate") if d in self.by_id]
            if pac:
                parts.append("Traditionally said to pacify " + ", ".join(pac) + ".")
            if agg:
                parts.append("Traditionally said to possibly aggravate " + ", ".join(agg) + ".")
        for key in ("elements", "functions"):
            values = entry.string_list(key)
            if values:
                parts.append(f"{key.capitalize()}: " + ", ".join(values) + ".")
        return " ".join(p for p in parts if p)


# --------------------------------------------------------------- helpers
def _dosha_list(herb: Entry, key: str) -> list[str]:
    assoc = herb.get("dosha_associations") or {}
    return [d for d in assoc.get(key, []) if isinstance(d, str)]


def herb_use_ids(herb: Entry) -> list[str]:
    return [u.get("term_id") for u in herb.get("traditional_uses") or [] if isinstance(u, dict)]


def herb_property_values(herb: Entry) -> list[str]:
    values: list[str] = []
    props = herb.get("traditional_properties") or {}
    for value in props.values():
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(v for v in value if isinstance(v, str))
    return list(dict.fromkeys(values))


def validate_kb_data(data: Any) -> tuple[list[Entry], dict[str, Any], list[str]]:
    """Validate parsed JSON. Returns ``(entries, metadata, warnings)``.

    Raises :class:`KnowledgeBaseError` listing every structural problem.
    """
    if isinstance(data, list):
        data = {"entries": data}
    if not isinstance(data, dict):
        raise KnowledgeBaseError("The knowledge base must be a JSON object with an \"entries\" list.")
    raw_entries = data.get("entries")
    if raw_entries is None:
        raise KnowledgeBaseError("The knowledge base has no \"entries\" key.")
    if not isinstance(raw_entries, list):
        raise KnowledgeBaseError("\"entries\" must be a JSON list.")
    if not raw_entries:
        raise KnowledgeBaseError("The knowledge base \"entries\" list is empty. Add at least one entry.")

    problems: list[str] = []
    warnings: list[str] = []
    entries: list[Entry] = []
    seen_ids: set[str] = set()

    for index, item in enumerate(raw_entries):
        where = f"entries[{index}]"
        if not isinstance(item, dict):
            problems.append(f"{where} must be an object, got {type(item).__name__}.")
            continue
        entry_id = item.get("id")
        name = item.get("canonical_name")
        if isinstance(entry_id, str):
            where = f"entries[{index}] (id={entry_id!r})"
        start = len(problems)

        if not isinstance(entry_id, str) or not _ID_RE.match(entry_id):
            problems.append(f"{where}: \"id\" must be lower-case letters, digits or underscores.")
        elif entry_id in seen_ids:
            problems.append(f"{where}: duplicate id {entry_id!r}.")
        if item.get("type") not in ENTRY_TYPES:
            problems.append(f"{where}: \"type\" must be one of {sorted(ENTRY_TYPES)}.")
        if not isinstance(name, str) or not name.strip():
            problems.append(f"{where}: \"canonical_name\" must be a non-empty string.")
        if not isinstance(item.get("summary"), str) or not item["summary"].strip():
            problems.append(f"{where}: \"summary\" must be a non-empty string.")
        for key in _STRING_LIST_FIELDS:
            if key in item and not (isinstance(item[key], list) and all(isinstance(v, str) for v in item[key])):
                problems.append(f"{where}: \"{key}\" must be a list of strings.")
        for key in _OPTIONAL_STRING_FIELDS:
            if key in item and item[key] is not None and not isinstance(item[key], str):
                problems.append(f"{where}: \"{key}\" must be a string or null.")
        if "source" in item and item["source"] is not None and not isinstance(item["source"], dict):
            problems.append(f"{where}: \"source\" must be an object or null.")

        if item.get("type") == "herb":
            uses = item.get("traditional_uses", [])
            if not isinstance(uses, list) or not all(
                isinstance(u, dict) and isinstance(u.get("term_id"), str) and isinstance(u.get("description"), str)
                for u in uses
            ):
                problems.append(f"{where}: \"traditional_uses\" must be a list of {{\"term_id\", \"description\"}} objects.")
            props = item.get("traditional_properties", {})
            if not isinstance(props, dict):
                problems.append(f"{where}: \"traditional_properties\" must be an object.")
            assoc = item.get("dosha_associations", {})
            if not isinstance(assoc, dict) or not all(
                isinstance(assoc.get(k, []), list) for k in ("pacifies", "may_aggravate")
            ):
                problems.append(f"{where}: \"dosha_associations\" must contain \"pacifies\"/\"may_aggravate\" lists.")
            if not item.get("botanical_name"):
                warnings.append(f"{where}: herb has no botanical_name.")

        if len(problems) == start:
            seen_ids.add(entry_id)
            entries.append(Entry(entry_id, item["type"], name.strip(), item["summary"].strip(), dict(item)))

    if problems:
        raise KnowledgeBaseError(f"The knowledge base has {len(problems)} problem(s):", problems)

    ids = {e.id for e in entries}
    for entry in entries:
        if entry.type != "herb":
            continue
        for term_id in herb_use_ids(entry):
            if term_id not in ids:
                warnings.append(f"{entry.id}: traditional use refers to unknown term_id {term_id!r}.")
        for key in ("pacifies", "may_aggravate"):
            for dosha_id in _dosha_list(entry, key):
                target = next((e for e in entries if e.id == dosha_id), None)
                if target is None or target.type != "dosha":
                    warnings.append(f"{entry.id}: dosha_associations.{key} refers to unknown dosha {dosha_id!r}.")
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    return entries, metadata, warnings
