"""Text helpers shared by the NER and retrieval engines.

The key function is :func:`fold_with_map`. It produces a lower-cased,
Latin-diacritic-free copy of a string *and* a map from every character of
that copy back to its position in the original. Matching is done on the
folded copy, while reported offsets always refer to the original text, so
highlighting stays correct for input such as ``"Tulasī"`` or ``"तुलसी"``.
"""

from __future__ import annotations

import re
import unicodedata

# Characters treated as part of a word when checking whole-word boundaries.
# \w covers letters/digits in all scripts; the Devanagari block is added so
# that vowel signs (matras), which are not \w, are never treated as gaps.
WORD_CHAR_CLASS = r"\w\u0900-\u097F"
BOUNDARY_BEFORE = rf"(?<![{WORD_CHAR_CLASS}])"
BOUNDARY_AFTER = rf"(?![{WORD_CHAR_CLASS}])"

_LATIN_COMBINING = range(0x0300, 0x0370)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TOKEN_RE = re.compile(rf"[{WORD_CHAR_CLASS}]+")


def fold_with_map(text: str) -> tuple[str, list[int]]:
    """Return ``(folded_text, index_map)``.

    ``folded_text[i]`` came from ``text[index_map[i]]``. Folding applies
    compatibility decomposition, removes Latin combining accents (so
    ``ā`` becomes ``a``) and case-folds. Devanagari marks are preserved.
    """
    out: list[str] = []
    index_map: list[int] = []
    for i, char in enumerate(text):
        for part in unicodedata.normalize("NFKD", char):
            if ord(part) in _LATIN_COMBINING:
                continue
            for folded in part.casefold():
                out.append(folded)
                index_map.append(i)
    return "".join(out), index_map


def fold(text: str) -> str:
    """Folded form of ``text`` (see :func:`fold_with_map`)."""
    return fold_with_map(text)[0]


def clean_user_text(text: object) -> str:
    """Convert arbitrary input to a clean string without control characters."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFC", text)
    return _CONTROL_CHARS.sub(" ", text).strip()


STOPWORDS = frozenset(
    """
    a about above after again all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers him his how i if in into is it its itself just me more
    most my no nor not now of off on once only or other our out over own same she should so some
    such than that the their them then there these they this those through to too under until up
    very was we were what when where which while who whom why will with would you your yours
    tell explain describe give show list please know want like
    help helps helpful good best use used using uses useful benefit benefits
    """.split()
)


def _light_stem(token: str) -> str:
    """Very small plural stripper so that ``wounds`` matches ``wound``."""
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Split text into folded, lightly stemmed word tokens."""
    tokens = [t for t in _TOKEN_RE.findall(fold(text)) if t != "_" and not t.isdigit()]
    tokens = [t.strip("_") for t in tokens if t.strip("_")]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [_light_stem(t) for t in tokens]
