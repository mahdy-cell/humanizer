"""
core/isolator.py — Layer 1: Isolation Layer

Detects and protects quoted text, citations, legal/ISO clauses,
mathematical equations, named entities, and acronyms.
Replaces protected spans with tokens like [REF_1], [QUOTE_1], [EQ_1], [NE_1].
"""

import re
import warnings
from typing import Dict, Tuple

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False
    warnings.warn("spaCy not available; NER-based isolation disabled.")

# --------------------------------------------------------------------------- #
# Token counter helpers
# --------------------------------------------------------------------------- #
_COUNTERS: Dict[str, int] = {}


def _next_token(prefix: str) -> str:
    _COUNTERS[prefix] = _COUNTERS.get(prefix, 0) + 1
    return f"[{prefix}_{_COUNTERS[prefix]}]"


def reset_counters() -> None:
    """Reset all protection counters (call before each document)."""
    _COUNTERS.clear()


# --------------------------------------------------------------------------- #
# Protection map type
# --------------------------------------------------------------------------- #
ProtectionMap = Dict[str, str]   # token → original span


# --------------------------------------------------------------------------- #
# Individual isolation routines
# --------------------------------------------------------------------------- #

def isolate_citations(text: str, pmap: ProtectionMap) -> str:
    """Replace (Author, YEAR) and Author (YEAR) style citations with [REF_n] tokens."""
    # Pattern 1: (Author, YEAR) or (Author et al., YEAR; Author2, YEAR)
    pattern1 = r'\(([A-Z][A-Za-z\-]+(?:\s+et\s+al\.?)?,\s*\d{4}(?:[a-z])?(?:;\s*[A-Z][A-Za-z\-]+(?:\s+et\s+al\.?)?,\s*\d{4}(?:[a-z])?)*)\)'
    # Pattern 2: Author (YEAR) — surname followed by parenthetical year
    pattern2 = r'\b([A-Z][A-Za-z\-]+(?:\s+et\s+al\.?)?)\s+\((\d{4}[a-z]?)\)'

    def replacer(m: re.Match) -> str:
        tok = _next_token("REF")
        pmap[tok] = m.group(0)
        return tok

    text = re.sub(pattern1, replacer, text)
    text = re.sub(pattern2, replacer, text)
    return text


def isolate_quotes(text: str, pmap: ProtectionMap) -> str:
    """Replace quoted strings ("...") with [QUOTE_n] tokens."""
    pattern = r'"([^"]{1,500})"'

    def replacer(m: re.Match) -> str:
        tok = _next_token("QUOTE")
        pmap[tok] = m.group(0)
        return tok

    return re.sub(pattern, replacer, text)


def isolate_equations(text: str, pmap: ProtectionMap) -> str:
    """Replace inline LaTeX / simple math expressions with [EQ_n] tokens."""
    patterns = [
        r'\$\$.*?\$\$',           # display math
        r'\$[^$\n]+?\$',          # inline math
        r'\\\[.*?\\\]',           # \[...\]
        r'\\\(.*?\\\)',           # \(...\)
        r'[A-Za-z]\s*=\s*[^\.,;]+',  # simple assignments like E = mc²
    ]
    combined = '|'.join(patterns)

    def replacer(m: re.Match) -> str:
        tok = _next_token("EQ")
        pmap[tok] = m.group(0)
        return tok

    return re.sub(combined, replacer, text, flags=re.DOTALL)


def isolate_legal_iso(text: str, pmap: ProtectionMap) -> str:
    """Replace ISO standard / legal clause references with [LEGAL_n] tokens."""
    pattern = r'(?:ISO|IEC|IEEE|ASTM|DIN|EN)\s+\d[\d\-:\.]*\b'

    def replacer(m: re.Match) -> str:
        tok = _next_token("LEGAL")
        pmap[tok] = m.group(0)
        return tok

    return re.sub(pattern, replacer, text, flags=re.IGNORECASE)


def isolate_acronyms(text: str, pmap: ProtectionMap) -> str:
    """Replace well-formed acronyms (2–6 capital letters) with [ACR_n] tokens."""
    pattern = r'\b[A-Z]{2,6}\b'

    def replacer(m: re.Match) -> str:
        # Don't re-protect already-protected tokens
        val = m.group(0)
        if val.startswith('[') and val.endswith(']'):
            return val
        tok = _next_token("ACR")
        pmap[tok] = val
        return tok

    return re.sub(pattern, replacer, text)


def isolate_named_entities(text: str, pmap: ProtectionMap) -> str:
    """Use spaCy NER to protect ORG, GPE, PERSON entities."""
    if not _SPACY_AVAILABLE:
        return text
    try:
        nlp = _get_nlp()
        doc = nlp(text)
        # Process in reverse order to keep offsets valid
        offsets = [
            (ent.start_char, ent.end_char, ent.text)
            for ent in doc.ents
            if ent.label_ in {"ORG", "GPE", "PERSON", "LOC"}
        ]
        result = list(text)
        for start, end, ent_text in sorted(offsets, reverse=True):
            tok = _next_token("NE")
            pmap[tok] = ent_text
            result[start:end] = list(tok)
        return "".join(result)
    except Exception as exc:
        warnings.warn(f"NER isolation failed: {exc}")
        return text


# --------------------------------------------------------------------------- #
# NLP model cache
# --------------------------------------------------------------------------- #
_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(
                ["python", "-m", "spacy", "download", "en_core_web_sm"],
                check=False,
            )
            _NLP = spacy.load("en_core_web_sm")
    return _NLP


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def isolate(text: str, use_ner: bool = True) -> Tuple[str, ProtectionMap]:
    """
    Apply all isolation routines in order.

    Returns:
        masked_text: text with protected spans replaced by tokens
        pmap: dict mapping token → original span
    """
    reset_counters()
    pmap: ProtectionMap = {}

    text = isolate_citations(text, pmap)
    text = isolate_quotes(text, pmap)
    text = isolate_equations(text, pmap)
    text = isolate_legal_iso(text, pmap)
    if use_ner:
        text = isolate_named_entities(text, pmap)
    text = isolate_acronyms(text, pmap)

    return text, pmap


def restore(text: str, pmap: ProtectionMap) -> str:
    """Restore all protected tokens back to their original spans."""
    for token, original in pmap.items():
        text = text.replace(token, original)
    return text


def is_protected(span: str, pmap: ProtectionMap) -> bool:
    """Return True if *span* is (or contains) a protection token."""
    for token in pmap:
        if token in span:
            return True
    return False
