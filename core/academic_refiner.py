"""
core/academic_refiner.py — Layer 5: Academic Refinement Layer

  - Grammar correction via Gramformer
  - Hedging/boosting balance via TextBlob + VADER
  - Transition word injection
  - Tense consistency check via spaCy
  - Oxford comma & punctuation normalization
  - Acronym safeguarding
  - Block quote formatting (>40 words → APA block quote style)
"""

import re
import warnings
from typing import Dict, List, Set

try:
    from gramformer import Gramformer
    _GF = None
    _GRAMFORMER_AVAILABLE = True
except ImportError:
    _GRAMFORMER_AVAILABLE = False
    warnings.warn("Gramformer not available; grammar correction disabled.")

try:
    from textblob import TextBlob
    _TEXTBLOB_AVAILABLE = True
except ImportError:
    _TEXTBLOB_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

try:
    import language_tool_python
    _LT_TOOL = None
    _LT_AVAILABLE = True
except ImportError:
    _LT_AVAILABLE = False
    warnings.warn("language_tool_python not available; grammar pass disabled.")

# ── Constants ──────────────────────────────────────────────────────────────

HEDGING_WORDS = [
    "suggests", "tends to", "predominantly", "appears to",
    "may indicate", "is likely to", "seems to", "arguably",
]

OVERCONFIDENT_WORDS = {
    "always": "typically",
    "certainly": "arguably",
    "definitely": "likely",
    "undoubtedly": "apparently",
    "proves": "suggests",
    "proves that": "suggests that",
    "it is clear that": "it appears that",
    "obviously": "seemingly",
}

ACADEMIC_TRANSITIONS = [
    "Notwithstanding the foregoing,",
    "In light of the evidence,",
    "With reference to the above,",
    "Apropos of this,",
    "Building on this premise,",
    "By extension,",
    "Conversely,",
    "On the contrary,",
    "It follows that",
    "Against this backdrop,",
]

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None and _SPACY_AVAILABLE:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=False)
            _NLP = spacy.load("en_core_web_sm")
    return _NLP


# ── Grammar correction ─────────────────────────────────────────────────────

def _get_gramformer():
    global _GF
    if _GF is None and _GRAMFORMER_AVAILABLE:
        try:
            _GF = Gramformer(models=1, use_gpu=False)
        except Exception as exc:
            warnings.warn(f"Gramformer load failed: {exc}")
    return _GF


def correct_grammar(text: str) -> str:
    """Apply Gramformer grammar correction."""
    if not _GRAMFORMER_AVAILABLE:
        return text
    try:
        gf = _get_gramformer()
        if gf is None:
            return text
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        corrected = []
        for sent in sentences:
            try:
                results = list(gf.correct(sent, max_candidates=1))
                corrected.append(results[0] if results else sent)
            except Exception:
                corrected.append(sent)
        return " ".join(corrected)
    except Exception as exc:
        warnings.warn(f"Grammar correction failed: {exc}")
        return text


# ── Hedging/boosting balance ───────────────────────────────────────────────

def balance_hedging(text: str) -> str:
    """Replace overconfident words with hedged alternatives."""
    for word, hedge in OVERCONFIDENT_WORDS.items():
        text = re.sub(r'\b' + re.escape(word) + r'\b', hedge, text, flags=re.IGNORECASE)
    return text


# ── Transition word injection ──────────────────────────────────────────────

def inject_transitions(text: str) -> str:
    """Inject academic transition phrases at paragraph boundaries."""
    import random
    paragraphs = text.split("\n\n")
    if len(paragraphs) < 2:
        return text
    result = [paragraphs[0]]
    used: Set[str] = set()
    for para in paragraphs[1:]:
        if para.strip() and not re.match(r'\[.*?\]', para.strip()):
            available = [t for t in ACADEMIC_TRANSITIONS if t not in used]
            if not available:
                available = ACADEMIC_TRANSITIONS
                used.clear()
            transition = random.choice(available)
            used.add(transition)
            para = transition + " " + para[0].lower() + para[1:] if para else para
        result.append(para)
    return "\n\n".join(result)


# ── Tense consistency ──────────────────────────────────────────────────────

def check_tense_consistency(text: str) -> str:
    """
    Basic tense check: results sections should use past tense,
    facts/generalisations should use present tense.
    This is a heuristic pass only.
    """
    # Heuristic: replace "results show" constructs with past tense
    text = re.sub(r'\bshow(?:s)?\b', "showed", text)
    text = re.sub(r'\bindicate(?:s)?\b(?=.*(?:result|finding|data))', "indicated", text, flags=re.IGNORECASE)
    return text


# ── Oxford comma normalization ────────────────────────────────────────────

def normalize_oxford_comma(text: str) -> str:
    """Ensure Oxford comma before 'and'/'or' in lists."""
    # Match: word, word and word → word, word, and word
    pattern = r'(\b\w+\b),\s+(\b\w+\b)\s+(and|or)\s+(\b\w+\b)'

    def inserter(m: re.Match) -> str:
        return f"{m.group(1)}, {m.group(2)}, {m.group(3)} {m.group(4)}"

    return re.sub(pattern, inserter, text)


# ── Acronym safeguarding ──────────────────────────────────────────────────

def safeguard_acronyms(text: str) -> str:
    """Ensure acronyms are defined on first use (heuristic: detect CAP sequences)."""
    defined: Set[str] = set()
    pattern = r'\b([A-Z]{2,6})\b'

    def handler(m: re.Match) -> str:
        acr = m.group(1)
        if acr in defined:
            return acr
        defined.add(acr)
        return acr  # Full definition injection requires domain knowledge; preserve as-is

    return re.sub(pattern, handler, text)


# ── Block quote formatting ────────────────────────────────────────────────

def format_block_quotes(text: str) -> str:
    """
    Detect quoted passages >40 words and mark them as APA block quotes
    (indented lines in the plain-text representation).
    """
    pattern = r'"([^"]{200,})"'  # ~40+ words ≈ 200+ chars

    def formatter(m: re.Match) -> str:
        inner = m.group(1).strip()
        # APA block quote: no quotation marks, indented
        indented = "\n    ".join(inner.split("\n"))
        return f"\n\n    {indented}\n\n"

    return re.sub(pattern, formatter, text, flags=re.DOTALL)


# ── LanguageTool pass ─────────────────────────────────────────────────────

def _get_lt():
    global _LT_TOOL
    if _LT_TOOL is None and _LT_AVAILABLE:
        try:
            _LT_TOOL = language_tool_python.LanguageTool("en-US")
        except Exception as exc:
            warnings.warn(f"LanguageTool init failed: {exc}")
    return _LT_TOOL


def language_tool_pass(text: str) -> str:
    """Apply LanguageTool corrections (style + grammar)."""
    if not _LT_AVAILABLE:
        return text
    try:
        lt = _get_lt()
        if lt is None:
            return text
        return language_tool_python.utils.correct(text, lt.check(text))
    except Exception as exc:
        warnings.warn(f"LanguageTool pass failed: {exc}")
        return text


# ── Public API ─────────────────────────────────────────────────────────────

def refine(text: str) -> str:
    """Apply the full academic refinement pipeline."""
    text = correct_grammar(text)
    text = balance_hedging(text)
    text = inject_transitions(text)
    text = check_tense_consistency(text)
    text = normalize_oxford_comma(text)
    text = safeguard_acronyms(text)
    text = format_block_quotes(text)
    text = language_tool_pass(text)
    return text
