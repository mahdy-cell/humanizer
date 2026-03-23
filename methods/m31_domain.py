"""
methods/m31_domain.py — Methods 31–40

31. Domain-Term Anchoring
32. Tabular Content Humanization
33. Cross-Lingual Pivot
34. Logical Flow Optimization
35. Footnote Integration & Management
36. Hedges and Boosters Balancing
37. Block Quote Formatting
38. Text-Graph Synchronicity
39. Named Entity Recognition
40. Abstract Structural Reform
"""

import re
import random
import warnings
from typing import Dict, List, Callable, Any

random.seed(31)

try:
    from keybert import KeyBERT
    _KEYBERT_AVAILABLE = True
except ImportError:
    _KEYBERT_AVAILABLE = False
    warnings.warn("keybert not available; domain-term anchoring limited.")

try:
    from deep_translator import GoogleTranslator
    _DT_AVAILABLE = True
except ImportError:
    _DT_AVAILABLE = False

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

ProgressCallback = Callable[[str, float], None]
_NOOP: ProgressCallback = lambda name, pct: None

CAUSAL_PHRASES = [
    "consequently", "it can be inferred that",
    "as a result of", "this implies", "it follows that",
    "which suggests", "thereby", "thus demonstrating",
]

HEDGING_MAP = {
    "always": "typically",
    "never": "rarely",
    "certainly": "presumably",
    "proves": "suggests",
    "is clear": "appears",
    "demonstrates": "suggests",
    "confirms": "implies",
}

_NLP = None
_KB_MODEL = None


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


def _get_keybert():
    global _KB_MODEL
    if _KB_MODEL is None and _KEYBERT_AVAILABLE:
        try:
            _KB_MODEL = KeyBERT()
        except Exception as exc:
            warnings.warn(f"KeyBERT load failed: {exc}")
    return _KB_MODEL


# ── Method 31: Domain-Term Anchoring ─────────────────────────────────────

def method_31_domain_anchoring(
    text: str,
    domain_terms: List[str] = None,
    progress: ProgressCallback = _NOOP,
) -> tuple:
    """Extract domain terms via KeyBERT and add to whitelist."""
    progress("Method 31: Domain Anchoring", 0.0)
    if domain_terms is None:
        domain_terms = []

    kb = _get_keybert()
    if kb:
        try:
            keywords = kb.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=15)
            for kw, score in keywords:
                if score > 0.35 and kw not in domain_terms:
                    domain_terms.append(kw)
        except Exception as exc:
            warnings.warn(f"KeyBERT extraction failed: {exc}")

    progress("Method 31: Domain Anchoring", 100.0)
    return text, domain_terms


# ── Method 32: Tabular Content Humanization ──────────────────────────────

def method_32_table_humanization(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Rewrite table captions as inferential statements."""
    progress("Method 32: Table Humanization", 0.0)
    INFERENTIAL_PREFIXES = [
        "The data illustrates",
        "The results reveal",
        "As demonstrated by the table,",
        "The findings indicate",
        "A notable pattern emerges:",
    ]

    def rewrite_caption(m: re.Match) -> str:
        table_num = m.group(1)
        caption = m.group(2).strip()
        prefix = random.choice(INFERENTIAL_PREFIXES)
        caption_lower = caption[0].lower() + caption[1:] if caption else caption
        return f"Table {table_num}. {prefix} {caption_lower}"

    text = re.sub(r'Table\s+(\d+)[.:]\s*(.+?)(?=\n|$)', rewrite_caption, text)
    progress("Method 32: Table Humanization", 100.0)
    return text


# ── Method 33: Cross-Lingual Pivot ───────────────────────────────────────

def method_33_cross_lingual_pivot(
    text: str,
    pivot: str = "fr",
    progress: ProgressCallback = _NOOP,
) -> str:
    """Single-language pivot for precise synonym selection."""
    progress("Method 33: Cross-Lingual Pivot", 0.0)
    if not _DT_AVAILABLE:
        progress("Method 33: Cross-Lingual Pivot", 100.0)
        return text
    try:
        pivoted = GoogleTranslator(source="en", target=pivot).translate(text) or text
        back = GoogleTranslator(source=pivot, target="en").translate(pivoted) or text
        progress("Method 33: Cross-Lingual Pivot", 100.0)
        return back
    except Exception as exc:
        warnings.warn(f"Cross-lingual pivot failed: {exc}")
        progress("Method 33: Cross-Lingual Pivot", 100.0)
        return text


# ── Method 34: Logical Flow Optimization ─────────────────────────────────

def method_34_logical_flow(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Rewrite results sentences with varied causal relations."""
    progress("Method 34: Logical Flow", 0.0)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    result = []
    for sent in sentences:
        if re.search(r'\b(result|finding|outcome|show|indicate)\b', sent, re.IGNORECASE):
            if not re.search(r'\b(' + '|'.join(CAUSAL_PHRASES[:3]) + r')\b', sent, re.IGNORECASE):
                phrase = random.choice(CAUSAL_PHRASES)
                sent = phrase.capitalize() + ", " + sent[0].lower() + sent[1:]
        result.append(sent)
    progress("Method 34: Logical Flow", 100.0)
    return " ".join(result)


# ── Method 35: Footnote Integration ──────────────────────────────────────

def method_35_footnote_integration(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Ensure sequential footnote numbering and humanize footnote text."""
    progress("Method 35: Footnote Integration", 0.0)
    counter = [0]

    def renumber(m: re.Match) -> str:
        counter[0] += 1
        return f"[{counter[0]}]"

    # Renumber [1], [2], ... style footnotes
    text = re.sub(r'\[\d+\](?!\s*http)', renumber, text)
    progress("Method 35: Footnote Integration", 100.0)
    return text


# ── Method 36: Hedges and Boosters Balancing ─────────────────────────────

def method_36_hedging_balance(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Detect overconfident sentences and inject hedging words."""
    progress("Method 36: Hedging Balance", 0.0)
    for overconfident, hedge in HEDGING_MAP.items():
        text = re.sub(r'\b' + re.escape(overconfident) + r'\b', hedge, text, flags=re.IGNORECASE)
    progress("Method 36: Hedging Balance", 100.0)
    return text


# ── Method 37: Block Quote Formatting ────────────────────────────────────

def method_37_block_quote_formatting(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Detect paragraphs >40 words between quotes and apply APA block quote style."""
    progress("Method 37: Block Quote Formatting", 0.0)
    from core.academic_refiner import format_block_quotes
    result = format_block_quotes(text)
    progress("Method 37: Block Quote Formatting", 100.0)
    return result


# ── Method 38: Text-Graph Synchronicity ──────────────────────────────────

def method_38_text_graph_sync(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Compare figure/table captions in text vs labels; flag mismatches."""
    progress("Method 38: Text-Graph Sync", 0.0)
    fig_refs = re.findall(r'Figure\s+(\d+)', text, re.IGNORECASE)
    fig_defs = re.findall(r'Fig(?:ure)?\.?\s+(\d+)[.:]', text, re.IGNORECASE)
    ref_set = set(fig_refs)
    def_set = set(fig_defs)
    missing = ref_set - def_set
    if missing:
        warnings.warn(f"Figure(s) referenced but not defined: {missing}")
    progress("Method 38: Text-Graph Sync", 100.0)
    return text


# ── Method 39: Named Entity Recognition ──────────────────────────────────

def method_39_ner_protection(
    text: str,
    pmap: Dict[str, str] = None,
    progress: ProgressCallback = _NOOP,
) -> str:
    """Protect ORG, GPE, PERSON entities from synonym replacement."""
    progress("Method 39: NER Protection", 0.0)
    if pmap is None:
        pmap = {}
    if not _SPACY_AVAILABLE:
        progress("Method 39: NER Protection", 100.0)
        return text
    try:
        nlp = _get_nlp()
        if nlp is None:
            progress("Method 39: NER Protection", 100.0)
            return text
        doc = nlp(text)
        offset = 0
        result = list(text)
        for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
            if ent.label_ in {"ORG", "GPE", "PERSON", "LOC"}:
                tok = f"[NE_{len(pmap)+1}]"
                pmap[tok] = ent.text
                result[ent.start_char:ent.end_char] = list(tok)
        progress("Method 39: NER Protection", 100.0)
        return "".join(result)
    except Exception as exc:
        warnings.warn(f"NER protection failed: {exc}")
        progress("Method 39: NER Protection", 100.0)
        return text


# ── Method 40: Abstract Structural Reform ────────────────────────────────

def method_40_abstract_reform(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Rebuild AI abstract in IMRAD structure (Problem/Method/Results/Conclusion)."""
    progress("Method 40: Abstract Reform", 0.0)
    # Only operate on text that looks like an abstract (< 400 words)
    word_count = len(text.split())
    if word_count > 400:
        progress("Method 40: Abstract Reform", 100.0)
        return text
    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.lex_rank import LexRankSummarizer

        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        sentences_raw = [str(s) for s in summarizer(parser.document, 4)]

        labels = ["This study addresses", "The methodology involves", "The results indicate", "In conclusion,"]
        rebuilt = []
        for label, sent in zip(labels, sentences_raw):
            rebuilt.append(f"{label} {sent[0].lower()}{sent[1:]}")
        result = " ".join(rebuilt)
        progress("Method 40: Abstract Reform", 100.0)
        return result
    except Exception as exc:
        warnings.warn(f"Abstract reform failed: {exc}")
        progress("Method 40: Abstract Reform", 100.0)
        return text


# ── Module convenience function ───────────────────────────────────────────

def run_all(
    text: str,
    config: Dict[str, Any] = None,
    progress: ProgressCallback = _NOOP,
) -> str:
    if config is None:
        config = {}
    domain_terms = config.get("domain_terms", [])
    text, domain_terms = method_31_domain_anchoring(text, domain_terms, progress)
    text = method_32_table_humanization(text, progress)
    text = method_34_logical_flow(text, progress)
    text = method_35_footnote_integration(text, progress)
    text = method_36_hedging_balance(text, progress)
    text = method_37_block_quote_formatting(text, progress)
    text = method_38_text_graph_sync(text, progress)
    text = method_40_abstract_reform(text, progress)
    return text
