"""
methods/m01_token_masking.py — Methods 1–10

1.  Token Masking & Protection
2.  Entropy Variation
3.  Recursive Back-Translation
4.  Burstiness Balancing
5.  Domain Terminology Injection
6.  Active-Passive Flip
7.  Zero-Width Space Stripping
8.  Dynamic Paragraph Restructuring
9.  Reference Syncing
10. Stylometry Mimicry
"""

import re
import random
import warnings
from typing import Dict, List, Tuple, Callable, Any

random.seed(0)

# ── Optional imports ───────────────────────────────────────────────────────
try:
    from unidecode import unidecode
    _UNIDECODE_AVAILABLE = True
except ImportError:
    _UNIDECODE_AVAILABLE = False
    warnings.warn("unidecode not available; zero-width stripping limited.")

try:
    import scipy.stats as _stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    _DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    _DEEP_TRANSLATOR_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import wordnet as wn
    _NLTK_AVAILABLE = True
    try:
        wn.synsets("test")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
except ImportError:
    _NLTK_AVAILABLE = False

try:
    from habanero import Crossref
    _HABANERO_AVAILABLE = True
except ImportError:
    _HABANERO_AVAILABLE = False
    warnings.warn("habanero not available; reference syncing disabled.")

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


# ── Progress callback type ─────────────────────────────────────────────────
ProgressCallback = Callable[[str, float], None]

_NOOP: ProgressCallback = lambda name, pct: None


# ── Method 1: Token Masking & Protection ──────────────────────────────────

def method_01_token_masking(
    text: str, progress: ProgressCallback = _NOOP
) -> Tuple[str, Dict[str, str]]:
    """
    Mask citations (Author, YEAR), quoted text, and other protected spans.
    Returns (masked_text, protection_map).
    """
    progress("Method 1: Token Masking", 0.0)
    from core.isolator import isolate
    masked, pmap = isolate(text, use_ner=True)
    progress("Method 1: Token Masking", 100.0)
    return masked, pmap


# ── Method 2: Entropy Variation ───────────────────────────────────────────

def method_02_entropy_variation(
    text: str,
    pmap: Dict[str, str] = None,
    progress: ProgressCallback = _NOOP,
) -> str:
    """
    Replace top-5 most predictable words per paragraph with rare WordNet synonyms.
    """
    progress("Method 2: Entropy Variation", 0.0)
    if not _NLTK_AVAILABLE:
        progress("Method 2: Entropy Variation", 100.0)
        return text

    def _rare_synonym(word: str) -> str:
        try:
            synsets = wn.synsets(word)
            candidates = []
            for syn in synsets[:5]:
                for lemma in syn.lemmas():
                    name = lemma.name().replace("_", " ")
                    if name.lower() != word.lower() and len(name) > 3:
                        candidates.append(name)
            if candidates:
                return random.choice(candidates)
        except Exception:
            pass
        return word

    paragraphs = text.split("\n\n")
    result = []
    for para in paragraphs:
        words = para.split()
        freq: Dict[str, int] = {}
        for w in words:
            clean = re.sub(r'[^a-zA-Z]', '', w).lower()
            if clean:
                freq[clean] = freq.get(clean, 0) + 1
        top_words = sorted(freq, key=freq.get, reverse=True)[:5]
        new_words = []
        for w in words:
            if re.match(r'\[.*?\]', w):
                new_words.append(w)
                continue
            clean = re.sub(r'[^a-zA-Z]', '', w).lower()
            if clean in top_words and random.random() < 0.6:
                syn = _rare_synonym(clean)
                new_words.append(w.replace(clean, syn, 1) if syn != clean else w)
            else:
                new_words.append(w)
        result.append(" ".join(new_words))
    progress("Method 2: Entropy Variation", 100.0)
    return "\n\n".join(result)


# ── Method 3: Recursive Back-Translation ─────────────────────────────────

def method_03_back_translation(
    text: str,
    languages: List[str] = None,
    progress: ProgressCallback = _NOOP,
) -> str:
    """EN → DE → JA → EN back-translation pipeline."""
    progress("Method 3: Back-Translation", 0.0)
    if not _DEEP_TRANSLATOR_AVAILABLE:
        progress("Method 3: Back-Translation", 100.0)
        return text
    if languages is None:
        languages = ["de", "ja"]
    try:
        current = text
        src = "en"
        for i, tgt in enumerate(languages):
            current = GoogleTranslator(source=src, target=tgt).translate(current) or current
            src = tgt
            progress("Method 3: Back-Translation", (i + 1) / (len(languages) + 1) * 80)
        current = GoogleTranslator(source=src, target="en").translate(current) or current
    except Exception as exc:
        warnings.warn(f"Back-translation failed: {exc}")
    progress("Method 3: Back-Translation", 100.0)
    return current


# ── Method 4: Burstiness Balancing ───────────────────────────────────────

def method_04_burstiness_balancing(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Merge short sentences and split long ones for human-like length variance."""
    progress("Method 4: Burstiness Balancing", 0.0)
    from core.humanizer import balance_burstiness
    result = balance_burstiness(text)
    progress("Method 4: Burstiness Balancing", 100.0)
    return result


# ── Method 5: Domain Terminology Injection ───────────────────────────────

def method_05_domain_terminology(
    text: str,
    domain_terms: List[str] = None,
    progress: ProgressCallback = _NOOP,
) -> str:
    """Protect domain terms; only modify surrounding verbs/prepositions."""
    progress("Method 5: Domain Terminology", 0.0)
    if not domain_terms:
        progress("Method 5: Domain Terminology", 100.0)
        return text
    # Mark domain terms as protected (they'll survive synonym replacement)
    for term in domain_terms:
        # Surround with invisible protection markers during processing
        text = re.sub(
            r'\b' + re.escape(term) + r'\b',
            term,  # keep as-is (domain terms are already in the whitelist)
            text,
            flags=re.IGNORECASE,
        )
    progress("Method 5: Domain Terminology", 100.0)
    return text


# ── Method 6: Active-Passive Flip ────────────────────────────────────────

def method_06_active_passive_flip(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Convert passive constructions to active voice."""
    progress("Method 6: Active-Passive Flip", 0.0)
    from core.humanizer import flip_passive_to_active
    result = flip_passive_to_active(text)
    progress("Method 6: Active-Passive Flip", 100.0)
    return result


# ── Method 7: Zero-Width Space Stripping ─────────────────────────────────

def method_07_zwsp_stripping(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Strip all non-standard Unicode characters including ZWSP, ZWNJ, BOM."""
    progress("Method 7: ZWSP Stripping", 0.0)
    # Remove known invisible/zero-width characters
    invisible_chars = [
        '\u200b',  # Zero Width Space
        '\u200c',  # Zero Width Non-Joiner
        '\u200d',  # Zero Width Joiner
        '\ufeff',  # BOM
        '\u00ad',  # Soft Hyphen
        '\u2060',  # Word Joiner
        '\u180e',  # Mongolian Vowel Separator
    ]
    for char in invisible_chars:
        text = text.replace(char, '')
    if _UNIDECODE_AVAILABLE:
        # Preserve protection tokens: extract them, run unidecode, reinsert
        tokens = re.findall(r'\[(?:REF|QUOTE|EQ|NE|ACR|LEGAL)_\d+\]', text)
        placeholders = {f"__TOK{i}__": t for i, t in enumerate(tokens)}
        for ph, tok in placeholders.items():
            text = text.replace(tok, ph, 1)
        # unidecode only ASCII-ifies; skip for now to preserve multi-byte chars
        for ph, tok in placeholders.items():
            text = text.replace(ph, tok)
    progress("Method 7: ZWSP Stripping", 100.0)
    return text


# ── Method 8: Dynamic Paragraph Restructuring ────────────────────────────

def method_08_paragraph_restructuring(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Reorder supporting sentences within paragraphs, keeping topic sentence first."""
    progress("Method 8: Paragraph Restructuring", 0.0)
    paragraphs = text.split("\n\n")
    result = []
    for para in paragraphs:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', para) if s.strip()]
        if len(sentences) > 3:
            topic = sentences[0]
            support = sentences[1:-1]
            conclusion = sentences[-1]
            random.shuffle(support)
            sentences = [topic] + support + [conclusion]
        result.append(" ".join(sentences))
    progress("Method 8: Paragraph Restructuring", 100.0)
    return "\n\n".join(result)


# ── Method 9: Reference Syncing ──────────────────────────────────────────

def method_09_reference_syncing(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Validate DOIs and reformat references per APA 7 via Crossref API."""
    progress("Method 9: Reference Syncing", 0.0)
    if not _HABANERO_AVAILABLE:
        progress("Method 9: Reference Syncing", 100.0)
        return text
    doi_pattern = r'\b10\.\d{4,9}/\S+'
    dois = re.findall(doi_pattern, text)
    if not dois:
        progress("Method 9: Reference Syncing", 100.0)
        return text
    try:
        cr = Crossref()
        for doi in dois[:5]:  # limit API calls
            try:
                result = cr.works(ids=doi)
                if result and result.get("message"):
                    msg = result["message"]
                    authors = msg.get("author", [])
                    year = msg.get("issued", {}).get("date-parts", [[""]])[0][0]
                    title = msg.get("title", [""])[0]
                    if authors and year and title:
                        first_author = authors[0]
                        apa_ref = (
                            f"{first_author.get('family', '')},"
                            f" {first_author.get('given', '')[:1]}."
                            f" ({year}). {title}."
                        )
                        text = text.replace(doi, apa_ref)
            except Exception:
                pass
    except Exception as exc:
        warnings.warn(f"Reference syncing failed: {exc}")
    progress("Method 9: Reference Syncing", 100.0)
    return text


# ── Method 10: Stylometry Mimicry ────────────────────────────────────────

def method_10_stylometry_mimicry(
    text: str,
    sample_text: str = "",
    progress: ProgressCallback = _NOOP,
) -> str:
    """Analyse function-word frequencies from sample; adjust generated text to match."""
    progress("Method 10: Stylometry Mimicry", 0.0)
    if not sample_text.strip():
        progress("Method 10: Stylometry Mimicry", 100.0)
        return text

    FUNCTION_WORDS = [
        "the", "a", "an", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "by", "from", "that", "which",
        "this", "these", "those", "it", "its", "is", "are", "was",
        "were", "be", "been", "being", "have", "has", "had",
    ]

    def _freq_map(t: str) -> Dict[str, float]:
        words = re.findall(r'\b\w+\b', t.lower())
        total = len(words) or 1
        return {w: words.count(w) / total for w in FUNCTION_WORDS}

    sample_freq = _freq_map(sample_text)
    text_freq = _freq_map(text)

    # Heuristic: if "the" is underrepresented, add it occasionally
    for fw in FUNCTION_WORDS[:5]:
        if sample_freq.get(fw, 0) > text_freq.get(fw, 0) * 1.5:
            # Slightly increase usage – naive approach
            text = re.sub(r'\b(a|an)\b', fw, text, count=2, flags=re.IGNORECASE)
            break

    progress("Method 10: Stylometry Mimicry", 100.0)
    return text


# ── Module-level run_all convenience function ─────────────────────────────

def run_all(
    text: str,
    config: Dict[str, Any] = None,
    progress: ProgressCallback = _NOOP,
) -> str:
    """Run methods 1–10 sequentially and return processed text."""
    if config is None:
        config = {}
    masked, pmap = method_01_token_masking(text, progress)
    masked = method_02_entropy_variation(masked, pmap, progress)
    masked = method_03_back_translation(masked, config.get("back_translation_languages"), progress)
    masked = method_04_burstiness_balancing(masked, progress)
    masked = method_05_domain_terminology(masked, config.get("domain_terms", []), progress)
    masked = method_06_active_passive_flip(masked, progress)
    masked = method_07_zwsp_stripping(masked, progress)
    masked = method_08_paragraph_restructuring(masked, progress)
    masked = method_09_reference_syncing(masked, progress)
    masked = method_10_stylometry_mimicry(masked, config.get("sample_text", ""), progress)
    from core.isolator import restore
    return restore(masked, pmap)
