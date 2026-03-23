"""
methods/m11_ref_freeze.py — Methods 11–20

11. Reference Freezing
12. Lexical Diversity Balancing
13. Bullet-Point Professionalization
14. Syntactic Academic Mimicry
15. Direct Quote Shielding
16. Automatic Style Alignment
17. Mathematical Symbology Preservation
18. Cross-Paragraph Cohesion
19. Contextual Grammar Correction
20. Cliché Neutralization
"""

import re
import random
import warnings
from typing import Dict, List, Callable, Any

random.seed(11)

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
    from wordfreq import word_frequency
    _WORDFREQ_AVAILABLE = True
except ImportError:
    _WORDFREQ_AVAILABLE = False
    warnings.warn("wordfreq not available; lexical diversity balancing limited.")

try:
    import language_tool_python
    _LT_TOOL = None
    _LT_AVAILABLE = True
except ImportError:
    _LT_AVAILABLE = False

ProgressCallback = Callable[[str, float], None]
_NOOP: ProgressCallback = lambda name, pct: None

# ── AI-overused words with academic replacements ──────────────────────────
AI_OVERUSED: Dict[str, str] = {
    "moreover": "in addition",
    "furthermore": "additionally",
    "transformative": "consequential",
    "comprehensive": "extensive",
    "innovative": "novel",
    "leveraging": "utilising",
    "robust": "reliable",
    "seamless": "efficient",
    "groundbreaking": "significant",
    "paradigm": "framework",
    "holistic": "integrated",
    "synergy": "collaboration",
    "utilize": "use",
    "facilitate": "enable",
    "implement": "apply",
}

ACADEMIC_TRANSITIONS = [
    "Notwithstanding the above,",
    "By extension,",
    "In light of this,",
    "Building on this premise,",
    "Against this backdrop,",
    "Apropos of this,",
    "It follows, therefore,",
]

ACTION_VERBS = [
    "Demonstrates", "Highlights", "Reveals", "Illustrates",
    "Identifies", "Presents", "Outlines", "Examines",
    "Provides", "Discusses", "Explains", "Describes",
]


# ── Method 11: Reference Freezing ────────────────────────────────────────

def method_11_reference_freezing(
    text: str,
    pmap: Dict[str, str] = None,
    progress: ProgressCallback = _NOOP,
) -> str:
    """Mark author names + years as immutable entities."""
    progress("Method 11: Reference Freezing", 0.0)
    if pmap is None:
        pmap = {}
    pattern = r'\(([A-Z][A-Za-z\-]+(?:\s+et\s+al\.?)?,\s*\d{4}[a-z]?)\)'

    def freezer(m: re.Match) -> str:
        tok = f"[FROZEN_{len(pmap) + 1}]"
        pmap[tok] = m.group(0)
        return tok

    result = re.sub(pattern, freezer, text)
    progress("Method 11: Reference Freezing", 100.0)
    return result


# ── Method 12: Lexical Diversity Balancing ───────────────────────────────

def method_12_lexical_diversity(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Replace AI-overused words with rarer academic synonyms."""
    progress("Method 12: Lexical Diversity", 0.0)
    for overused, replacement in AI_OVERUSED.items():
        text = re.sub(r'\b' + re.escape(overused) + r'\b', replacement, text, flags=re.IGNORECASE)

    # wordfreq-based replacement for high-frequency academic filler
    if _WORDFREQ_AVAILABLE and _NLTK_AVAILABLE:
        words = text.split()
        result = []
        for w in words:
            if re.match(r'\[.*?\]', w):
                result.append(w)
                continue
            clean = re.sub(r'[^a-zA-Z]', '', w).lower()
            if len(clean) > 4:
                freq = word_frequency(clean, 'en')
                if freq > 0.001 and random.random() < 0.1:
                    syns = [
                        l.name().replace("_", " ")
                        for s in wn.synsets(clean)[:3]
                        for l in s.lemmas()
                        if l.name().lower() != clean
                    ]
                    rare = [s for s in syns if _WORDFREQ_AVAILABLE and word_frequency(s, 'en') < freq]
                    if rare:
                        w = w.replace(clean, random.choice(rare), 1)
            result.append(w)
        text = " ".join(result)
    progress("Method 12: Lexical Diversity", 100.0)
    return text


# ── Method 13: Bullet-Point Professionalization ───────────────────────────

def method_13_bullet_professionalization(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Rewrite each bullet to start with a different action verb."""
    progress("Method 13: Bullet Professionalization", 0.0)
    lines = text.split("\n")
    result = []
    verb_cycle = iter(ACTION_VERBS * 10)
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(("•", "-", "*", "–")) and len(stripped) > 3:
            bullet_char = stripped[0]
            content = stripped[1:].strip()
            # Remove existing leading verb if present
            content = re.sub(r'^[A-Z][a-z]+s?\s+', '', content)
            verb = next(verb_cycle)
            result.append(f"{line[:len(line)-len(stripped)]}{bullet_char} {verb} {content[0].lower()}{content[1:]}")
        else:
            result.append(line)
    progress("Method 13: Bullet Professionalization", 100.0)
    return "\n".join(result)


# ── Method 14: Syntactic Academic Mimicry ────────────────────────────────

def method_14_syntactic_mimicry(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Increase use of relative clauses and parenthetical expressions."""
    progress("Method 14: Syntactic Mimicry", 0.0)
    # Add parenthetical clarifications to some sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    result = []
    for i, sent in enumerate(sentences):
        if i % 5 == 0 and len(sent.split()) > 10 and not re.search(r'\[.*?\]', sent):
            # Add a parenthetical
            words = sent.split()
            mid = len(words) // 2
            words.insert(mid, "(i.e., as previously noted)")
            result.append(" ".join(words))
        else:
            result.append(sent)
    progress("Method 14: Syntactic Mimicry", 100.0)
    return " ".join(result)


# ── Method 15: Direct Quote Shielding ────────────────────────────────────

def method_15_quote_shielding(
    text: str,
    pmap: Dict[str, str] = None,
    progress: ProgressCallback = _NOOP,
) -> str:
    """Detect "..." spans and apply pass-through rule."""
    progress("Method 15: Quote Shielding", 0.0)
    if pmap is None:
        pmap = {}
    counter = [len(pmap)]

    def shield(m: re.Match) -> str:
        counter[0] += 1
        tok = f"[SHIELD_{counter[0]}]"
        pmap[tok] = m.group(0)
        return tok

    result = re.sub(r'"[^"]{10,500}"', shield, text)
    progress("Method 15: Quote Shielding", 100.0)
    return result


# ── Method 16: Automatic Style Alignment ─────────────────────────────────

def method_16_style_alignment(
    text: str,
    style: str = "APA7",
    progress: ProgressCallback = _NOOP,
) -> str:
    """Apply style-specific text conventions."""
    progress("Method 16: Style Alignment", 0.0)
    if style.upper() in ("APA7", "APA"):
        # APA: numbers below 10 spelled out
        for i in range(1, 10):
            text = re.sub(rf'\b{i}\b(?!\s*%|\s*cm|\s*kg|\s*mm)', _NUM_WORDS.get(i, str(i)), text)
    elif style.upper() == "IEEE":
        # IEEE: use numerals throughout
        for word, numeral in _WORD_NUMS.items():
            text = re.sub(rf'\b{word}\b', str(numeral), text, flags=re.IGNORECASE)
    progress("Method 16: Style Alignment", 100.0)
    return text


_NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
              6: "six", 7: "seven", 8: "eight", 9: "nine"}
_WORD_NUMS = {v: k for k, v in _NUM_WORDS.items()}


# ── Method 17: Mathematical Symbology Preservation ───────────────────────

def method_17_math_preservation(
    text: str,
    pmap: Dict[str, str] = None,
    progress: ProgressCallback = _NOOP,
) -> str:
    """Detect math expressions and protect them during processing."""
    progress("Method 17: Math Preservation", 0.0)
    if pmap is None:
        pmap = {}
    counter = [len(pmap)]

    def protect(m: re.Match) -> str:
        counter[0] += 1
        tok = f"[MATH_{counter[0]}]"
        pmap[tok] = m.group(0)
        return tok

    # Protect LaTeX math
    text = re.sub(r'\$\$.*?\$\$', protect, text, flags=re.DOTALL)
    text = re.sub(r'\$[^$\n]+?\$', protect, text)
    # Protect Unicode math symbols
    text = re.sub(r'[α-ωΑ-Ω∑∏∂∇∫∞±×÷≤≥≠≈∈∉⊂⊃∪∩]+', protect, text)
    progress("Method 17: Math Preservation", 100.0)
    return text


# ── Method 18: Cross-Paragraph Cohesion ──────────────────────────────────

def method_18_cross_paragraph_cohesion(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Inject transition words at paragraph boundaries."""
    progress("Method 18: Cross-Paragraph Cohesion", 0.0)
    from core.academic_refiner import inject_transitions
    result = inject_transitions(text)
    progress("Method 18: Cross-Paragraph Cohesion", 100.0)
    return result


# ── Method 19: Contextual Grammar Correction ─────────────────────────────

def method_19_grammar_correction(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Correct grammar with preference for formal academic suggestions."""
    progress("Method 19: Grammar Correction", 0.0)
    if not _LT_AVAILABLE:
        progress("Method 19: Grammar Correction", 100.0)
        return text
    try:
        if _LT_TOOL is None:
            lt = language_tool_python.LanguageTool("en-US")
        else:
            lt = _LT_TOOL
        text = language_tool_python.utils.correct(text, lt.check(text))
    except Exception as exc:
        warnings.warn(f"Grammar correction failed: {exc}")
    progress("Method 19: Grammar Correction", 100.0)
    return text


# ── Method 20: Cliché Neutralization ─────────────────────────────────────

def method_20_cliche_neutralization(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Detect AI clichés and replace with academic alternatives."""
    progress("Method 20: Cliché Neutralization", 0.0)
    from core.humanizer import neutralize_cliches
    result = neutralize_cliches(text)
    progress("Method 20: Cliché Neutralization", 100.0)
    return result


# ── Module convenience function ───────────────────────────────────────────

def run_all(
    text: str,
    config: Dict[str, Any] = None,
    progress: ProgressCallback = _NOOP,
) -> str:
    if config is None:
        config = {}
    pmap: Dict[str, str] = {}
    text = method_11_reference_freezing(text, pmap, progress)
    text = method_12_lexical_diversity(text, progress)
    text = method_13_bullet_professionalization(text, progress)
    text = method_14_syntactic_mimicry(text, progress)
    text = method_15_quote_shielding(text, pmap, progress)
    text = method_16_style_alignment(text, config.get("citation_style", "APA7"), progress)
    text = method_17_math_preservation(text, pmap, progress)
    text = method_18_cross_paragraph_cohesion(text, progress)
    text = method_19_grammar_correction(text, progress)
    text = method_20_cliche_neutralization(text, progress)
    from core.isolator import restore
    return restore(text, pmap)
