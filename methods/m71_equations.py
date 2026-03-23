"""
methods/m71_equations.py — Methods 71–80

71. Equation Cross-Verification
72. Equation Narrative Diversification
73. Self-Plagiarism Check Integration
74. Greek/Latin Symbol Normalization
75. Microsoft Equation Format
76. Explanatory Footnote Generation
77. Functional Word Scaling
78. Multiple Citation Ordering
79. Hidden Font Sanitization
80. Logic Consistency Test
"""

import re
import random
import difflib
import warnings
from typing import Dict, List, Callable, Any

random.seed(71)

try:
    from sentence_transformers import SentenceTransformer, util
    _ST_MODEL_INST = None
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    import numpy as np
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

ProgressCallback = Callable[[str, float], None]
_NOOP: ProgressCallback = lambda name, pct: None

_ST_MODEL = None

EQUATION_NARRATIVE_TEMPLATES = [
    "where {var} is",
    "the variable {var} denotes",
    "representing {var},",
    "{var} corresponds to",
    "in which {var} signifies",
    "{var} is defined as",
]

BRIDGE_PHRASES = [
    "This relationship can be understood as follows:",
    "To clarify the connection between these concepts,",
    "The link between the preceding and following arguments warrants elaboration.",
    "A brief explanation bridges these two observations.",
]


def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None and _ST_AVAILABLE:
        try:
            _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:
            warnings.warn(f"SentenceTransformer load failed: {exc}")
    return _ST_MODEL


# ── Method 71: Equation Cross-Verification ───────────────────────────────

def method_71_equation_verification(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Verify variable consistency throughout paper."""
    progress("Method 71: Equation Verification", 0.0)
    # Extract simple variable definitions
    definitions: Dict[str, List[str]] = {}
    pattern = r'\b([A-Za-z])\s*(?:=|denotes|represents|is defined as)\s+([^.,;]+)'
    for m in re.finditer(pattern, text):
        var = m.group(1)
        defn = m.group(2).strip()
        definitions.setdefault(var, []).append(defn)

    # Warn about inconsistent definitions
    for var, defns in definitions.items():
        if len(set(defns)) > 1:
            warnings.warn(f"Variable '{var}' has inconsistent definitions: {defns}")
    progress("Method 71: Equation Verification", 100.0)
    return text


# ── Method 72: Equation Narrative Diversification ────────────────────────

def method_72_equation_narrative(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Vary equation description language."""
    progress("Method 72: Equation Narrative", 0.0)
    # Replace repeated "where X is" with varied templates
    counter = [0]

    def vary_description(m: re.Match) -> str:
        var = m.group(1)
        template = EQUATION_NARRATIVE_TEMPLATES[counter[0] % len(EQUATION_NARRATIVE_TEMPLATES)]
        counter[0] += 1
        return template.format(var=var)

    text = re.sub(r'where\s+([A-Za-z])\s+is', vary_description, text)
    progress("Method 72: Equation Narrative", 100.0)
    return text


# ── Method 73: Self-Plagiarism Check ─────────────────────────────────────

def method_73_self_plagiarism_check(
    text: str,
    reference_texts: List[str] = None,
    threshold: float = 0.7,
    progress: ProgressCallback = _NOOP,
) -> str:
    """Detect self-similar passages and rewrite via difflib."""
    progress("Method 73: Self-Plagiarism Check", 0.0)
    if not reference_texts:
        progress("Method 73: Self-Plagiarism Check", 100.0)
        return text

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    all_ref_sentences = []
    for ref in reference_texts:
        all_ref_sentences.extend([s.strip() for s in re.split(r'(?<=[.!?])\s+', ref) if s.strip()])

    result = []
    for sent in sentences:
        similar = difflib.get_close_matches(sent, all_ref_sentences, n=1, cutoff=threshold)
        if similar:
            # Simple rewrite: shuffle words slightly
            words = sent.split()
            if len(words) > 5:
                mid = len(words) // 2
                rewritten = words[:2] + words[2:mid][::-1] + words[mid:]
                result.append(" ".join(rewritten))
            else:
                result.append(sent)
        else:
            result.append(sent)

    progress("Method 73: Self-Plagiarism Check", 100.0)
    return " ".join(result)


# ── Method 74: Greek/Latin Symbol Normalization ───────────────────────────

def method_74_symbol_normalization(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Ensure α, β, Δ are mathematical symbols not text characters."""
    progress("Method 74: Symbol Normalization", 0.0)
    # Map common text representations to proper Unicode mathematical symbols
    normalizations = {
        r'\balpha\b': 'α',
        r'\bbeta\b': 'β',
        r'\bgamma\b': 'γ',
        r'\bdelta\b': 'δ',
        r'\bDelta\b': 'Δ',
        r'\bsigma\b': 'σ',
        r'\bSigma\b': 'Σ',
        r'\bmu\b(?!\w)': 'μ',
        r'\bpi\b(?!\w)': 'π',
        r'\btheta\b': 'θ',
        r'\blambda\b': 'λ',
    }
    for pattern, symbol in normalizations.items():
        text = re.sub(pattern, symbol, text)
    progress("Method 74: Symbol Normalization", 100.0)
    return text


# ── Method 75: Microsoft Equation Format ─────────────────────────────────

def method_75_equation_format(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Centre-align equations and add right-side numbering."""
    progress("Method 75: Equation Format", 0.0)
    eq_counter = [0]

    def format_eq(m: re.Match) -> str:
        eq_counter[0] += 1
        eq_text = m.group(0).strip()
        return f"\n\n    {eq_text}    (Eq. {eq_counter[0]})\n\n"

    text = re.sub(r'\$\$([^$]+)\$\$', format_eq, text, flags=re.DOTALL)
    text = re.sub(r'\\\[([^\]]+)\\\]', format_eq, text, flags=re.DOTALL)
    progress("Method 75: Equation Format", 100.0)
    return text


# ── Method 76: Explanatory Footnote Generation ───────────────────────────

def method_76_footnote_generation(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Detect rare technical terms; suggest footnote definitions."""
    progress("Method 76: Footnote Generation", 0.0)
    # Heuristic: terms that are longer than 12 chars and not in common vocabulary
    long_terms = re.findall(r'\b[A-Za-z]{12,}\b', text)
    footnotes = []
    fn_counter = [0]
    defined: set = set()

    for term in long_terms:
        if term.lower() not in defined:
            fn_counter[0] += 1
            defined.add(term.lower())
            footnotes.append(f"{fn_counter[0]}. {term}: A technical term used in this context.")
            # Add footnote marker after first occurrence
            text = re.sub(r'\b' + re.escape(term) + r'\b', term + f"[{fn_counter[0]}]", text, count=1)

    if footnotes:
        text += "\n\n---\n" + "\n".join(footnotes)
    progress("Method 76: Footnote Generation", 100.0)
    return text


# ── Method 77: Functional Word Scaling ───────────────────────────────────

def method_77_functional_word_scaling(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Adjust and/but/of ratios to match expert researcher profile."""
    progress("Method 77: Functional Word Scaling", 0.0)
    # Target ratios based on academic writing norms
    TARGET_RATIOS = {"and": 0.05, "the": 0.07, "of": 0.04, "in": 0.03}

    words = text.split()
    total = len(words) or 1
    current_ratios = {fw: words.count(fw) / total for fw in TARGET_RATIOS}

    for fw, target in TARGET_RATIOS.items():
        current = current_ratios.get(fw, 0)
        if current > target * 1.5:
            # Reduce occurrences
            excess = int((current - target) * total)
            reduced = 0
            result_words = []
            for w in words:
                if w.lower() == fw and reduced < excess and random.random() < 0.3:
                    result_words.append("")  # remove
                    reduced += 1
                else:
                    result_words.append(w)
            words = result_words
            text = " ".join(w for w in words if w)

    progress("Method 77: Functional Word Scaling", 100.0)
    return text


# ── Method 78: Multiple Citation Ordering ────────────────────────────────

def method_78_citation_ordering(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Sort multiple citations in one bracket chronologically."""
    progress("Method 78: Citation Ordering", 0.0)

    def sort_citations(m: re.Match) -> str:
        inner = m.group(1)
        # Split by semicolon
        parts = [p.strip() for p in inner.split(';')]
        # Extract year for sorting
        def get_year(part):
            year_match = re.search(r'\d{4}', part)
            return int(year_match.group()) if year_match else 9999
        parts_sorted = sorted(parts, key=get_year)
        return '(' + '; '.join(parts_sorted) + ')'

    text = re.sub(r'\(([^)]+;\s*[^)]+)\)', sort_citations, text)
    progress("Method 78: Citation Ordering", 100.0)
    return text


# ── Method 79: Hidden Font Sanitization ──────────────────────────────────

def method_79_font_sanitization(
    docx_path: str = None, progress: ProgressCallback = _NOOP
) -> bool:
    """Scan .docx for invisible 1px fonts and unify to Times New Roman."""
    progress("Method 79: Font Sanitization", 0.0)
    if docx_path is None:
        progress("Method 79: Font Sanitization", 100.0)
        return True
    try:
        from docx import Document
        from docx.shared import Pt
        doc = Document(docx_path)
        for para in doc.paragraphs:
            for run in para.runs:
                if run.font.size is not None and run.font.size < Pt(2):
                    run.font.size = Pt(12)
                    run.font.name = "Times New Roman"
        doc.save(docx_path)
        progress("Method 79: Font Sanitization", 100.0)
        return True
    except Exception as exc:
        warnings.warn(f"Font sanitization failed: {exc}")
        progress("Method 79: Font Sanitization", 100.0)
        return False


# ── Method 80: Logic Consistency Test ────────────────────────────────────

def method_80_logic_consistency(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Measure semantic similarity between consecutive sentences; inject bridge."""
    progress("Method 80: Logic Consistency", 0.0)
    model = _get_st_model()
    if model is None:
        progress("Method 80: Logic Consistency", 100.0)
        return text
    try:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if len(sentences) < 2:
            progress("Method 80: Logic Consistency", 100.0)
            return text
        embeddings = model.encode(sentences)
        result = [sentences[0]]
        for i in range(1, len(sentences)):
            sim = float(util.cos_sim(embeddings[i - 1], embeddings[i]))
            if sim < 0.15:  # Large semantic gap
                bridge = random.choice(BRIDGE_PHRASES)
                result.append(bridge)
            result.append(sentences[i])
        progress("Method 80: Logic Consistency", 100.0)
        return " ".join(result)
    except Exception as exc:
        warnings.warn(f"Logic consistency test failed: {exc}")
        progress("Method 80: Logic Consistency", 100.0)
        return text


# ── Module convenience function ───────────────────────────────────────────

def run_all(
    text: str,
    config: Dict[str, Any] = None,
    progress: ProgressCallback = _NOOP,
) -> str:
    if config is None:
        config = {}
    text = method_71_equation_verification(text, progress)
    text = method_72_equation_narrative(text, progress)
    text = method_73_self_plagiarism_check(text, config.get("reference_texts", []), progress=progress)
    text = method_74_symbol_normalization(text, progress)
    text = method_75_equation_format(text, progress)
    text = method_77_functional_word_scaling(text, progress)
    text = method_78_citation_ordering(text, progress)
    text = method_80_logic_consistency(text, progress)
    return text
