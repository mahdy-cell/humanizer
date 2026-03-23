"""
methods/m81_structure.py — Methods 81–90

81. Dynamic TOC Generation
82. Triple-Layer Linguistic Review
83. Appendices Formatting
84. Reporting Verbs Modulation
85. XML Structure Sanitization
86. Intersentential Cohesion
87. Translated Quote Handling
88. Stylistic Micro-Variation Simulation
89. Bibliography Styling
90. Live Link Verification
"""

import re
import random
import warnings
from typing import Dict, List, Callable, Any

random.seed(81)

try:
    import requests
    from bs4 import BeautifulSoup
    _REQUESTS_AVAILABLE = True
    _BS4_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False
    _BS4_AVAILABLE = False

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

ProgressCallback = Callable[[str, float], None]
_NOOP: ProgressCallback = lambda name, pct: None

TENTATIVE_VERBS = ["suggests", "implies", "appears to show", "indicates", "may demonstrate"]
CRITICAL_VERBS = ["demonstrates", "establishes", "proves", "confirms", "shows definitively"]

MICRO_VARIATIONS = [
    (" which ", " that "),
    (" the data shows", " the data reveal"),
    (" analyse", " analyze"),
    (" colour", " color"),
    (" realise", " realize"),
]

WAYBACK_API = "https://archive.org/wayback/available?url={url}"


# ── Method 81: Dynamic TOC Generation ────────────────────────────────────

def method_81_toc_generation(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Build TOC from Heading patterns in plain text."""
    progress("Method 81: TOC Generation", 0.0)
    headings = []
    lines = text.split("\n")
    for line in lines:
        # Match: "1. Introduction", "1.1 Background", "## Methods"
        m = re.match(r'^((?:\d+\.)+\d*\s+|#{1,3}\s+)(.+)', line.strip())
        if m:
            level = m.group(1).count('.') if '.' in m.group(1) else m.group(1).count('#')
            title = m.group(2).strip()
            headings.append((level, title))

    if not headings:
        progress("Method 81: TOC Generation", 100.0)
        return text

    toc_lines = ["TABLE OF CONTENTS\n"]
    for level, title in headings:
        indent = "    " * max(level - 1, 0)
        toc_lines.append(f"{indent}{title}")
    toc = "\n".join(toc_lines) + "\n\n"
    progress("Method 81: TOC Generation", 100.0)
    return toc + text


# ── Method 82: Triple-Layer Linguistic Review ─────────────────────────────

def method_82_triple_layer_review(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Sequential: LanguageTool → Gramformer → Proselint."""
    progress("Method 82: Triple-Layer Review", 0.0)
    try:
        import language_tool_python
        lt = language_tool_python.LanguageTool("en-US")
        text = language_tool_python.utils.correct(text, lt.check(text))
        progress("Method 82: Triple-Layer Review", 33.0)
    except Exception as exc:
        warnings.warn(f"LanguageTool layer failed: {exc}")

    try:
        from gramformer import Gramformer
        gf = Gramformer(models=1, use_gpu=False)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        corrected = []
        for sent in sentences:
            try:
                results = list(gf.correct(sent, max_candidates=1))
                corrected.append(results[0] if results else sent)
            except Exception:
                corrected.append(sent)
        text = " ".join(corrected)
        progress("Method 82: Triple-Layer Review", 66.0)
    except Exception as exc:
        warnings.warn(f"Gramformer layer failed: {exc}")

    try:
        import proselint
        suggestions = proselint.tools.lint(text)
        for check, message, line, col, start, end, severity, replacements, *_ in suggestions:
            if replacements and start < len(text):
                text = text[:start] + replacements[0] + text[end:]
    except Exception as exc:
        warnings.warn(f"Proselint layer failed: {exc}")

    progress("Method 82: Triple-Layer Review", 100.0)
    return text


# ── Method 83: Appendices Formatting ─────────────────────────────────────

def method_83_appendices_formatting(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Format appendices with A-1/A-2 numbering."""
    progress("Method 83: Appendices Formatting", 0.0)
    appendix_counter = [0]

    def format_appendix(m: re.Match) -> str:
        appendix_counter[0] += 1
        title = m.group(1).strip() if m.group(1) else f"Appendix {appendix_counter[0]}"
        return f"\nAPPENDIX {chr(64 + appendix_counter[0])}: {title.upper()}\n"

    text = re.sub(r'(?i)^appendix\s*[A-Z]?\s*:?\s*(.*?)$', format_appendix, text, flags=re.MULTILINE)
    progress("Method 83: Appendices Formatting", 100.0)
    return text


# ── Method 84: Reporting Verbs Modulation ────────────────────────────────

def method_84_reporting_verbs_modulation(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Use tentative verbs for weak sources; critical verbs for strong sources."""
    progress("Method 84: Reporting Verbs Modulation", 0.0)
    if not _TEXTBLOB_AVAILABLE:
        progress("Method 84: Reporting Verbs Modulation", 100.0)
        return text

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    result = []
    reporting_pattern = r'\b(argues|states|claims|shows|suggests|demonstrates|finds|reports)\b'
    for sent in sentences:
        try:
            blob = TextBlob(sent)
            polarity = blob.sentiment.polarity
            if re.search(reporting_pattern, sent, re.IGNORECASE):
                if polarity < 0.1:
                    replacement = random.choice(TENTATIVE_VERBS)
                else:
                    replacement = random.choice(CRITICAL_VERBS)
                sent = re.sub(reporting_pattern, replacement, sent, count=1, flags=re.IGNORECASE)
        except Exception:
            pass
        result.append(sent)
    progress("Method 84: Reporting Verbs Modulation", 100.0)
    return " ".join(result)


# ── Method 85: XML Structure Sanitization ────────────────────────────────

def method_85_xml_sanitization(
    docx_path: str = None, progress: ProgressCallback = _NOOP
) -> bool:
    """Remove rsidR tracking tags and revision history from .docx XML."""
    progress("Method 85: XML Sanitization", 0.0)
    if docx_path is None:
        progress("Method 85: XML Sanitization", 100.0)
        return True
    try:
        from core.exporter import _sanitize_xml
        from docx import Document
        doc = Document(docx_path)
        _sanitize_xml(doc)
        doc.save(docx_path)
        progress("Method 85: XML Sanitization", 100.0)
        return True
    except Exception as exc:
        warnings.warn(f"XML sanitization failed: {exc}")
        progress("Method 85: XML Sanitization", 100.0)
        return False


# ── Method 86: Intersentential Cohesion ──────────────────────────────────

def method_86_intersentential_cohesion(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Prevent same transition word in consecutive paragraphs."""
    progress("Method 86: Intersentential Cohesion", 0.0)
    TRANSITIONS = [
        "Furthermore", "Moreover", "Additionally", "In addition",
        "However", "Nevertheless", "Nonetheless", "Conversely",
        "Therefore", "Consequently", "Hence", "Thus",
        "Building on this", "In light of this", "Against this backdrop",
    ]
    paragraphs = text.split("\n\n")
    result = []
    last_used: List[str] = []

    for para in paragraphs:
        for trans in TRANSITIONS:
            if para.strip().startswith(trans) and trans in last_used:
                # Replace with an alternative
                alternatives = [t for t in TRANSITIONS if t not in last_used]
                if alternatives:
                    replacement = random.choice(alternatives)
                    para = replacement + para[len(trans):]
                    break
        # Track last 2 transition words used
        for trans in TRANSITIONS:
            if para.strip().startswith(trans):
                last_used.append(trans)
                if len(last_used) > 2:
                    last_used.pop(0)
                break
        result.append(para)

    progress("Method 86: Intersentential Cohesion", 100.0)
    return "\n\n".join(result)


# ── Method 87: Translated Quote Handling ─────────────────────────────────

def method_87_translated_quotes(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Translate foreign-language quotes; append (Author's translation) tag."""
    progress("Method 87: Translated Quotes", 0.0)
    if not _DT_AVAILABLE:
        progress("Method 87: Translated Quotes", 100.0)
        return text

    # Detect non-ASCII quoted passages
    pattern = r'"([^"\x00-\x7F]{10,})"'

    def translate_quote(m: re.Match) -> str:
        try:
            translated = GoogleTranslator(source="auto", target="en").translate(m.group(1))
            if translated:
                return f'"{translated}" (Author\'s translation)'
        except Exception:
            pass
        return m.group(0)

    text = re.sub(pattern, translate_quote, text)
    progress("Method 87: Translated Quotes", 100.0)
    return text


# ── Method 88: Stylistic Micro-Variation Simulation ──────────────────────

def method_88_micro_variation(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Inject minor acceptable stylistic variations to break AI over-perfection."""
    progress("Method 88: Micro-Variation", 0.0)
    for original, variant in MICRO_VARIATIONS:
        if random.random() < 0.4:
            text = text.replace(original, variant)
    progress("Method 88: Micro-Variation", 100.0)
    return text


# ── Method 89: Bibliography Styling ──────────────────────────────────────

def method_89_bibliography_styling(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Sort references alphabetically; apply hanging indent format."""
    progress("Method 89: Bibliography Styling", 0.0)
    # Locate reference section
    ref_section_pattern = r'(References?|Bibliography)\s*\n((?:.+\n?)+)'
    m = re.search(ref_section_pattern, text, re.IGNORECASE | re.MULTILINE)
    if not m:
        progress("Method 89: Bibliography Styling", 100.0)
        return text

    header = m.group(1)
    ref_block = m.group(2)
    refs = [r.strip() for r in ref_block.split('\n') if r.strip()]
    refs_sorted = sorted(refs, key=lambda r: r.split(',')[0].lower())
    formatted_refs = "\n".join(f"    {r}" for r in refs_sorted)
    replacement = f"{header}\n{formatted_refs}\n"
    text = text[:m.start()] + replacement + text[m.end():]
    progress("Method 89: Bibliography Styling", 100.0)
    return text


# ── Method 90: Live Link Verification ────────────────────────────────────

def method_90_live_link_verification(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Ping all URLs; replace dead links with Wayback Machine archived versions."""
    progress("Method 90: Live Link Verification", 0.0)
    if not _REQUESTS_AVAILABLE:
        progress("Method 90: Live Link Verification", 100.0)
        return text

    urls = re.findall(r'https?://\S+', text)
    for url in urls[:10]:  # Limit to prevent timeout
        url_clean = url.rstrip('.,;)')
        try:
            resp = requests.head(url_clean, timeout=5, allow_redirects=True)
            if resp.status_code >= 400:
                # Try Wayback Machine
                wb_resp = requests.get(WAYBACK_API.format(url=url_clean), timeout=5)
                if wb_resp.ok:
                    data = wb_resp.json()
                    archived = data.get("archived_snapshots", {}).get("closest", {}).get("url")
                    if archived:
                        text = text.replace(url_clean, archived)
        except Exception:
            pass

    progress("Method 90: Live Link Verification", 100.0)
    return text


# ── Module convenience function ───────────────────────────────────────────

def run_all(
    text: str,
    config: Dict[str, Any] = None,
    progress: ProgressCallback = _NOOP,
) -> str:
    if config is None:
        config = {}
    text = method_84_reporting_verbs_modulation(text, progress)
    text = method_86_intersentential_cohesion(text, progress)
    text = method_87_translated_quotes(text, progress)
    text = method_88_micro_variation(text, progress)
    text = method_89_bibliography_styling(text, progress)
    text = method_90_live_link_verification(text, progress)
    return text
