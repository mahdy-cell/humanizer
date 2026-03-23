"""
methods/m91_final.py — Methods 91–100

91. Cluster Analysis
92. Peer-Review Simulation
93. Cross-Reference Formatting
94. Deep Steganography Removal
95. Paragraph Burstiness Balancing
96. Oxford Comma & Punctuation Normalization
97. Executive Summary Generation
98. Commercial Detector API Testing
99. Header/Footer Integrity
100. Clean Export Protocol
"""

import re
import random
import warnings
from typing import Dict, List, Callable, Any, Optional

random.seed(91)

try:
    from sklearn.cluster import KMeans
    import numpy as np
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    _ST_AVAILABLE = True
    _ST_MODEL_INST = None
except ImportError:
    _ST_AVAILABLE = False

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

try:
    import scipy.stats as _stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    _SUMY_AVAILABLE = True
except ImportError:
    _SUMY_AVAILABLE = False

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

ProgressCallback = Callable[[str, float], None]
_NOOP: ProgressCallback = lambda name, pct: None

_ST_MODEL = None
_NLP = None

VAGUE_PHRASES = {
    r'\bit is evident that\b': "the evidence suggests",
    r'\bit is obvious that\b': "the data indicate",
    r'\bwithout a doubt\b': "with a high degree of confidence",
    r'\bclearly,?\s': "the findings demonstrate that ",
    r'\bit can be seen\b': "the results show",
}

WAYBACK_API = "https://archive.org/wayback/available?url={url}"


def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None and _ST_AVAILABLE:
        try:
            _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:
            warnings.warn(f"SentenceTransformer load failed: {exc}")
    return _ST_MODEL


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


# ── Method 91: Cluster Analysis ──────────────────────────────────────────

def method_91_cluster_analysis(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Verify word distribution is not centred (AI trait)."""
    progress("Method 91: Cluster Analysis", 0.0)
    model = _get_st_model()
    if model is None or not _SKLEARN_AVAILABLE:
        progress("Method 91: Cluster Analysis", 100.0)
        return text
    try:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if len(sentences) < 4:
            progress("Method 91: Cluster Analysis", 100.0)
            return text
        embeddings = model.encode(sentences)
        n_clusters = min(3, len(sentences))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        # Check for over-concentrated clustering
        from collections import Counter
        counts = Counter(labels)
        dominant_fraction = max(counts.values()) / len(sentences)
        if dominant_fraction > 0.8:
            # Too concentrated → reshuffle between clusters
            warnings.warn("Text shows AI-like clustering. Applying diversification.")
            # Re-apply entropy variation
            from core.humanizer import replace_synonyms
            text = replace_synonyms(text, replacement_rate=0.2)
    except Exception as exc:
        warnings.warn(f"Cluster analysis failed: {exc}")
    progress("Method 91: Cluster Analysis", 100.0)
    return text


# ── Method 92: Peer-Review Simulation ────────────────────────────────────

def method_92_peer_review(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Detect vague/weak argument sentences; rewrite to be more specific."""
    progress("Method 92: Peer-Review Simulation", 0.0)
    for pattern, replacement in VAGUE_PHRASES.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    progress("Method 92: Peer-Review Simulation", 100.0)
    return text


# ── Method 93: Cross-Reference Formatting ────────────────────────────────

def method_93_cross_reference_formatting(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Convert See Table N / Figure N to formatted references."""
    progress("Method 93: Cross-Reference Formatting", 0.0)
    # Standardise cross-references
    text = re.sub(
        r'[Ss]ee\s+[Tt]able\s+(\d+)',
        r'(refer to Table \1)',
        text,
    )
    text = re.sub(
        r'[Ss]ee\s+[Ff]igure\s+(\d+)',
        r'(refer to Figure \1)',
        text,
    )
    text = re.sub(
        r'[Ss]ee\s+[Aa]ppendix\s+([A-Z])',
        r'(refer to Appendix \1)',
        text,
    )
    progress("Method 93: Cross-Reference Formatting", 100.0)
    return text


# ── Method 94: Deep Steganography Removal ────────────────────────────────

def method_94_steganography_removal(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Scan source for hidden bits/watermark sequences."""
    progress("Method 94: Steganography Removal", 0.0)
    # Remove zero-width characters used as steganographic markers
    steg_chars = [
        '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
        '\u00ad', '\u034f', '\u17b4', '\u17b5',
    ]
    for ch in steg_chars:
        text = text.replace(ch, '')
    # Remove unusual Unicode homoglyphs
    homoglyph_map = {
        '\u0430': 'a',  # Cyrillic а → Latin a
        '\u0435': 'e',  # Cyrillic е → Latin e
        '\u043e': 'o',  # Cyrillic о → Latin o
        '\u0440': 'r',  # Cyrillic р → Latin r
        '\u0441': 'c',  # Cyrillic с → Latin c
        '\u0445': 'x',  # Cyrillic х → Latin x
        '\u0456': 'i',  # Cyrillic і → Latin i
    }
    for homoglyph, latin in homoglyph_map.items():
        text = text.replace(homoglyph, latin)
    progress("Method 94: Steganography Removal", 100.0)
    return text


# ── Method 95: Paragraph Burstiness Balancing ────────────────────────────

def method_95_paragraph_burstiness(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Ensure long analytical paragraphs alternate with short summary paragraphs."""
    progress("Method 95: Paragraph Burstiness", 0.0)
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < 2:
        progress("Method 95: Paragraph Burstiness", 100.0)
        return text

    lengths = [len(p.split()) for p in paragraphs]
    if _SCIPY_AVAILABLE and len(lengths) > 1:
        current_std = float(_stats.tstd(lengths))
    else:
        mean = sum(lengths) / len(lengths)
        current_std = (sum((l - mean) ** 2 for l in lengths) / max(len(lengths) - 1, 1)) ** 0.5

    # If paragraphs are all similar length, split the longest ones
    if current_std < 20:
        result = []
        for para in paragraphs:
            words = para.split()
            if len(words) > 150:
                mid = len(words) // 2
                result.append(" ".join(words[:mid]))
                result.append(" ".join(words[mid:]))
            else:
                result.append(para)
        progress("Method 95: Paragraph Burstiness", 100.0)
        return "\n\n".join(result)

    progress("Method 95: Paragraph Burstiness", 100.0)
    return text


# ── Method 96: Oxford Comma & Punctuation Normalization ──────────────────

def method_96_oxford_comma(
    text: str, progress: ProgressCallback = _NOOP
) -> str:
    """Enforce Oxford comma, colon/semicolon rules per APA/Chicago style guide."""
    progress("Method 96: Oxford Comma", 0.0)
    from core.academic_refiner import normalize_oxford_comma
    text = normalize_oxford_comma(text)
    # Additional punctuation rules:
    # Colon should be preceded by a complete sentence
    text = re.sub(r'\s+:', ':', text)
    # Semicolons: remove space before
    text = re.sub(r'\s+;', ';', text)
    # Double punctuation
    text = re.sub(r'([.!?]){2,}', r'\1', text)
    progress("Method 96: Oxford Comma", 100.0)
    return text


# ── Method 97: Executive Summary Generation ──────────────────────────────

def method_97_executive_summary(
    text: str,
    sentences: int = 5,
    progress: ProgressCallback = _NOOP,
) -> str:
    """Generate executive summary via sumy LexRank."""
    progress("Method 97: Executive Summary", 0.0)
    if not _SUMY_AVAILABLE:
        progress("Method 97: Executive Summary", 100.0)
        return text
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary_sentences = [str(s) for s in summarizer(parser.document, sentences)]
        summary = " ".join(summary_sentences)
        executive_summary = f"EXECUTIVE SUMMARY\n\n{summary}\n\n{'─' * 60}\n\n"
        progress("Method 97: Executive Summary", 100.0)
        return executive_summary + text
    except Exception as exc:
        warnings.warn(f"Executive summary generation failed: {exc}")
        progress("Method 97: Executive Summary", 100.0)
        return text


# ── Method 98: Commercial Detector API Testing ───────────────────────────

def method_98_detector_testing(
    text: str,
    api_endpoint: str = "",
    threshold: float = 0.05,
    progress: ProgressCallback = _NOOP,
) -> str:
    """Send text chunks to detection APIs; re-apply Method 23 on flagged paragraphs."""
    progress("Method 98: Detector Testing", 0.0)
    if not api_endpoint or not _REQUESTS_AVAILABLE:
        progress("Method 98: Detector Testing", 100.0)
        return text
    try:
        paragraphs = text.split("\n\n")
        result = []
        for para in paragraphs:
            if not para.strip():
                result.append(para)
                continue
            try:
                resp = requests.post(
                    api_endpoint,
                    json={"text": para},
                    timeout=10,
                )
                if resp.ok:
                    score = resp.json().get("score", 0.0)
                    if score > threshold:
                        from methods.m21_metadata import method_23_predictability_breaking
                        para = method_23_predictability_breaking(para)
            except Exception:
                pass
            result.append(para)
        progress("Method 98: Detector Testing", 100.0)
        return "\n\n".join(result)
    except Exception as exc:
        warnings.warn(f"Detector testing failed: {exc}")
        progress("Method 98: Detector Testing", 100.0)
        return text


# ── Method 99: Header/Footer Integrity ───────────────────────────────────

def method_99_header_footer(
    docx_path: str = None,
    config: Dict[str, Any] = None,
    progress: ProgressCallback = _NOOP,
) -> bool:
    """Add running head, page numbers, 2.54cm margins per APA standard."""
    progress("Method 99: Header/Footer Integrity", 0.0)
    if docx_path is None or config is None:
        progress("Method 99: Header/Footer Integrity", 100.0)
        return True
    try:
        from core.exporter import _add_page_numbers, _set_margins
        from docx import Document
        doc = Document(docx_path)
        _set_margins(doc, config.get("margins_cm", 2.54))
        title = config.get("running_head", config.get("title", "RUNNING HEAD"))
        _add_page_numbers(doc, running_head=title[:50].upper())
        doc.save(docx_path)
        progress("Method 99: Header/Footer Integrity", 100.0)
        return True
    except Exception as exc:
        warnings.warn(f"Header/footer setup failed: {exc}")
        progress("Method 99: Header/Footer Integrity", 100.0)
        return False


# ── Method 100: Clean Export Protocol ────────────────────────────────────

def method_100_clean_export(
    text: str,
    output_path: str = "output.pdf",
    config: Dict[str, Any] = None,
    progress: ProgressCallback = _NOOP,
) -> bool:
    """Export PDF/A with embedded fonts; remove all Track Changes."""
    progress("Method 100: Clean Export", 0.0)
    if config is None:
        config = {}
    from core.exporter import export
    result = export(text, output_path, format="pdf", config=config)
    progress("Method 100: Clean Export", 100.0)
    return result


# ── Module convenience function ───────────────────────────────────────────

def run_all(
    text: str,
    config: Dict[str, Any] = None,
    progress: ProgressCallback = _NOOP,
) -> str:
    if config is None:
        config = {}
    text = method_91_cluster_analysis(text, progress)
    text = method_92_peer_review(text, progress)
    text = method_93_cross_reference_formatting(text, progress)
    text = method_94_steganography_removal(text, progress)
    text = method_95_paragraph_burstiness(text, progress)
    text = method_96_oxford_comma(text, progress)
    return text
