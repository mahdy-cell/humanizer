"""
core/analyzer.py — Layer 2: Analysis Layer

Measures the current AI fingerprint of text using:
  - Perplexity & burstiness via GPT-2 log-probabilities
  - Shannon entropy via scipy
  - Sentence-length standard deviation via scipy.stats
  - TF-IDF keyword density via scikit-learn
  - Sentiment/confidence via TextBlob + VADER
"""

import math
import re
import warnings
from typing import Dict, Any, List

# ── optional heavy imports ─────────────────────────────────────────────────
try:
    import scipy.stats as _stats
    import scipy.special as _special
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    warnings.warn("scipy not available; some analysis metrics disabled.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available; TF-IDF analysis disabled.")

try:
    from textblob import TextBlob
    _TEXTBLOB_AVAILABLE = True
except ImportError:
    _TEXTBLOB_AVAILABLE = False
    warnings.warn("TextBlob not available; sentiment analysis disabled.")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_ANALYZER = SentimentIntensityAnalyzer()
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False
    warnings.warn("vaderSentiment not available; VADER sentiment disabled.")

try:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    import torch
    _GPT2_MODEL = None
    _GPT2_TOKENIZER = None
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers/torch not available; GPT-2 perplexity disabled.")


# ── GPT-2 helpers ──────────────────────────────────────────────────────────

def _load_gpt2():
    global _GPT2_MODEL, _GPT2_TOKENIZER
    if _GPT2_MODEL is None:
        _GPT2_TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")
        _GPT2_MODEL = GPT2LMHeadModel.from_pretrained("gpt2")
        _GPT2_MODEL.eval()


def compute_perplexity(text: str) -> float:
    """Return GPT-2 perplexity for *text* (lower = more predictable / AI-like)."""
    if not _TRANSFORMERS_AVAILABLE:
        return 0.0
    try:
        _load_gpt2()
        encodings = _GPT2_TOKENIZER(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = _GPT2_MODEL(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss.item()
        return math.exp(loss)
    except Exception as exc:
        warnings.warn(f"GPT-2 perplexity failed: {exc}")
        return 0.0


def compute_burstiness(sentences: List[str]) -> float:
    """
    Burstiness B = (σ - μ) / (σ + μ) of sentence-length distribution.
    B > 0 indicates human-like variance; B ≈ -1 is very regular (AI-like).
    """
    if not _SCIPY_AVAILABLE or len(sentences) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    mu = sum(lengths) / len(lengths)
    sigma = _stats.tstd(lengths)
    return (sigma - mu) / (sigma + mu) if (sigma + mu) > 0 else 0.0


# ── Public helpers ─────────────────────────────────────────────────────────

def _tokenize_sentences(text: str) -> List[str]:
    """Simple sentence splitter (fallback when spaCy unavailable)."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def compute_entropy(text: str) -> float:
    """Shannon entropy on word-unigram frequency distribution."""
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    total = len(words)
    probs = [c / total for c in freq.values()]
    if _SCIPY_AVAILABLE:
        try:
            import numpy as _np
            return float(_special.entr(_np.array(probs)).sum())
        except Exception:
            pass
    # pure-Python fallback
    return -sum(p * math.log(p) for p in probs if p > 0)


def compute_sentence_length_std(text: str) -> float:
    """Standard deviation of sentence lengths (in words)."""
    sentences = _tokenize_sentences(text)
    if len(sentences) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    if _SCIPY_AVAILABLE:
        return float(_stats.tstd(lengths))
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / (len(lengths) - 1)
    return variance ** 0.5


def compute_tfidf_density(text: str, top_n: int = 10) -> Dict[str, float]:
    """Return top-N TF-IDF keywords and their scores."""
    if not _SKLEARN_AVAILABLE:
        return {}
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n)
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        return {feature_names[i]: float(scores[i]) for i in scores.argsort()[::-1][:top_n]}
    except Exception as exc:
        warnings.warn(f"TF-IDF failed: {exc}")
        return {}


def compute_sentiment(text: str) -> Dict[str, float]:
    """Return sentiment scores using TextBlob and VADER."""
    result: Dict[str, float] = {}
    if _TEXTBLOB_AVAILABLE:
        try:
            blob = TextBlob(text)
            result["textblob_polarity"] = blob.sentiment.polarity
            result["textblob_subjectivity"] = blob.sentiment.subjectivity
        except Exception as exc:
            warnings.warn(f"TextBlob sentiment failed: {exc}")
    if _VADER_AVAILABLE:
        try:
            scores = _VADER_ANALYZER.polarity_scores(text)
            result.update({f"vader_{k}": v for k, v in scores.items()})
        except Exception as exc:
            warnings.warn(f"VADER sentiment failed: {exc}")
    return result


# ── Master analysis function ───────────────────────────────────────────────

def analyze(text: str) -> Dict[str, Any]:
    """
    Run all analysis metrics on *text*.

    Returns a dict with keys:
        perplexity, burstiness, entropy, sentence_length_std,
        tfidf_keywords, sentiment
    """
    sentences = _tokenize_sentences(text)
    return {
        "perplexity": compute_perplexity(text),
        "burstiness": compute_burstiness(sentences),
        "entropy": compute_entropy(text),
        "sentence_length_std": compute_sentence_length_std(text),
        "tfidf_keywords": compute_tfidf_density(text),
        "sentiment": compute_sentiment(text),
        "sentence_count": len(sentences),
        "word_count": len(re.findall(r'\b\w+\b', text)),
    }
