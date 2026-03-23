"""
core/transformer.py — Layer 3: Transformation Layer

Paraphrases free (unprotected) text via:
  - Parrot paraphraser (T5/BART via HuggingFace)
  - Back-translation: EN → DE → JA → EN via deep_translator
  - Dependency-tree-based sentence reconstruction via spaCy
"""

import re
import warnings
from typing import List, Optional

try:
    from deep_translator import GoogleTranslator
    _DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    _DEEP_TRANSLATOR_AVAILABLE = False
    warnings.warn("deep_translator not available; back-translation disabled.")

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False
    warnings.warn("spaCy not available; dependency reconstruction disabled.")

try:
    from parrot import Parrot
    _PARROT: Optional[object] = None
    _PARROT_AVAILABLE = True
except ImportError:
    _PARROT_AVAILABLE = False
    warnings.warn("parrot-paraphraser not available; parrot paraphrase disabled.")


# ── spaCy model cache ──────────────────────────────────────────────────────
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


# ── Parrot paraphraser ─────────────────────────────────────────────────────

def _get_parrot():
    global _PARROT
    if _PARROT is None and _PARROT_AVAILABLE:
        try:
            _PARROT = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")
        except Exception as exc:
            warnings.warn(f"Parrot model load failed: {exc}")
    return _PARROT


def parrot_paraphrase(text: str) -> str:
    """Paraphrase *text* using the Parrot model."""
    if not _PARROT_AVAILABLE:
        return text
    try:
        parrot = _get_parrot()
        if parrot is None:
            return text
        # Parrot works sentence by sentence
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        results = []
        for sent in sentences:
            try:
                paraphrases = parrot.augment(input_phrase=sent, use_gpu=False)
                if paraphrases:
                    results.append(paraphrases[0][0])
                else:
                    results.append(sent)
            except Exception:
                results.append(sent)
        return " ".join(results)
    except Exception as exc:
        warnings.warn(f"Parrot paraphrase failed: {exc}")
        return text


# ── Back-translation ───────────────────────────────────────────────────────

def back_translate(text: str, pivot_languages: List[str] = None) -> str:
    """
    Translate text through a chain of pivot languages and back to English.
    Default chain: EN → DE → JA → EN.
    """
    if not _DEEP_TRANSLATOR_AVAILABLE:
        return text
    if pivot_languages is None:
        pivot_languages = ["de", "ja"]
    try:
        current = text
        current_lang = "en"
        for target_lang in pivot_languages:
            current = GoogleTranslator(source=current_lang, target=target_lang).translate(current)
            current_lang = target_lang
        # Translate back to English
        current = GoogleTranslator(source=current_lang, target="en").translate(current)
        return current if current else text
    except Exception as exc:
        warnings.warn(f"Back-translation failed: {exc}")
        return text


# ── Dependency-tree reconstruction ────────────────────────────────────────

def _reconstruct_sentence(sent) -> str:
    """
    Given a spaCy Span, attempt a simple subject-verb-object reordering
    to break GPT sentence patterns while preserving meaning.
    """
    try:
        root = [t for t in sent if t.dep_ == "ROOT"]
        if not root:
            return sent.text
        root = root[0]

        subjects = [t for t in root.children if t.dep_ in {"nsubj", "nsubjpass"}]
        objects = [t for t in root.children if t.dep_ in {"dobj", "pobj", "attr"}]

        if subjects and objects:
            # Reconstruct: subject + verb + object + rest
            subj_span = " ".join(
                t.text for t in sorted(subjects[0].subtree, key=lambda x: x.i)
            )
            obj_span = " ".join(
                t.text for t in sorted(objects[0].subtree, key=lambda x: x.i)
            )
            verb_span = root.text
            return f"{subj_span} {verb_span} {obj_span}."
    except Exception:
        pass
    return sent.text


def dependency_reconstruct(text: str) -> str:
    """Reconstruct sentences via spaCy dependency trees."""
    if not _SPACY_AVAILABLE:
        return text
    try:
        nlp = _get_nlp()
        if nlp is None:
            return text
        doc = nlp(text)
        result = []
        for sent in doc.sents:
            result.append(_reconstruct_sentence(sent))
        return " ".join(result)
    except Exception as exc:
        warnings.warn(f"Dependency reconstruction failed: {exc}")
        return text


# ── Public API ─────────────────────────────────────────────────────────────

def transform(
    text: str,
    use_parrot: bool = True,
    use_back_translation: bool = True,
    use_dependency: bool = True,
    pivot_languages: List[str] = None,
) -> str:
    """
    Apply the full transformation pipeline to *text*.

    Parameters:
        text: input text (with protection tokens intact)
        use_parrot: whether to apply Parrot paraphrasing
        use_back_translation: whether to apply back-translation
        use_dependency: whether to apply dependency reconstruction
        pivot_languages: languages for back-translation (default: ['de', 'ja'])
    """
    if use_back_translation:
        text = back_translate(text, pivot_languages)
    if use_parrot:
        text = parrot_paraphrase(text)
    if use_dependency:
        text = dependency_reconstruct(text)
    return text
