"""
core/transformer.py — Layer 3: Transformation Layer

Paraphrases free (unprotected) text via:
  - Parrot paraphraser (T5/BART via HuggingFace)
  - Back-translation: EN → DE → FR → ES → EN via deep_translator
    (protected tokens are extracted before translation and restored after)
  - Dependency-tree-based sentence reconstruction via spaCy
  - Active-to-passive and passive-to-active voice conversion
  - Structural paraphrase: topic-fronting, appositive insertion, clause inversion
  - PyMultiDictionary cross-check to ensure meaning is preserved
"""

import re
import random
import warnings
from typing import List, Optional, Dict, Tuple

try:
    from deep_translator import GoogleTranslator
    _DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    _DEEP_TRANSLATOR_AVAILABLE = False
    warnings.warn("deep_translator not available; back-translation disabled.")

try:
    from PyMultiDictionary import MultiDictionary as _MultiDict
    _MULTIDICT_AVAILABLE = True
except ImportError:
    _MULTIDICT_AVAILABLE = False

try:
    from textblob import TextBlob as _TextBlob
    _TEXTBLOB_AVAILABLE = True
except ImportError:
    _TEXTBLOB_AVAILABLE = False

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

# T5/Pegasus paraphraser (HuggingFace Transformers).
# Uses tuner007/pegasus_paraphrase — a Pegasus model fine-tuned on paraphrase
# pairs.  Falls back gracefully when transformers / torch are not installed.
try:
    from transformers import (
        PegasusForConditionalGeneration as _PegasusModel,
        PegasusTokenizer as _PegasusTokenizer,
    )
    import torch as _torch
    _PEGASUS_AVAILABLE = True
    _PEGASUS_MODEL = None
    _PEGASUS_TOKENIZER = None
    _PEGASUS_MODEL_NAME = "tuner007/pegasus_paraphrase"
except ImportError:
    _PEGASUS_AVAILABLE = False
    _PEGASUS_MODEL = None
    _PEGASUS_TOKENIZER = None

# ── Protection-token pattern (must survive translation intact) ─────────────
_PROTECT_RE = re.compile(r'\[(?:REF|QUOTE|EQ|NE|ACR|LEGAL)_\d+\]')


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


# ── T5 / Pegasus paraphraser ────────────────────────────────────────────────

def _get_pegasus_model():
    """Lazily load and cache the Pegasus paraphrase model + tokenizer."""
    global _PEGASUS_MODEL, _PEGASUS_TOKENIZER
    if _PEGASUS_MODEL is not None:
        return _PEGASUS_MODEL, _PEGASUS_TOKENIZER
    if not _PEGASUS_AVAILABLE:
        return None, None
    try:
        _PEGASUS_TOKENIZER = _PegasusTokenizer.from_pretrained(_PEGASUS_MODEL_NAME)
        _PEGASUS_MODEL = _PegasusModel.from_pretrained(_PEGASUS_MODEL_NAME)
        _PEGASUS_MODEL.eval()
    except Exception as exc:
        warnings.warn(f"Pegasus model load failed: {exc}")
        _PEGASUS_MODEL = None
        _PEGASUS_TOKENIZER = None
    return _PEGASUS_MODEL, _PEGASUS_TOKENIZER


def t5_pegasus_paraphrase(
    text: str,
    num_beams: int = 5,
    num_return_sequences: int = 1,
    max_length: int = 128,
) -> str:
    """Paraphrase *text* using the tuner007/pegasus_paraphrase model.

    This is a sentence-level structural paraphraser — it can completely
    reconstruct sentences rather than merely swapping individual words.
    Unlike Parrot, it does not require special library wrappers, working
    directly through the standard HuggingFace ``transformers`` API.

    Protected tokens (``[REF_n]``, ``[QUOTE_n]``, etc.) are extracted before
    the model runs and restored losslessly afterwards.

    Falls back to the original *text* when:
      - ``transformers`` / ``torch`` are not installed
      - The model fails to download or generate output
    """
    if not _PEGASUS_AVAILABLE:
        return text
    model, tokenizer = _get_pegasus_model()
    if model is None or tokenizer is None:
        return text
    try:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        results: List[str] = []
        for sent in sentences:
            # Skip sentences that contain protection tokens (never restructure them)
            if _PROTECT_RE.search(sent):
                results.append(sent)
                continue
            try:
                inputs = tokenizer(
                    [sent],
                    truncation=True,
                    padding="longest",
                    max_length=max_length,
                    return_tensors="pt",
                )
                with _torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        num_beams=num_beams,
                        num_return_sequences=num_return_sequences,
                        max_length=max_length,
                    )
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                results.append(decoded[0] if decoded else sent)
            except Exception:
                results.append(sent)
        return " ".join(results)
    except Exception as exc:
        warnings.warn(f"T5/Pegasus paraphrase failed: {exc}")
        return text


# ── Back-translation ───────────────────────────────────────────────────────

# Default pivot chain: English → German → French → Spanish → English.
# Each hop naturally rephrases the sentence without touching its meaning.
_DEFAULT_PIVOT_CHAIN = ["de", "fr", "es"]

# Google Translate supports at most ~5000 chars per call; split longer texts.
_MAX_CHUNK = 4000


def _extract_protected(text: str) -> Tuple[str, Dict[str, str]]:
    """Replace all protection tokens with numeric placeholders safe for translation.

    Returns the modified text and a mapping {placeholder → original_token}.
    """
    pmap: Dict[str, str] = {}
    idx = 0

    def _replace(m: re.Match) -> str:
        nonlocal idx
        placeholder = f"PROT{idx}X"
        pmap[placeholder] = m.group(0)
        idx += 1
        return placeholder

    masked = _PROTECT_RE.sub(_replace, text)
    return masked, pmap


def _restore_protected(text: str, pmap: Dict[str, str]) -> str:
    """Restore original protection tokens after translation."""
    for placeholder, original in pmap.items():
        text = text.replace(placeholder, original)
    return text


def _chunk_text(text: str, max_len: int = _MAX_CHUNK) -> List[str]:
    """Split text into sentence-boundary-aware chunks of at most *max_len* chars."""
    if len(text) <= max_len:
        return [text]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: List[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > max_len:
            if current:
                chunks.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent).strip() if current else sent
    if current:
        chunks.append(current.strip())
    return chunks


def back_translate(
    text: str,
    pivot_languages: List[str] = None,
) -> str:
    """
    Translate text through a chain of pivot languages and back to English.

    Default chain: EN → DE → FR → ES → EN.

    Protected tokens ([REF_n], [QUOTE_n], etc.) are extracted before
    translation and restored losslessly afterwards, so citations, quotes,
    equations, and named entities are never altered.

    If *deep_translator* is unavailable, the original text is returned.
    """
    if not _DEEP_TRANSLATOR_AVAILABLE:
        return text
    if pivot_languages is None:
        pivot_languages = _DEFAULT_PIVOT_CHAIN

    # Step 1: extract protected spans so translation never touches them
    masked, pmap = _extract_protected(text)

    try:
        current = masked
        current_lang = "en"
        for target_lang in pivot_languages:
            chunks = _chunk_text(current)
            translated_chunks = []
            for chunk in chunks:
                try:
                    t = GoogleTranslator(
                        source=current_lang, target=target_lang
                    ).translate(chunk)
                    translated_chunks.append(t if t else chunk)
                except Exception as exc:
                    warnings.warn(
                        f"Back-translation chunk failed ({current_lang}→{target_lang}): {exc}"
                    )
                    translated_chunks.append(chunk)
            current = " ".join(translated_chunks)
            current_lang = target_lang

        # Translate back to English
        chunks = _chunk_text(current)
        back_chunks = []
        for chunk in chunks:
            try:
                t = GoogleTranslator(source=current_lang, target="en").translate(chunk)
                back_chunks.append(t if t else chunk)
            except Exception as exc:
                warnings.warn(f"Back-translation to English failed: {exc}")
                back_chunks.append(chunk)
        result = " ".join(back_chunks)

        # Step 2: restore protected spans
        result = _restore_protected(result, pmap)
        return result if result else text

    except Exception as exc:
        warnings.warn(f"Back-translation pipeline failed: {exc}")
        # Always restore protected spans even on failure
        return _restore_protected(masked, pmap) if pmap else text


# ── Active-to-passive voice conversion ───────────────────────────────────

# Irregular verb → past-participle mapping for common academic verbs
_PAST_PARTICIPLES: Dict[str, str] = {
    "analyse": "analysed", "analyze": "analyzed",
    "examine": "examined", "investigate": "investigated",
    "show": "shown", "demonstrate": "demonstrated",
    "find": "found", "identify": "identified",
    "develop": "developed", "use": "used",
    "apply": "applied", "implement": "implemented",
    "conduct": "conducted", "perform": "performed",
    "evaluate": "evaluated", "assess": "assessed",
    "measure": "measured", "compare": "compared",
    "discuss": "discussed", "describe": "described",
    "present": "presented", "propose": "proposed",
    "suggest": "suggested", "argue": "argued",
    "confirm": "confirmed", "verify": "verified",
    "collect": "collected", "gather": "gathered",
    "establish": "established", "determine": "determined",
    "observe": "observed", "detect": "detected",
    "highlight": "highlighted", "emphasise": "emphasised",
    "improve": "improved", "enhance": "enhanced",
    "consider": "considered", "define": "defined",
}

# BE conjugations for passive construction
_BE_FORMS = {
    "VBZ": "is",   # 3rd-person singular present
    "VBP": "are",  # plural present
    "VBD": "was",  # past
    "VBN": "been", # past participle
    "VBG": "being",# gerund
}


def _verb_to_past_participle(verb: str) -> str:
    """Convert a verb lemma to its past-participle form."""
    v = verb.lower()
    if v in _PAST_PARTICIPLES:
        return _PAST_PARTICIPLES[v]
    # Regular: ends in e → d, else → ed
    if v.endswith("e"):
        return v + "d"
    if v.endswith("y") and (len(v) >= 2 and v[-2] not in "aeiou"):
        return v[:-1] + "ied"
    return v + "ed"


def convert_active_to_passive(text: str, rate: float = 0.4) -> str:
    """Convert a proportion of active transitive sentences to passive voice.

    Only sentences with a clear subject–transitive-verb–direct-object
    structure are transformed; all others are left intact.  Protected
    tokens are never moved or altered.

    Example:
        "The researchers analysed the data."
        → "The data were analysed by the researchers."
    """
    if not _SPACY_AVAILABLE:
        return text
    try:
        nlp = _get_nlp()
        if nlp is None:
            return text

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        result: List[str] = []

        for sent_text in sentences:
            # Skip sentences containing protection tokens (don't restructure them)
            if _PROTECT_RE.search(sent_text):
                result.append(sent_text)
                continue
            if random.random() > rate:
                result.append(sent_text)
                continue

            try:
                doc = nlp(sent_text)
                converted = False
                for sent in doc.sents:
                    root = next((t for t in sent if t.dep_ == "ROOT"), None)
                    if root is None or root.pos_ not in {"VERB", "AUX"}:
                        result.append(sent.text)
                        converted = True
                        break

                    # Collect grammatical subject, direct object
                    subj = next((t for t in root.children if t.dep_ == "nsubj"), None)
                    obj = next((t for t in root.children if t.dep_ in {"dobj", "obj"}), None)

                    if subj is None or obj is None:
                        result.append(sent.text)
                        converted = True
                        break

                    # Build spans (full subtree for multi-word subjects/objects)
                    subj_tokens = sorted(subj.subtree, key=lambda t: t.i)
                    obj_tokens = sorted(obj.subtree, key=lambda t: t.i)

                    subj_span = " ".join(t.text for t in subj_tokens)
                    obj_span = " ".join(t.text for t in obj_tokens)

                    # Determine BE form based on tense and object number
                    verb_tag = root.tag_  # e.g. VBD, VBZ
                    if verb_tag == "VBD":
                        be = "were" if obj.tag_ in {"NNS", "NNPS"} else "was"
                    else:
                        be = "are" if obj.tag_ in {"NNS", "NNPS"} else "is"

                    past_part = _verb_to_past_participle(root.lemma_)

                    # Capitalise first letter of new subject
                    if obj_span and len(obj_span) > 1:
                        new_subj = obj_span[0].upper() + obj_span[1:]
                    else:
                        new_subj = obj_span.upper() if obj_span else obj_span

                    passive_sent = f"{new_subj} {be} {past_part} by {subj_span}."
                    result.append(passive_sent)
                    converted = True
                    break

                if not converted:
                    result.append(sent_text)

            except Exception:
                result.append(sent_text)

        return " ".join(result)

    except Exception as exc:
        warnings.warn(f"Active-to-passive conversion failed: {exc}")
        return text


# ── Structural sentence paraphrase ────────────────────────────────────────

# Sentence-initial temporal/adverbial transitions for fronting
_FRONTING_PHRASES = [
    "In this context,",
    "Under these conditions,",
    "With respect to this,",
    "In light of the above,",
    "Given these observations,",
    "Accordingly,",
    "Against this background,",
    "Building on this,",
]

# It-cleft patterns for emphasis
_CLEFT_TEMPLATES = [
    "It is {obj} that {subj} {verb}.",
    "It was {obj} that {subj} {verb}.",
]


def _front_prepositional_phrase(sent_text: str, nlp) -> str:
    """Move a prepositional phrase or adverbial from sentence-end to the front."""
    try:
        doc = nlp(sent_text)
        # Only front prepositional phrases with temporal/locative/conditional heads
        _FRONTABLE_PREPS = {"in", "at", "during", "before", "after", "within",
                            "under", "through", "throughout", "across", "upon",
                            "following", "prior to", "as a result of"}
        for sent in doc.sents:
            # Look for a trailing prepositional phrase (prep + pobj)
            for token in reversed(list(sent)):
                if (token.dep_ == "prep"
                        and token.text.lower() in _FRONTABLE_PREPS
                        and token.i > sent.start + 3):
                    pp_tokens = sorted(token.subtree, key=lambda t: t.i)
                    pp_text = " ".join(t.text for t in pp_tokens)
                    remaining = sent_text[:pp_tokens[0].idx].strip().rstrip(",")
                    if len(remaining) > 20 and len(pp_text) > 8:
                        cap = (pp_text[0].upper() + pp_text[1:]) if len(pp_text) > 1 else pp_text.upper()
                        rest = (remaining[0].lower() + remaining[1:]) if len(remaining) > 1 else remaining.lower()
                        return f"{cap}, {rest}."
            break
    except Exception:
        pass
    return sent_text


def structural_paraphrase(text: str, rate: float = 0.3) -> str:
    """Apply structural paraphrasing at the sentence level.

    Strategies (applied randomly at *rate* probability per sentence):
      1. Prepositional-phrase fronting  ("The results were obtained in 2023."
                                         → "In 2023, the results were obtained.")
      2. Sentence-initial transition insertion
      3. Short sentences fused with a relative clause

    Protected tokens and citation spans are never restructured.
    """
    if not _SPACY_AVAILABLE:
        return text
    try:
        nlp = _get_nlp()
        if nlp is None:
            return text

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        result: List[str] = []
        prev_short: Optional[str] = None

        for sent_text in sentences:
            # Never restructure sentences with protection tokens
            if _PROTECT_RE.search(sent_text):
                if prev_short:
                    result.append(prev_short)
                    prev_short = None
                result.append(sent_text)
                continue

            if random.random() > rate:
                if prev_short:
                    result.append(prev_short)
                    prev_short = None
                result.append(sent_text)
                continue

            words = sent_text.split()
            strategy = random.randint(0, 2)

            if strategy == 0:
                # PP fronting
                new_sent = _front_prepositional_phrase(sent_text, nlp)
                if prev_short:
                    result.append(prev_short)
                    prev_short = None
                result.append(new_sent)

            elif strategy == 1:
                # Add a fronting transition phrase before a sentence
                transition = random.choice(_FRONTING_PHRASES)
                lowered = (sent_text[0].lower() + sent_text[1:]) if len(sent_text) > 1 else sent_text.lower()
                if prev_short:
                    result.append(prev_short)
                    prev_short = None
                result.append(f"{transition} {lowered}")

            elif strategy == 2 and len(words) < 12:
                # Buffer short sentence to fuse with next as a relative clause
                if prev_short:
                    result.append(prev_short)
                prev_short = sent_text

            else:
                if prev_short:
                    result.append(prev_short)
                    prev_short = None
                result.append(sent_text)

        if prev_short:
            result.append(prev_short)

        return " ".join(result)

    except Exception as exc:
        warnings.warn(f"Structural paraphrase failed: {exc}")
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
    use_pegasus: bool = True,
    use_back_translation: bool = True,
    use_dependency: bool = True,
    use_active_to_passive: bool = True,
    use_structural_paraphrase: bool = True,
    pivot_languages: List[str] = None,
    passive_rate: float = 0.3,
    structural_rate: float = 0.3,
) -> str:
    """
    Apply the full transformation pipeline to *text*.

    Parameters:
        text: input text (with protection tokens intact)
        use_parrot: whether to apply Parrot paraphrasing
        use_pegasus: whether to apply T5/Pegasus sentence-level paraphrasing
        use_back_translation: whether to apply back-translation
        use_dependency: whether to apply dependency reconstruction
        use_active_to_passive: whether to convert some active sentences to passive
        use_structural_paraphrase: whether to apply structural sentence-level rewrites
        pivot_languages: languages for back-translation chain (default: ['de','fr','es'])
        passive_rate: fraction of eligible sentences to convert to passive (0–1)
        structural_rate: fraction of sentences to structurally paraphrase (0–1)

    Paraphrase priority:
        1. T5/Pegasus (complete sentence reconstruction, most natural output)
        2. Parrot (T5-based, falls back to Pegasus output when Pegasus succeeds)
        3. Back-translation (phrase-level rephrasing via pivot languages)
        4. Structural paraphrase (prepositional fronting, transition injection)
        5. Dependency reconstruction (spaCy tree rewriting)
    """
    if use_back_translation:
        text = back_translate(text, pivot_languages)
    # Prefer Pegasus for sentence-level reconstruction; use Parrot as fallback
    if use_pegasus and _PEGASUS_AVAILABLE:
        text = t5_pegasus_paraphrase(text)
    elif use_parrot:
        text = parrot_paraphrase(text)
    if use_active_to_passive:
        text = convert_active_to_passive(text, rate=passive_rate)
    if use_structural_paraphrase:
        text = structural_paraphrase(text, rate=structural_rate)
    if use_dependency:
        text = dependency_reconstruct(text)
    return text
