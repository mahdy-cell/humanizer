# Academic Text Humanizer

A full AI-to-human academic text transformation pipeline with a Tkinter GUI, 100 modular methods, and a 6-layer processing architecture.

---

## Quick Start

### Linux / macOS
```bash
git clone https://github.com/mahdy-cell/humanizer.git
cd humanizer
bash run.sh
```

### Windows
```bat
git clone https://github.com/mahdy-cell/humanizer.git
cd humanizer
run.bat
```

The launcher scripts automatically:
1. Create a `.venv/` virtual environment
2. Install all dependencies from `requirements.txt`
3. Launch the GUI

---

## Running Modes

```bash
bash run.sh                          # Launch GUI (default)
bash run.sh --test                   # Run unit tests
bash run.sh --cli input.txt --mode full  # CLI mode
```

---

## Troubleshooting: "Failed to execute git" / "Error switching to pull request"

If you see one of these errors when trying to open the PR in VS Code, GitHub Desktop, or the `gh` CLI:

```
Error switching to pull request: Failed to execute git
Failed to apply changes from pull request: Failed to execute git
```

**Run these commands in your terminal:**

```bash
# 1. Ensure git is installed and on PATH
git --version

# 2. Clone the repository (full clone, not shallow)
git clone --no-local https://github.com/mahdy-cell/humanizer.git
cd humanizer

# 3. Fetch the PR branch
git fetch origin

# 4. Checkout the PR branch directly
git checkout copilot/build-academic-text-humanization-program
```

If you already have a clone that was checked out shallowly, run:
```bash
git fetch --unshallow origin
git fetch origin
git checkout copilot/build-academic-text-humanization-program
```

---

## Architecture

### 6-Layer Pipeline

| Layer | Module | Purpose |
|---|---|---|
| 1 | `core/isolator.py` | Protect citations, quotes, equations, named entities |
| 2 | `core/analyzer.py` | Measure AI fingerprint (perplexity, burstiness, entropy) |
| 3 | `core/transformer.py` | Paraphrase via back-translation (EN→DE→FR→ES→EN) |
| 4 | `core/humanizer.py` | Synonym replacement, voice conversion, burstiness |
| 5 | `core/academic_refiner.py` | Grammar, hedging, transitions, tense consistency |
| 6 | `core/exporter.py` | .docx (APA 7) and PDF/A export |

### Protection System

All protected spans (citations, quotes, equations, named entities, acronyms, legal clauses) are tokenized as `[REF_n]`, `[QUOTE_n]`, etc. before any method runs, and restored losslessly after all transformations.

```python
from core.isolator import isolate, restore

masked, pmap = isolate(text, use_ner=True)   # Layer 1: protect spans
result = some_method(masked)                  # Run any of 100 methods
output = restore(result, pmap)               # Restore protected spans
```

---

## Configuration (`config/settings.yaml`)

```yaml
synonym_rate: 0.20           # Standard synonym replacement rate
deep_synonym_rate: 0.35      # Deep synonym replacement rate
enable_deep_synonyms: true   # Enable second synonym pass
enable_back_translation: true
back_translation_languages: [de, fr, es]  # EN→DE→FR→ES→EN chain
enable_active_to_passive: true
passive_rate: 0.25           # Rate of active→passive conversion
```

---

## Requirements

- Python 3.9+
- See `requirements.txt` for full dependency list
- Optional heavy dependencies (auto-detected, graceful fallback if missing):
  - `torch` / `transformers` — GPT-2 perplexity scoring
  - `spacy` — dependency parsing for voice conversion
  - `deep-translator` — back-translation chain
  - `sentence-transformers` — semantic clustering