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

These errors have **two root causes**:

1. **Shallow clone** — git history is incomplete; the initial commit appears as `(grafted)`.
2. **Restricted fetch refspec** — only the PR branch is tracked, so `origin/main` is missing and tools cannot find the merge base.

**Fix both with these commands (run inside the repo folder):**

```bash
# Fix 1: restore full history and remove the shallow marker
git fetch --unshallow origin

# Fix 2: configure git to fetch ALL branches (not just the PR branch)
git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"

# Fetch everything (this will now also download origin/main)
git fetch origin

# Switch to the PR branch
git checkout copilot/build-academic-text-humanization-program
```

**Alternatively, do a fresh full clone:**

```bash
git clone --no-local https://github.com/mahdy-cell/humanizer.git
cd humanizer
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