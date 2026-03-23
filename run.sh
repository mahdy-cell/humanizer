#!/usr/bin/env bash
# =============================================================================
#  run.sh — Launcher for the Academic Text Humanizer
#  Usage:  bash run.sh               → Launch GUI
#          bash run.sh --cli file.txt → CLI mode
#          bash run.sh --test         → Run tests
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colour helpers ────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Find Python ───────────────────────────────────────────────────────────
PYTHON=""
for cmd in python3 python python3.12 python3.11 python3.10; do
    if command -v "$cmd" &>/dev/null; then
        MAJOR=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        MINOR=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [[ "$MAJOR" -ge 3 && "$MINOR" -ge 9 ]] 2>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    error "Python 3.9+ not found. Please install Python from https://python.org"
    exit 1
fi

info "Using Python: $PYTHON ($($PYTHON --version))"

# ── Virtual environment ───────────────────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment at .venv/ ..."
    $PYTHON -m venv "$VENV_DIR"
fi

# Activate venv
if [[ -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
    info "Virtual environment activated."
else
    warn "Could not activate venv; using system Python."
fi

# ── Install / upgrade dependencies ───────────────────────────────────────
if [[ "$1" != "--skip-install" ]]; then
    info "Checking dependencies (requirements.txt) ..."
    pip install --quiet --upgrade pip
    # Install only the lightweight core dependencies that are always needed
    # (heavy ML libs like torch/transformers are optional and can be installed separately)
    CORE_DEPS=(
        "pyyaml>=6.0.1"
        "python-docx>=1.0.1"
        "reportlab>=4.0.4"
        "requests>=2.31.0"
        "beautifulsoup4>=4.12.2"
        "unidecode>=1.3.7"
    )
    for dep in "${CORE_DEPS[@]}"; do
        pip install --quiet "$dep" || warn "Could not install $dep (optional)"
    done

    # Attempt full requirements install (failures are non-fatal)
    if [[ -f "$SCRIPT_DIR/requirements.txt" ]]; then
        pip install --quiet -r "$SCRIPT_DIR/requirements.txt" \
            --no-deps \
            || warn "Some optional dependencies could not be installed; the app will still run."
    fi
fi

# ── Check tkinter availability ────────────────────────────────────────────
if [[ "$1" != "--cli" && "$1" != "--test" ]]; then
    if ! $PYTHON -c "import tkinter" 2>/dev/null; then
        warn "tkinter not found. Installing python3-tk ..."
        if command -v apt-get &>/dev/null; then
            sudo apt-get install -y python3-tk 2>/dev/null \
                || warn "Could not install python3-tk automatically. Run: sudo apt-get install python3-tk"
        elif command -v brew &>/dev/null; then
            brew install python-tk 2>/dev/null \
                || warn "Could not install python-tk. Run: brew install python-tk"
        else
            warn "Please install tkinter for your OS to use the GUI."
        fi
    fi
fi

# ── Run application ───────────────────────────────────────────────────────
info "Starting Academic Text Humanizer ..."
echo ""

$PYTHON "$SCRIPT_DIR/main.py" "$@"
