"""
main.py — Entry point for the Academic Text Humanizer

Usage:
    python main.py                    # Launch the GUI
    python main.py --cli <input.txt>  # CLI mode (basic pipeline, stdout output)
    python main.py --test             # Run test suite
"""

import os
import sys
import argparse

# ── Repo root on path ─────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def run_gui():
    """Launch the Tkinter GUI."""
    try:
        from gui.interface import launch
        launch()
    except ImportError as exc:
        print(f"[ERROR] Could not launch GUI: {exc}")
        print("Ensure tkinter is available. On Linux: apt-get install python3-tk")
        sys.exit(1)


def run_cli(input_path: str, output_path: str = None, mode: str = "standard"):
    """Run the pipeline in CLI mode."""
    import warnings
    warnings.filterwarnings("ignore")

    print(f"[INFO] Loading: {input_path}")
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    print(f"[INFO] Processing mode: {mode}")
    print(f"[INFO] Input word count: {len(text.split())}")

    # Load config
    config = _load_config()

    # Run pipeline
    result = _run_pipeline(text, config, mode)

    print(f"[INFO] Output word count: {len(result.split())}")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"[INFO] Saved to: {output_path}")
    else:
        print("\n" + "=" * 60)
        print(result)


def _load_config() -> dict:
    """Load configuration from settings.yaml."""
    config_path = os.path.join(_REPO_ROOT, "config", "settings.yaml")
    try:
        import yaml
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _run_pipeline(text: str, config: dict, mode: str = "standard") -> str:
    """Run the humanization pipeline."""
    from core.isolator import isolate, restore
    from core.humanizer import humanize
    from core.academic_refiner import refine

    def progress(name: str, pct: float):
        if pct == 0 or pct == 100:
            print(f"  [{pct:3.0f}%] {name}")

    # Layer 1: Isolation
    print("[INFO] Layer 1: Isolation")
    masked, pmap = isolate(text, use_ner=True)

    # Layer 2: Analysis
    print("[INFO] Layer 2: Analysis")
    try:
        from core.analyzer import analyze
        metrics = analyze(masked)
        print(f"  Entropy: {metrics['entropy']:.3f}")
        print(f"  Sentence length std: {metrics['sentence_length_std']:.2f}")
        print(f"  Word count: {metrics['word_count']}")
    except Exception as e:
        print(f"  [WARN] Analysis failed: {e}")

    # Layer 4: Humanization
    print("[INFO] Layer 4: Humanization")
    masked = humanize(masked, config)

    # Apply methods based on mode
    if mode in ("standard", "full"):
        print("[INFO] Applying methods 11–50")
        try:
            import methods.m11_ref_freeze as m11
            import methods.m21_metadata as m21
            import methods.m31_domain as m31
            import methods.m41_variation as m41

            masked = m11.run_all(masked, config, progress)
            masked = m21.run_all(masked, config, progress)
            masked = m31.run_all(masked, config, progress)
            masked = m41.run_all(masked, config, progress)
        except Exception as e:
            print(f"  [WARN] Methods 11–50 failed: {e}")

    if mode == "full":
        print("[INFO] Applying methods 51–100")
        try:
            import methods.m51_hedging as m51
            import methods.m61_tables as m61
            import methods.m71_equations as m71
            import methods.m81_structure as m81
            import methods.m91_final as m91

            masked = m51.run_all(masked, config, progress)
            masked = m61.run_all(masked, config, progress)
            masked = m71.run_all(masked, config, progress)
            masked = m81.run_all(masked, config, progress)
            masked = m91.run_all(masked, config, progress)
        except Exception as e:
            print(f"  [WARN] Methods 51–100 failed: {e}")

    # Layer 5: Academic Refinement
    print("[INFO] Layer 5: Academic Refinement")
    try:
        masked = refine(masked)
    except Exception as e:
        print(f"  [WARN] Academic refinement failed: {e}")

    # Layer 1 (restore): De-isolation
    print("[INFO] Restoring protected regions")
    result = restore(masked, pmap)

    return result


def run_tests():
    """Run the test suite."""
    import unittest
    loader = unittest.TestLoader()
    tests_dir = os.path.join(_REPO_ROOT, "tests")
    suite = loader.discover(tests_dir, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


# ── CLI argument parsing ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Academic Text Humanizer — 100-Method AI-to-Human Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           Launch the GUI
  python main.py --cli input.txt           Process input.txt (quick mode)
  python main.py --cli input.txt -o out.txt --mode full
  python main.py --test                    Run unit tests
        """,
    )
    parser.add_argument("--cli", metavar="INPUT", help="Run in CLI mode on INPUT file")
    parser.add_argument("-o", "--output", metavar="OUTPUT", help="Output file path (CLI mode)")
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "full"],
        default="standard",
        help="Processing mode: quick (1-10), standard (1-50), full (all 100)",
    )
    parser.add_argument("--test", action="store_true", help="Run test suite")

    args = parser.parse_args()

    if args.test:
        run_tests()
    elif args.cli:
        run_cli(args.cli, args.output, args.mode)
    else:
        run_gui()


if __name__ == "__main__":
    main()
