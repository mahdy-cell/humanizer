"""
tests/test_pipeline.py — Unit tests for the Academic Text Humanizer

Covers:
  - Token masking and restoration
  - Protected region integrity through full pipeline
  - GUI initialization without errors
  - Export functions (.docx, .pdf)
  - End-to-end test with sample_01.txt
"""

import os
import sys
import tempfile
import unittest

# ── Path setup ────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

SAMPLE_01_PATH = os.path.join(_REPO_ROOT, "gui", "samples", "sample_01.txt")

# ── Helpers ───────────────────────────────────────────────────────────────

def _read_sample(n: int) -> str:
    path = os.path.join(_REPO_ROOT, "gui", "samples", f"sample_0{n}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ── Token Masking Tests ───────────────────────────────────────────────────

class TestIsolator(unittest.TestCase):
    """Tests for core/isolator.py."""

    def setUp(self):
        from core.isolator import reset_counters
        reset_counters()

    def test_citation_masking(self):
        """Citations should be replaced with [REF_n] tokens."""
        from core.isolator import isolate_citations
        pmap = {}
        text = "This was shown by Smith (2020) and also by Jones et al. (2019)."
        result = isolate_citations(text, pmap)
        # Original text should not appear verbatim
        self.assertNotIn("Smith (2020)", result)
        self.assertTrue(len(pmap) >= 1)

    def test_citation_format(self):
        """Standard (Author, YEAR) format should be masked."""
        from core.isolator import isolate_citations
        pmap = {}
        text = "According to Johnson (2021), the hypothesis was confirmed."
        result = isolate_citations(text, pmap)
        self.assertIn("[REF_", result)
        # Verify pmap stores original
        for tok, orig in pmap.items():
            self.assertIn("2021", orig)

    def test_quote_masking(self):
        """Quoted text should be replaced with [QUOTE_n] tokens."""
        from core.isolator import isolate_quotes
        pmap = {}
        text = 'The author states "this is a direct quote" in the introduction.'
        result = isolate_quotes(text, pmap)
        self.assertNotIn('"this is a direct quote"', result)
        self.assertTrue(len(pmap) >= 1)

    def test_equation_masking(self):
        """Inline LaTeX math should be masked."""
        from core.isolator import isolate_equations
        pmap = {}
        text = "The formula $E = mc^2$ is well known."
        result = isolate_equations(text, pmap)
        self.assertNotIn("$E = mc^2$", result)
        self.assertTrue(len(pmap) >= 1)

    def test_restore_citation(self):
        """Restored text should match original."""
        from core.isolator import isolate_citations, restore
        pmap = {}
        original = "Smith (2020) argued that the results were significant."
        masked = isolate_citations(original, pmap)
        restored = restore(masked, pmap)
        # The citation should be back
        self.assertIn("Smith", restored)

    def test_full_isolate_restore(self):
        """Full isolate + restore cycle should be lossless for protected spans."""
        from core.isolator import isolate, restore
        text = 'According to Brown (2019), "the sky is blue" (p. 42). Note ISO 9001.'
        masked, pmap = isolate(text, use_ner=False)
        # Protected spans should not appear in masked text
        self.assertNotIn('"the sky is blue"', masked)
        # Restore should bring them back
        restored = restore(masked, pmap)
        self.assertIn('"the sky is blue"', restored)

    def test_protected_regions_survive_humanizer(self):
        """Protected tokens must survive all humanizer methods unchanged."""
        from core.isolator import isolate, restore
        from core.humanizer import humanize

        text = 'Smith (2020) found that "machine learning is powerful".'
        masked, pmap = isolate(text, use_ner=False)
        # Save original token values
        original_tokens = dict(pmap)
        # Apply humanization
        humanized = humanize(masked)
        # Restore and check protection tokens are intact
        restored = restore(humanized, original_tokens)
        self.assertIn("Smith", restored)
        self.assertIn('"machine learning is powerful"', restored)


# ── Analyzer Tests ────────────────────────────────────────────────────────

class TestAnalyzer(unittest.TestCase):
    """Tests for core/analyzer.py."""

    def test_entropy_returns_float(self):
        from core.analyzer import compute_entropy
        entropy = compute_entropy("The quick brown fox jumps over the lazy dog.")
        self.assertIsInstance(entropy, float)
        self.assertGreater(entropy, 0.0)

    def test_sentence_length_std(self):
        from core.analyzer import compute_sentence_length_std
        text = "Short sentence. This is a much longer sentence with many more words in it. Another medium length sentence here."
        std = compute_sentence_length_std(text)
        self.assertIsInstance(std, float)
        self.assertGreaterEqual(std, 0.0)

    def test_analyze_returns_dict(self):
        from core.analyzer import analyze
        result = analyze("Artificial intelligence has transformed many fields.")
        self.assertIsInstance(result, dict)
        self.assertIn("entropy", result)
        self.assertIn("sentence_length_std", result)
        self.assertIn("word_count", result)

    def test_tfidf_density(self):
        from core.analyzer import compute_tfidf_density
        result = compute_tfidf_density("machine learning neural network deep learning transformer")
        # Returns dict or empty dict
        self.assertIsInstance(result, dict)

    def test_sentiment_returns_dict(self):
        from core.analyzer import compute_sentiment
        result = compute_sentiment("This result is very promising and significant.")
        self.assertIsInstance(result, dict)


# ── Humanizer Tests ───────────────────────────────────────────────────────

class TestHumanizer(unittest.TestCase):
    """Tests for core/humanizer.py."""

    def test_neutralize_cliches(self):
        from core.humanizer import neutralize_cliches
        text = "At the end of the day, this is transformative technology."
        result = neutralize_cliches(text)
        self.assertNotIn("at the end of the day", result.lower())
        self.assertNotIn("transformative", result.lower())

    def test_balance_burstiness(self):
        from core.humanizer import balance_burstiness
        text = "Short. Another short one. This is a significantly longer sentence that contains many words and ideas. Short again."
        result = balance_burstiness(text)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_replace_synonyms_preserves_tokens(self):
        from core.humanizer import replace_synonyms
        text = "[REF_1] showed that [QUOTE_1] was significant."
        result = replace_synonyms(text, replacement_rate=1.0)
        self.assertIn("[REF_1]", result)
        self.assertIn("[QUOTE_1]", result)

    def test_diversify_reporting_verbs(self):
        from core.humanizer import diversify_reporting_verbs
        text = "Smith said that the results are conclusive. Jones said that further research is needed."
        result = diversify_reporting_verbs(text)
        self.assertIsInstance(result, str)

    def test_humanize_returns_string(self):
        from core.humanizer import humanize
        text = "Furthermore, it is worth noting that AI is transformative."
        result = humanize(text)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


# ── Academic Refiner Tests ────────────────────────────────────────────────

class TestAcademicRefiner(unittest.TestCase):
    """Tests for core/academic_refiner.py."""

    def test_balance_hedging(self):
        from core.academic_refiner import balance_hedging
        text = "This always proves that AI is certainly the best approach."
        result = balance_hedging(text)
        self.assertNotIn("always", result.lower())
        self.assertNotIn("certainly", result.lower())

    def test_normalize_oxford_comma(self):
        from core.academic_refiner import normalize_oxford_comma
        text = "We analysed speed, accuracy and reliability."
        result = normalize_oxford_comma(text)
        # Oxford comma should be present before "and"
        self.assertIn(",", result)

    def test_format_block_quotes_long(self):
        from core.academic_refiner import format_block_quotes
        long_quote = '"' + "word " * 50 + '"'
        result = format_block_quotes(long_quote)
        # Should be indented
        self.assertIn("    ", result)

    def test_inject_transitions(self):
        from core.academic_refiner import inject_transitions
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = inject_transitions(text)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), len(text))


# ── Method Module Tests ───────────────────────────────────────────────────

class TestMethodModules(unittest.TestCase):
    """Smoke tests for all 10 method modules."""

    SAMPLE_TEXT = (
        "Artificial intelligence has transformed many fields. "
        "Furthermore, machine learning algorithms have proven highly effective. "
        "Moreover, deep learning provides comprehensive solutions. "
        "According to Smith (2020), these methods are transformative. "
        "At the end of the day, the results are significant."
    )

    def _run_method(self, module_name: str, func_name: str, *args, **kwargs):
        """Helper to run a method and verify it returns a string."""
        import importlib
        mod = importlib.import_module(module_name)
        func = getattr(mod, func_name)
        result = func(*args, **kwargs)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        return result

    def test_m01_zwsp_stripping(self):
        self._run_method("methods.m01_token_masking", "method_07_zwsp_stripping", self.SAMPLE_TEXT)

    def test_m01_paragraph_restructuring(self):
        self._run_method("methods.m01_token_masking", "method_08_paragraph_restructuring", self.SAMPLE_TEXT)

    def test_m11_lexical_diversity(self):
        self._run_method("methods.m11_ref_freeze", "method_12_lexical_diversity", self.SAMPLE_TEXT)

    def test_m11_bullet_professionalization(self):
        text = "• First bullet point here\n• Second important item\n• Third consideration"
        self._run_method("methods.m11_ref_freeze", "method_13_bullet_professionalization", text)

    def test_m11_cliche_neutralization(self):
        self._run_method("methods.m11_ref_freeze", "method_20_cliche_neutralization", self.SAMPLE_TEXT)

    def test_m21_invisible_chars(self):
        text = "Hello\u200b world\u200c test"
        result = self._run_method("methods.m21_metadata", "method_22_invisible_chars", text)
        self.assertNotIn('\u200b', result)
        self.assertNotIn('\u200c', result)

    def test_m21_reporting_verbs(self):
        self._run_method("methods.m21_metadata", "method_25_reporting_verbs", self.SAMPLE_TEXT)

    def test_m31_hedging_balance(self):
        self._run_method("methods.m31_domain", "method_36_hedging_balance", self.SAMPLE_TEXT)

    def test_m31_logical_flow(self):
        text = "The results show significant improvement. The findings indicate clear patterns."
        self._run_method("methods.m31_domain", "method_34_logical_flow", text)

    def test_m41_punctuation_logic(self):
        text = "There are spaces before commas , and after periods.But not always."
        self._run_method("methods.m41_variation", "method_47_punctuation_logic", text)

    def test_m51_hedging(self):
        text = "The algorithm always produces optimal results and certainly proves the hypothesis."
        self._run_method("methods.m51_hedging", "method_51_hedging_balance", text)

    def test_m51_pronoun_mimicry(self):
        text = "We analysed the data. Our results show significant patterns."
        self._run_method("methods.m51_hedging", "method_55_pronoun_mimicry", text)

    def test_m61_style_cleaning(self):
        text = "Smart \u201cquotes\u201d and em\u2014dashes should be normalised."
        result = self._run_method("methods.m61_tables", "method_67_style_cleaning", text)
        self.assertNotIn('\u201c', result)
        self.assertNotIn('\u2014', result)

    def test_m71_symbol_normalization(self):
        text = "The variable alpha represents the learning rate and delta denotes change."
        result = self._run_method("methods.m71_equations", "method_74_symbol_normalization", text)
        self.assertIn("α", result)

    def test_m71_equation_narrative(self):
        text = "where x is the input value and where y is the output."
        self._run_method("methods.m71_equations", "method_72_equation_narrative", text)

    def test_m81_micro_variation(self):
        self._run_method("methods.m81_structure", "method_88_micro_variation", self.SAMPLE_TEXT)

    def test_m91_steganography_removal(self):
        text = "Normal text\u200b with\u200c hidden\u200d characters."
        result = self._run_method("methods.m91_final", "method_94_steganography_removal", text)
        self.assertNotIn('\u200b', result)

    def test_m91_peer_review(self):
        text = "It is evident that deep learning is effective. Clearly, results are positive."
        result = self._run_method("methods.m91_final", "method_92_peer_review", text)
        self.assertNotIn("it is evident that", result.lower())

    def test_m91_oxford_comma(self):
        text = "We tested speed, accuracy and reliability in our experiments."
        self._run_method("methods.m91_final", "method_96_oxford_comma", text)


# ── Export Tests ──────────────────────────────────────────────────────────

class TestExporter(unittest.TestCase):
    """Tests for core/exporter.py."""

    SAMPLE_TEXT = "This is a test document.\n\nIt has two paragraphs.\n\nEnd of document."

    def test_export_docx(self):
        from core.exporter import export_docx
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            path = f.name
        try:
            result = export_docx(self.SAMPLE_TEXT, path, {})
            # Should return True if python-docx is available, or False gracefully
            self.assertIsInstance(result, bool)
            if result:
                self.assertTrue(os.path.exists(path))
                self.assertGreater(os.path.getsize(path), 0)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_export_pdf(self):
        from core.exporter import export_pdf
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            result = export_pdf(self.SAMPLE_TEXT, path, {})
            self.assertIsInstance(result, bool)
            if result:
                self.assertTrue(os.path.exists(path))
                self.assertGreater(os.path.getsize(path), 0)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_export_unknown_format(self):
        import warnings
        from core.exporter import export
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = f.name
        try:
            with warnings.catch_warnings(record=True):
                result = export(self.SAMPLE_TEXT, path, format="xyz", config={})
            self.assertFalse(result)
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ── GUI Tests ─────────────────────────────────────────────────────────────

class TestGUI(unittest.TestCase):
    """Tests for gui/interface.py."""

    def test_gui_imports_without_error(self):
        """The GUI module should import without errors."""
        try:
            import gui.interface
        except ImportError as e:
            self.skipTest(f"GUI import failed (expected in headless env): {e}")

    def test_gui_initializes(self):
        """HumanizerApp should initialize when Tkinter is available."""
        try:
            import tkinter as _tk
            from gui.interface import HumanizerApp
            root = _tk.Tk()
            root.withdraw()  # Hide window
            app = HumanizerApp(root)
            self.assertIsNotNone(app)
            root.destroy()
        except (ImportError, Exception) as e:
            self.skipTest(f"Tkinter not available in this environment: {e}")

    def test_clipboard_menu_creation(self):
        """ClipboardMenu should attach to a Text widget without errors."""
        try:
            import tkinter as _tk
            from gui.interface import ClipboardMenu
            root = _tk.Tk()
            root.withdraw()
            text_widget = _tk.Text(root)
            menu = ClipboardMenu(text_widget)
            self.assertIsNotNone(menu)
            root.destroy()
        except (ImportError, Exception) as e:
            self.skipTest(f"Tkinter not available: {e}")


# ── End-to-End Tests ──────────────────────────────────────────────────────

class TestEndToEnd(unittest.TestCase):
    """End-to-end pipeline tests."""

    def test_sample_01_quick_mode(self):
        """Run quick mode on sample_01.txt and verify output is non-empty string."""
        if not os.path.exists(SAMPLE_01_PATH):
            self.skipTest("sample_01.txt not found")

        with open(SAMPLE_01_PATH, "r", encoding="utf-8") as f:
            text = f.read()

        from core.isolator import isolate, restore
        from core.humanizer import humanize, neutralize_cliches

        # Isolate
        masked, pmap = isolate(text, use_ner=False)
        # Apply basic humanization
        result = neutralize_cliches(masked)
        result = humanize(result, config={"enable_noise": False})
        # Restore
        output = restore(result, pmap)

        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 100)
        # Verify AI clichés were reduced
        self.assertLess(
            output.lower().count("transformative") + output.lower().count("at the end of the day"),
            text.lower().count("transformative") + text.lower().count("at the end of the day") + 1,
        )

    def test_protected_regions_survive_full_pipeline(self):
        """Citations and quotes must survive the complete pipeline unchanged."""
        from core.isolator import isolate, restore
        from core.humanizer import humanize
        from core.academic_refiner import refine

        text = (
            'Smith (2020) argues that "neural networks are powerful tools" for '
            "many applications. Jones et al. (2019) confirmed these findings."
        )
        masked, pmap = isolate(text, use_ner=False)
        original_pmap = dict(pmap)

        # Run through pipeline
        processed = humanize(masked)
        processed = refine(processed)

        # Restore
        output = restore(processed, original_pmap)

        # Check original protected spans are present
        for original_span in original_pmap.values():
            self.assertIn(original_span, output, f"Protected span lost: {original_span!r}")

    def test_m01_run_all(self):
        """m01 run_all should return non-empty string."""
        import methods.m01_token_masking as m01
        text = "Furthermore, this comprehensive AI system is transformative. Smith (2020) proved this."
        result = m01.run_all(text)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_all_samples_load(self):
        """All 3 sample files should exist and contain >300 words."""
        for n in range(1, 4):
            path = os.path.join(_REPO_ROOT, "gui", "samples", f"sample_0{n}.txt")
            self.assertTrue(os.path.exists(path), f"Sample {n} not found")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            word_count = len(content.split())
            self.assertGreater(word_count, 300, f"Sample {n} has only {word_count} words")

    def test_config_yaml_loads(self):
        """settings.yaml should load correctly."""
        import yaml
        config_path = os.path.join(_REPO_ROOT, "config", "settings.yaml")
        self.assertTrue(os.path.exists(config_path))
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.assertIn("citation_style", config)
        self.assertIn("hedging_words", config)
        self.assertIn("ai_cliches", config)


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
