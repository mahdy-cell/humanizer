"""
gui/interface.py — Tkinter-based GUI for the Academic Text Humanizer

Features:
  - Left panel: scrollable input text area with full clipboard support
  - Right panel: scrollable output text area with same clipboard features
  - Middle controls: style selector, language, processing mode, domain field
  - Progress bar and status label
  - Export buttons (.docx, PDF)
  - Load sample button (3 academic samples)
  - Right-click context menu
  - Drag-and-drop file support
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")


# ── Clipboard context menu ────────────────────────────────────────────────

class ClipboardMenu:
    """Right-click context menu providing Copy/Paste/Cut/Select All."""

    def __init__(self, widget: tk.Text):
        self.widget = widget
        self.menu = tk.Menu(widget, tearoff=0)
        self.menu.add_command(label="Cut", accelerator="Ctrl+X", command=self._cut)
        self.menu.add_command(label="Copy", accelerator="Ctrl+C", command=self._copy)
        self.menu.add_command(label="Paste", accelerator="Ctrl+V", command=self._paste)
        self.menu.add_separator()
        self.menu.add_command(label="Select All", accelerator="Ctrl+A", command=self._select_all)
        widget.bind("<Button-3>", self._show_menu)
        # Keyboard bindings
        widget.bind("<Control-c>", lambda e: self._copy())
        widget.bind("<Control-v>", lambda e: self._paste())
        widget.bind("<Control-x>", lambda e: self._cut())
        widget.bind("<Control-a>", lambda e: self._select_all())

    def _show_menu(self, event):
        self.menu.tk_popup(event.x_root, event.y_root)

    def _copy(self):
        try:
            self.widget.event_generate("<<Copy>>")
        except Exception:
            pass

    def _paste(self):
        try:
            self.widget.event_generate("<<Paste>>")
        except Exception:
            pass

    def _cut(self):
        try:
            self.widget.event_generate("<<Cut>>")
        except Exception:
            pass

    def _select_all(self):
        self.widget.tag_add("sel", "1.0", "end")
        self.widget.mark_set("insert", "1.0")
        self.widget.see("insert")
        return "break"


# ── Main Application Window ───────────────────────────────────────────────

class HumanizerApp:
    """Main GUI application for the Academic Text Humanizer."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Academic Text Humanizer — 100-Method System")
        self.root.geometry("1280x800")
        self.root.minsize(900, 600)

        self._processing = False
        self._setup_styles()
        self._build_ui()

    # ── UI Setup ──────────────────────────────────────────────────────────

    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("Header.TLabel", font=("Helvetica", 12, "bold"))
        style.configure("Process.TButton", font=("Helvetica", 11, "bold"), padding=8)
        style.configure("TCombobox", font=("Helvetica", 10))

    def _build_ui(self):
        """Build the complete UI layout."""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ── Main frame ─────────────────────────────────────────────────
        main_frame = ttk.Frame(self.root, padding="8")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=3)
        main_frame.rowconfigure(1, weight=1)

        # ── Title bar ──────────────────────────────────────────────────
        title_label = ttk.Label(
            main_frame,
            text="🎓 Academic Text Humanizer — 100-Method AI-to-Human Pipeline",
            style="Header.TLabel",
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 8))

        # ── Input panel (left) ─────────────────────────────────────────
        input_frame = ttk.LabelFrame(main_frame, text="Input Text", padding="4")
        input_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        input_frame.rowconfigure(0, weight=1)
        input_frame.columnconfigure(0, weight=1)

        self.input_text = scrolledtext.ScrolledText(
            input_frame,
            wrap=tk.WORD,
            font=("Times New Roman", 11),
            undo=True,
        )
        self.input_text.grid(row=0, column=0, sticky="nsew")
        self._input_clipboard = ClipboardMenu(self.input_text)

        # Input toolbar
        input_toolbar = ttk.Frame(input_frame)
        input_toolbar.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        ttk.Button(input_toolbar, text="📂 Open File", command=self._open_file).pack(side="left", padx=2)
        ttk.Button(input_toolbar, text="🗑 Clear", command=lambda: self.input_text.delete("1.0", "end")).pack(
            side="left", padx=2
        )
        self._input_word_count = ttk.Label(input_toolbar, text="Words: 0")
        self._input_word_count.pack(side="right", padx=4)
        self.input_text.bind("<KeyRelease>", self._update_word_counts)

        # ── Controls panel (middle) ────────────────────────────────────
        ctrl_frame = ttk.LabelFrame(main_frame, text="Controls", padding="8")
        ctrl_frame.grid(row=1, column=1, sticky="ns", padx=4)

        # Citation style
        ttk.Label(ctrl_frame, text="Citation Style:").pack(anchor="w", pady=(0, 2))
        self.style_var = tk.StringVar(value="APA 7")
        style_combo = ttk.Combobox(
            ctrl_frame,
            textvariable=self.style_var,
            values=["APA 7", "Chicago", "IEEE"],
            state="readonly",
            width=14,
        )
        style_combo.pack(fill="x", pady=(0, 8))

        # Language
        ttk.Label(ctrl_frame, text="Language:").pack(anchor="w", pady=(0, 2))
        self.lang_var = tk.StringVar(value="English")
        lang_combo = ttk.Combobox(
            ctrl_frame,
            textvariable=self.lang_var,
            values=["English", "Arabic + English"],
            state="readonly",
            width=14,
        )
        lang_combo.pack(fill="x", pady=(0, 8))

        # Processing mode
        ttk.Label(ctrl_frame, text="Processing Mode:").pack(anchor="w", pady=(0, 2))
        self.mode_var = tk.StringVar(value="Standard (1–50)")
        mode_combo = ttk.Combobox(
            ctrl_frame,
            textvariable=self.mode_var,
            values=["Quick (1–10)", "Standard (1–50)", "Full (All 100)"],
            state="readonly",
            width=14,
        )
        mode_combo.pack(fill="x", pady=(0, 8))

        # Domain terms
        ttk.Label(ctrl_frame, text="Domain Terms:").pack(anchor="w", pady=(0, 2))
        self.domain_var = tk.StringVar()
        ttk.Entry(ctrl_frame, textvariable=self.domain_var, width=16).pack(fill="x", pady=(0, 8))

        # Author / University
        ttk.Label(ctrl_frame, text="Author Name:").pack(anchor="w", pady=(0, 2))
        self.author_var = tk.StringVar()
        ttk.Entry(ctrl_frame, textvariable=self.author_var, width=16).pack(fill="x", pady=(0, 8))

        ttk.Label(ctrl_frame, text="University:").pack(anchor="w", pady=(0, 2))
        self.university_var = tk.StringVar()
        ttk.Entry(ctrl_frame, textvariable=self.university_var, width=16).pack(fill="x", pady=(0, 8))

        ttk.Separator(ctrl_frame).pack(fill="x", pady=8)

        # Process button
        self.process_btn = ttk.Button(
            ctrl_frame,
            text="▶ HUMANIZE",
            style="Process.TButton",
            command=self._start_processing,
        )
        self.process_btn.pack(fill="x", pady=(0, 4))

        # Stop button
        self.stop_btn = ttk.Button(
            ctrl_frame,
            text="⏹ Stop",
            command=self._stop_processing,
            state="disabled",
        )
        self.stop_btn.pack(fill="x", pady=(0, 8))

        ttk.Separator(ctrl_frame).pack(fill="x", pady=8)

        # Load sample buttons
        ttk.Label(ctrl_frame, text="Load Sample:").pack(anchor="w", pady=(0, 2))
        for i in range(1, 4):
            labels = ["CS/AI Paper", "Medical Paper", "Social Sciences"]
            ttk.Button(
                ctrl_frame,
                text=f"Sample {i}: {labels[i-1]}",
                command=lambda n=i: self._load_sample(n),
            ).pack(fill="x", pady=1)

        ttk.Separator(ctrl_frame).pack(fill="x", pady=8)

        # Export buttons
        ttk.Label(ctrl_frame, text="Export:").pack(anchor="w", pady=(0, 2))
        ttk.Button(ctrl_frame, text="💾 Export .docx", command=self._export_docx).pack(fill="x", pady=1)
        ttk.Button(ctrl_frame, text="📄 Export PDF", command=self._export_pdf).pack(fill="x", pady=1)

        # ── Output panel (right) ───────────────────────────────────────
        output_frame = ttk.LabelFrame(main_frame, text="Humanized Output", padding="4")
        output_frame.grid(row=1, column=2, sticky="nsew", padx=(4, 0))
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)

        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            font=("Times New Roman", 11),
        )
        self.output_text.grid(row=0, column=0, sticky="nsew")
        self._output_clipboard = ClipboardMenu(self.output_text)

        # Output toolbar
        output_toolbar = ttk.Frame(output_frame)
        output_toolbar.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(
            output_toolbar, text="🗑 Clear", command=lambda: self.output_text.delete("1.0", "end")
        ).pack(side="left", padx=2)
        self._output_word_count = ttk.Label(output_toolbar, text="Words: 0")
        self._output_word_count.pack(side="right", padx=4)
        self.output_text.bind("<KeyRelease>", self._update_word_counts)

        # ── Status bar (bottom) ────────────────────────────────────────
        status_frame = ttk.Frame(main_frame, padding="4")
        status_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        status_frame.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Ready. Paste or load text to begin.")
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=0, sticky="w")

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            status_frame,
            variable=self.progress_var,
            maximum=100,
            length=300,
        )
        self.progress_bar.grid(row=0, column=1, sticky="e", padx=(8, 0))

        # ── Drag and drop ──────────────────────────────────────────────
        self._setup_drag_drop()

    def _setup_drag_drop(self):
        """Configure drag-and-drop for .txt and .docx files."""
        try:
            self.input_text.drop_target_register("DND_Files")  # type: ignore
            self.input_text.dnd_bind("<<Drop>>", self._on_drop)  # type: ignore
        except Exception:
            # tkinterdnd2 not available; skip
            pass

    def _on_drop(self, event):
        path = event.data.strip("{}")
        self._load_file(path)

    # ── File I/O ──────────────────────────────────────────────────────────

    def _open_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("Word documents", "*.docx"), ("All files", "*.*")]
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str):
        try:
            if path.endswith(".docx"):
                try:
                    from docx import Document
                    doc = Document(path)
                    content = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
                except ImportError:
                    content = f"[python-docx not available; cannot read {os.path.basename(path)}]"
            else:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            self.input_text.delete("1.0", "end")
            self.input_text.insert("1.0", content)
            self._update_word_counts()
            self.status_var.set(f"Loaded: {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Error", f"Could not load file:\n{exc}")

    def _load_sample(self, n: int):
        sample_path = os.path.join(SAMPLES_DIR, f"sample_0{n}.txt")
        if os.path.exists(sample_path):
            self._load_file(sample_path)
        else:
            messagebox.showwarning("Sample not found", f"Sample file not found:\n{sample_path}")

    # ── Processing ────────────────────────────────────────────────────────

    def _start_processing(self):
        if self._processing:
            return
        text = self.input_text.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("No Input", "Please enter or load text before processing.")
            return
        self._processing = True
        self.process_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress_var.set(0.0)
        self.status_var.set("Initialising pipeline…")
        self.output_text.delete("1.0", "end")

        thread = threading.Thread(target=self._run_pipeline, args=(text,), daemon=True)
        thread.start()

    def _stop_processing(self):
        self._processing = False
        self.status_var.set("Processing stopped by user.")
        self.process_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def _run_pipeline(self, text: str):
        """Run the humanization pipeline in a background thread."""
        try:
            config = self._build_config()
            mode = self.mode_var.get()

            def progress_cb(method_name: str, pct: float):
                if not self._processing:
                    raise InterruptedError("Processing stopped.")
                total_methods = 10 if "Quick" in mode else 50 if "Standard" in mode else 100
                # pct is per-method 0–100; scale to global 0–100
                self.root.after(0, self.status_var.set, f"{method_name} ({pct:.0f}%)")

            result = self._apply_methods(text, config, mode, progress_cb)

            self.root.after(0, self._finish_processing, result)
        except InterruptedError:
            pass
        except Exception as exc:
            self.root.after(0, messagebox.showerror, "Processing Error", str(exc))
            self.root.after(0, self._reset_controls)

    def _apply_methods(self, text: str, config: dict, mode: str, progress_cb) -> str:
        """Apply method groups based on selected processing mode."""
        import methods.m01_token_masking as m01
        import methods.m11_ref_freeze as m11
        import methods.m21_metadata as m21
        import methods.m31_domain as m31
        import methods.m41_variation as m41

        # Quick mode: methods 1–10 only
        # Standard mode: methods 1–50
        # Full mode: all 100

        masked, pmap = m01.method_01_token_masking(text, progress_cb)
        total_progress = [0.0]
        total_methods = 10 if "Quick" in mode else 50 if "Standard" in mode else 100

        def track(name, pct):
            progress_cb(name, pct)
            if pct >= 100:
                total_progress[0] = min(100, total_progress[0] + (100 / total_methods))
                self.root.after(0, self.progress_var.set, total_progress[0])

        masked = m01.method_02_entropy_variation(masked, pmap, track)
        masked = m01.method_04_burstiness_balancing(masked, track)
        masked = m01.method_05_domain_terminology(masked, config.get("domain_terms", []), track)
        masked = m01.method_06_active_passive_flip(masked, track)
        masked = m01.method_07_zwsp_stripping(masked, track)
        masked = m01.method_08_paragraph_restructuring(masked, track)

        if not "Quick" in mode:
            masked = m11.method_12_lexical_diversity(masked, track)
            masked = m11.method_13_bullet_professionalization(masked, track)
            masked = m11.method_14_syntactic_mimicry(masked, track)
            masked = m11.method_18_cross_paragraph_cohesion(masked, track)
            masked = m11.method_20_cliche_neutralization(masked, track)

            masked = m21.method_22_invisible_chars(masked, track)
            masked = m21.method_24_linguistic_noise(masked, progress=track)
            masked = m21.method_25_reporting_verbs(masked, track)
            masked = m21.method_26_connective_adjustment(masked, track)
            masked = m21.method_28_header_optimization(masked, track)

            masked = m31.method_32_table_humanization(masked, track)
            masked = m31.method_34_logical_flow(masked, track)
            masked = m31.method_36_hedging_balance(masked, track)
            masked = m31.method_37_block_quote_formatting(masked, track)

            masked = m41.method_41_length_variation(masked, track)
            masked = m41.method_44_adverbial_shifting(masked, track)
            masked = m41.method_46_academic_transitions(masked, track)
            masked = m41.method_47_punctuation_logic(masked, track)
            masked = m41.method_48_tense_consistency(masked, track)

        if "Full" in mode:
            import methods.m51_hedging as m51
            import methods.m61_tables as m61
            import methods.m71_equations as m71
            import methods.m81_structure as m81
            import methods.m91_final as m91

            masked = m51.method_51_hedging_balance(masked, track)
            masked = m51.method_53_rare_connectives(masked, track)
            masked = m51.method_55_pronoun_mimicry(masked, track)
            masked = m51.method_59_whitespace_correction(masked, track)

            masked = m61.method_61_numerical_sync(masked, track)
            masked = m61.method_62_caption_humanization(masked, track)
            masked = m61.method_64_acronym_safeguarding(masked, track)
            masked = m61.method_66_statistical_tone(masked, track)
            masked = m61.method_67_style_cleaning(masked, track)

            masked = m71.method_72_equation_narrative(masked, track)
            masked = m71.method_74_symbol_normalization(masked, track)
            masked = m71.method_78_citation_ordering(masked, track)

            masked = m81.method_86_intersentential_cohesion(masked, track)
            masked = m81.method_88_micro_variation(masked, track)
            masked = m81.method_89_bibliography_styling(masked, track)

            masked = m91.method_91_cluster_analysis(masked, track)
            masked = m91.method_92_peer_review(masked, track)
            masked = m91.method_93_cross_reference_formatting(masked, track)
            masked = m91.method_94_steganography_removal(masked, track)
            masked = m91.method_95_paragraph_burstiness(masked, track)
            masked = m91.method_96_oxford_comma(masked, track)

        from core.isolator import restore
        return restore(masked, pmap)

    def _finish_processing(self, result: str):
        self.output_text.insert("1.0", result)
        self._update_word_counts()
        self.progress_var.set(100.0)
        self.status_var.set("✅ Processing complete.")
        self._reset_controls()

    def _reset_controls(self):
        self._processing = False
        self.process_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    # ── Export ────────────────────────────────────────────────────────────

    def _export_docx(self):
        text = self.output_text.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("No Output", "Nothing to export. Run the humanizer first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".docx",
            filetypes=[("Word Document", "*.docx")],
            initialfile="humanized_output.docx",
        )
        if path:
            try:
                from core.exporter import export_docx
                config = self._build_config()
                success = export_docx(text, path, config)
                if success:
                    messagebox.showinfo("Export", f"Saved to:\n{path}")
                else:
                    messagebox.showwarning("Export", "Export completed with warnings (python-docx may not be installed).")
            except Exception as exc:
                messagebox.showerror("Export Error", str(exc))

    def _export_pdf(self):
        text = self.output_text.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("No Output", "Nothing to export. Run the humanizer first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF Document", "*.pdf")],
            initialfile="humanized_output.pdf",
        )
        if path:
            try:
                from core.exporter import export_pdf
                config = self._build_config()
                success = export_pdf(text, path, config)
                if success:
                    messagebox.showinfo("Export", f"Saved to:\n{path}")
                else:
                    messagebox.showwarning("Export", "Export completed with warnings (reportlab may not be installed).")
            except Exception as exc:
                messagebox.showerror("Export Error", str(exc))

    # ── Utilities ─────────────────────────────────────────────────────────

    def _build_config(self) -> dict:
        domain_raw = self.domain_var.get().strip()
        domain_terms = [t.strip() for t in domain_raw.split(",") if t.strip()] if domain_raw else []
        return {
            "citation_style": self.style_var.get().replace(" ", "").upper(),
            "language": self.lang_var.get(),
            "domain_terms": domain_terms,
            "author_name": self.author_var.get().strip(),
            "university_name": self.university_var.get().strip(),
            "margins_cm": 2.54,
            "font_name": "Times New Roman",
            "font_size": 12,
        }

    def _update_word_counts(self, event=None):
        in_words = len(self.input_text.get("1.0", "end-1c").split())
        out_words = len(self.output_text.get("1.0", "end-1c").split())
        self._input_word_count.config(text=f"Words: {in_words}")
        self._output_word_count.config(text=f"Words: {out_words}")


# ── Entry point ───────────────────────────────────────────────────────────

def launch():
    """Launch the GUI application."""
    root = tk.Tk()
    app = HumanizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
