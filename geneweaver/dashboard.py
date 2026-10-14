"""Textual terminal dashboard for interactive FASTA comparisons."""

from math import ceil
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, Label, ProgressBar, Select, Static

from geneweaver.benchmark import format_seconds
from geneweaver.cpu_engine import compare_cpu
from geneweaver.gpu_engine import GPUUnavailableError, compare_gpu, cuda_available
from geneweaver.parser import FastaInputError, parse_fasta_file

DASHBOARD_CHUNK_SIZE = 4096


class GeneWeaverApp(App[None]):
    """A small path-based interface for comparing query FASTA records."""

    TITLE = "GeneWeaver — DNA comparison"

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="form"):
            yield Label("Reference FASTA")
            yield Input(
                value="sample_data/reference.fasta",
                placeholder="Path to one reference FASTA record",
                id="reference-path",
            )
            yield Label("Query FASTA")
            yield Input(
                value="sample_data/queries.fasta",
                placeholder="Path to one or more query records",
                id="query-path",
            )
            with Horizontal():
                yield Select(
                    [("Automatic (GPU if available)", "auto"), ("CPU", "cpu"), ("GPU", "gpu")],
                    value="auto",
                    id="execution-mode",
                    allow_blank=False,
                )
                yield Button("Compare", variant="primary", id="compare")
            yield ProgressBar(total=1, show_eta=False, id="progress")
            yield Static("Enter FASTA paths and choose Compare.", id="status")
            yield Static("", id="results", markup=False)
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "compare":
            self.run_comparison(
                self.query_one("#reference-path", Input).value,
                self.query_one("#query-path", Input).value,
                str(self.query_one("#execution-mode", Select).value),
            )

    @work(thread=True, exclusive=True)
    def run_comparison(
        self, reference_path: str, query_path: str, requested_mode: str
    ) -> None:
        """Run file loading and comparisons away from the UI event loop."""
        self.call_from_thread(self._begin_progress)

        try:
            references = parse_fasta_file(Path(reference_path))
            queries = parse_fasta_file(Path(query_path))
            if len(references) != 1:
                raise FastaInputError(
                    f"Reference FASTA must contain exactly one record; "
                    f"found {len(references)}."
                )

            reference = references[0]
            mode = requested_mode
            fallback_note = ""
            if mode == "auto":
                mode = "gpu" if cuda_available() else "cpu"
                if mode == "cpu":
                    fallback_note = "CUDA unavailable; CPU fallback selected. "
            elif mode == "gpu" and not cuda_available():
                mode = "cpu"
                fallback_note = "CUDA unavailable; CPU fallback selected. "

            chunks_total = sum(
                ceil(max(len(reference.sequence), len(query.sequence)) / DASHBOARD_CHUNK_SIZE)
                for query in queries
            )
            self.call_from_thread(
                self._set_progress_total, max(chunks_total, 1), len(queries)
            )
            lines = [
                f"Reference: {reference.identifier} ({len(reference.sequence)} bases)",
                f"Sequences: {len(queries)} | chunks processed: {chunks_total} "
                f"| mode: {mode.upper()}",
            ]
            if fallback_note:
                lines.append(fallback_note.strip())

            for query in queries:
                if mode == "gpu":
                    result, elapsed = compare_gpu(
                        reference.sequence,
                        query.sequence,
                        chunk_size=DASHBOARD_CHUNK_SIZE,
                    )
                else:
                    result, elapsed = compare_cpu(
                        reference.sequence,
                        query.sequence,
                        chunk_size=DASHBOARD_CHUNK_SIZE,
                    )
                query_chunks = ceil(
                    max(len(reference.sequence), len(query.sequence))
                    / DASHBOARD_CHUNK_SIZE
                )
                positions = ", ".join(map(str, result.mismatch_positions)) or "none"
                lines.append(
                    f"{query.identifier}: length={result.query_length}, "
                    f"mismatches={result.mismatch_count}, positions: {positions}, "
                    f"time={format_seconds(elapsed)}"
                )
                self.call_from_thread(self._advance_progress, query_chunks)

            self.call_from_thread(self._show_results, "\n".join(lines))
        except (FastaInputError, GPUUnavailableError, MemoryError, RuntimeError, ValueError) as error:
            self.call_from_thread(self._show_error, str(error))

    def _begin_progress(self) -> None:
        """Reset the progress display before a new job starts."""
        self.query_one("#progress", ProgressBar).update(total=1, progress=0)
        self.query_one("#results", Static).update("")
        self.query_one("#status", Static).update("Loading FASTA files...")

    def _set_progress_total(self, total_chunks: int, query_count: int) -> None:
        self.query_one("#progress", ProgressBar).update(total=total_chunks, progress=0)
        self.query_one("#status", Static).update(
            f"Reference loaded. {query_count} query sequence(s); "
            f"{total_chunks} chunk(s) to process."
        )

    def _advance_progress(self, chunk_count: int) -> None:
        self.query_one("#progress", ProgressBar).advance(chunk_count)

    def _show_results(self, text: str) -> None:
        self.query_one("#results", Static).update(text)
        self.query_one("#status", Static).update("Comparison complete.")

    def _show_error(self, message: str) -> None:
        self.query_one("#status", Static).update(f"Error: {message}")


if __name__ == "__main__":
    GeneWeaverApp().run()
